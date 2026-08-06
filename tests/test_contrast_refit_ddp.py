"""The contrast refit under DDP.

The property under test is that every rank trains the next epoch under the
*same* map. Each rank sees a different shard, so an unsynced fit gives each one
a different table -- silently, since only rank 0 prints.

Workers assert internally; ``mp.spawn`` re-raises their failures. Every process
group carries a short timeout so a collective that only some ranks reach fails
the test instead of hanging it.
"""

from __future__ import annotations

import os
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from contrast import ContrastComposed
from util import contrast_refit as refit

TIMEOUT = timedelta(seconds=30)


def _loopback_interface() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    raise RuntimeError(f"no loopback network interface found in {sorted(names)}")


def _join(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface())
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world_size, timeout=TIMEOUT)


class _Base(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid((x[:, :1] - 0.5) * 4.0)


def _model():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fno = ContrastComposed(_Base(), "xhi", schedule_kind="stepped",
                                        n_bins=8, key_mode="monotone")
    return _M()


def _shard(rank: int, n_cubes: int = 2, n_los: int = 16):
    """A different shard per rank -- otherwise the test proves nothing."""
    g = torch.Generator().manual_seed(100 + rank)
    return [{"x": torch.rand(1, 1, 4, 4, n_los, generator=g),
             "y": (torch.rand(1, 1, 4, 4, n_los, generator=g) > 0.5).float()}
            for _ in range(n_cubes)]


# ---------------------------------------------------------------- workers

def _identical_tables_worker(rank: int, world_size: int, init_file: str) -> None:
    _join(rank, world_size, init_file)
    try:
        model = _model()
        stats = refit.refit_and_install(model, _shard(rank), "cpu",
                                        max_samples=32, steps=20, batch=1,
                                        sync=True)
        mine = torch.tensor(stats["thetas"], dtype=torch.float64)
        gathered = [torch.zeros_like(mine) for _ in range(world_size)]
        dist.all_gather(gathered, mine)
        for other in gathered:
            assert torch.equal(mine, other), (
                f"rank {rank} installed {mine.tolist()} but another rank "
                f"installed {other.tolist()}")
        # And the *installed* schedule, not just the reported dict.
        live = torch.tensor(
            [float(v) for v in model.fno.contrast.schedule.thetas().detach()],
            dtype=torch.float64)
        assert torch.allclose(live, mine, atol=1e-6)
    finally:
        dist.destroy_process_group()


def _global_counts_worker(rank: int, world_size: int, init_file: str) -> None:
    _join(rank, world_size, init_file)
    try:
        model = _model()
        stats = refit.refit_and_install(model, _shard(rank), "cpu",
                                        max_samples=32, steps=5, batch=1,
                                        sync=True)
        # 2 cubes x 16 LOS slices per rank, pooled over both ranks.
        assert stats["n_slices"] == 32 * world_size
        assert sum(stats["bin_counts"]) == 32 * world_size
        assert 0.0 <= stats["frac_band"] <= 1.0
    finally:
        dist.destroy_process_group()


def _sync_is_not_a_noop_worker(rank: int, world_size: int, init_file: str) -> None:
    """The synced table must reflect both shards, not just the local one."""
    _join(rank, world_size, init_file)
    try:
        alone = refit.refit_and_install(_model(), _shard(rank), "cpu",
                                        max_samples=32, steps=20, batch=1,
                                        sync=False)
        together = refit.refit_and_install(_model(), _shard(rank), "cpu",
                                           max_samples=32, steps=20, batch=1,
                                           sync=True)
        assert alone["thetas"] != together["thetas"], (
            "synced fit reproduced the rank-local fit exactly; the shards are "
            "probably identical, so this test proves nothing")
        assert alone["n_slices"] == 32
        assert together["n_slices"] == 32 * world_size
    finally:
        dist.destroy_process_group()


def _nonfinite_on_one_rank_worker(rank: int, world_size: int,
                                  init_file: str) -> None:
    """A NaN loss on a single rank must stop every rank, not deadlock them.

    Rank 1 breaks out of the fit immediately; rank 0's loss stays finite. If
    the break were decided locally, rank 0 would keep issuing collectives that
    rank 1 has already stopped reaching, and the group would hang until the
    timeout.
    """
    _join(rank, world_size, init_file)
    try:
        def objective(out, y):
            if rank == 1:
                return torch.tensor(float("nan"), requires_grad=True)
            return ((out - y) ** 2).mean()

        stats = refit.refit_and_install(_model(), _shard(rank), "cpu",
                                        max_samples=32, steps=50, batch=1,
                                        objective=objective, sync=True)
        # The schedule is untouched by a fit that never took a step, so it is
        # still the identity-initialised table, floored.
        assert all(v >= 0.25 for v in stats["thetas"])
        mine = torch.tensor(stats["thetas"], dtype=torch.float64)
        gathered = [torch.zeros_like(mine) for _ in range(world_size)]
        dist.all_gather(gathered, mine)
        assert all(torch.equal(mine, o) for o in gathered)
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------------ tests

def _run(worker, world_size: int = 2) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = Path(tmpdir) / "distributed-init"
        mp.spawn(worker, args=(world_size, str(init_file)), nprocs=world_size,
                 join=True)


def test_every_rank_installs_the_same_table() -> None:
    _run(_identical_tables_worker)


def test_reported_counts_are_global() -> None:
    _run(_global_counts_worker)


def test_syncing_actually_pools_the_shards() -> None:
    _run(_sync_is_not_a_noop_worker)


def test_a_nonfinite_loss_on_one_rank_does_not_deadlock() -> None:
    _run(_nonfinite_on_one_rank_worker)


def test_sync_is_inert_without_a_process_group() -> None:
    """Single-process runs must take exactly the old path."""
    assert refit._world(sync=True) == 1
    stats = refit.refit_and_install(_model(), _shard(0), "cpu", max_samples=32,
                                    steps=5, batch=1, sync=True)
    assert stats["n_slices"] == 32
    assert stats["rejected"] == 0.0
