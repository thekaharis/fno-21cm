from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fno_21cm_3d import _all_reduce_weighted_metrics


def _loopback_interface() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    raise RuntimeError(f"no loopback network interface found in {sorted(names)}")


def _reduction_worker(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface())
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_counts = (2, 1)
        local_metrics = (
            {"val_l2": 2.0, "val_h1": 4.0},
            {"val_l2": 8.0, "val_h1": 1.0},
        )
        reduced = _all_reduce_weighted_metrics(
            local_metrics[rank],
            local_sample_count=local_counts[rank],
            world_size=world_size,
            device="cpu",
        )
        assert reduced["val_l2"] == pytest.approx(4.0)
        assert reduced["val_h1"] == pytest.approx(3.0)
    finally:
        dist.destroy_process_group()


def test_weighted_metric_reduction_handles_uneven_rank_counts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = Path(tmpdir) / "distributed-init"
        mp.spawn(
            _reduction_worker,
            args=(2, str(init_file)),
            nprocs=2,
            join=True,
        )


def test_weighted_metric_reduction_is_noop_without_ddp() -> None:
    metrics = {
        "val_l2": torch.tensor(1.25),
        "val_h1": 2.5,
    }
    reduced = _all_reduce_weighted_metrics(
        metrics,
        local_sample_count=3,
        world_size=1,
        device="cpu",
    )
    assert reduced["val_l2"] == pytest.approx(1.25)
    assert reduced["val_h1"] == pytest.approx(2.5)
