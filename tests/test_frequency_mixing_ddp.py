"""Prepared factors shared by patches must participate correctly in DDP."""
import copy
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models_zre_2d import LocalFNO2d


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    loopback = next(name for _, name in socket.if_nameindex() if name in {"lo", "lo0"})
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(15)
        model = LocalFNO2d(in_channels=2, base_width=4, spectral_rank=2,
                           local_window=(4, 4), local_modes=(1, 1), global_modes=(1, 1),
                           local_operator="frequency_mixing", global_operator="frequency_mixing",
                           local_operator_kwargs={"mixing_rank": 3, "hidden_dim": 7},
                           global_operator_kwargs={"mixing_rank": 3, "hidden_dim": 7},
                           patch_chunk_size=8).double()
        reference = copy.deepcopy(model)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
        x = torch.randn(2, 2, 12, 12, dtype=torch.float64)
        y = torch.randn(2, 1, 12, 12, dtype=torch.float64)
        for _ in range(3):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            (wrapped(x[rank:rank + 1]) - y[rank:rank + 1]).square().mean().backward()
            (reference(x) - y).square().mean().backward()
            for a, b in zip(model.parameters(), reference.parameters()):
                assert a.grad is not None and b.grad is not None
                torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)
            optimizer.step()
            reference_optimizer.step()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
    finally:
        dist.destroy_process_group()


def test_distributed_matches_full_batch(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
