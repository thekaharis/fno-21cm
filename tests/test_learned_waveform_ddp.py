"""Shared branch banks must accumulate the same gradients under DDP."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from learned_waveform_operator import waveform_parameter_groups
from models_zre_2d import LocalFNO2d
from waveform_training import WaveformTrainingConfig, WaveformTrainingController


def _worker(rank, init_file, mode):
    torch.set_num_threads(1)
    loopback = next(name for _, name in socket.if_nameindex() if name in {"lo", "lo0"})
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(29)
        model = LocalFNO2d(in_channels=2, base_width=4, spectral_rank=2,
                           local_window=(4, 4), local_modes=(3, 3), global_modes=(3, 3),
                           local_operator="waveform", global_operator="waveform",
                           local_operator_kwargs={"bins": 7}, global_operator_kwargs={"bins": 9},
                           patch_chunk_size=1000).double()
        reference = copy.deepcopy(model)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        optim = torch.optim.Adam(waveform_parameter_groups(wrapped, lr=.001, weight_decay=0.))
        ref_optim = torch.optim.Adam(waveform_parameter_groups(reference, lr=.001, weight_decay=0.))
        config = WaveformTrainingConfig(mode, waveform_epochs=1, kernel_epochs=1)
        controller = WaveformTrainingController(wrapped, optim, config)
        ref_controller = WaveformTrainingController(reference, ref_optim, config)
        x = torch.randn(2, 2, 12, 12, dtype=torch.float64)
        y = torch.randn(2, 1, 12, 12, dtype=torch.float64)
        for epoch in range(4):
            controller.begin_epoch(epoch)
            ref_controller.begin_epoch(epoch)
            controller.prepare_batch()
            ref_controller.prepare_batch()
            optim.zero_grad()
            ref_optim.zero_grad()
            (wrapped(x[rank:rank + 1]) - y[rank:rank + 1]).square().mean().backward()
            (reference(x) - y).square().mean().backward()
            for p, q in zip(model.parameters(), reference.parameters()):
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)
            optim.step()
            ref_optim.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-8)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["joint", "waveform_only", "alternating"])
def test_two_rank_gradients_match_full_batch_with_shared_bottleneck(tmp_path, mode):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), mode), nprocs=2, join=True)
