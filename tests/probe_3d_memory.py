"""Peak GPU memory for the 3-D localop run, to decide if an A100 can hold it.

The 3-D configuration was written for 141 GB H200s and nothing in the pipeline
records peak memory, so whether it fits an 80 GB A100 has been guesswork. This
builds the model through the same path fno_21cm_3d.py uses, runs one real
forward and backward on a full 140x140x256 cube under the run's own loss, and
reports the peak.

Also reports free space in the job's local scratch: the training script stages a
180 GB cube cache there, which is a separate failure mode from VRAM.

Run: sbatch slurm/probe_3d_memory.sbatch
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import h5py
import numpy as np
import torch

from dataset import paths
from losses import ExponentialWallDistance
from modeling import ModelConfig, TrainerModel, build_3d_model

IN_CHANNELS = 13          # density + 1/(1+z) + 11 parameters, as in the 3-D run


def main() -> None:
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {name}  ({total:.1f} GiB total)\n")

    tmp = os.environ.get("TMPDIR", "/tmp")
    try:
        du = shutil.disk_usage(tmp)
        print(f"local scratch {tmp}: {du.free / 2**30:.0f} GiB free of "
              f"{du.total / 2**30:.0f} GiB  "
              f"(staging needs ~180 GiB)")
    except OSError as exc:
        print(f"local scratch {tmp}: unavailable ({exc})")

    cfg = ModelConfig.from_env()
    fno = build_3d_model(cfg, IN_CHANNELS)
    model = TrainerModel(fno).to(dev)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel: {cfg.kind} ({os.environ.get('LOCAL_OPERATOR')} local / "
          f"{os.environ.get('GLOBAL_OPERATOR')} global) -> {n:,} parameters")

    # a real cube, so the shape and value distribution are the training ones
    with h5py.File(paths.CUBES, "r") as f:
        y = np.asarray(f["neutral_fraction"][0], dtype=np.float32)   # (140,140,256)
        d = np.asarray(f["density"][0], dtype=np.float32)
    D, H, W = y.shape
    print(f"cube: {D}x{H}x{W}")

    # channel 0 density, channel 1 the 1/(1+z) ramp, rest constant parameters
    x = torch.zeros(1, IN_CHANNELS, D, H, W, device=dev)
    x[0, 0] = torch.as_tensor(d, device=dev) / 10.0
    x[0, 1] = torch.linspace(1 / 26, 1 / 6, W, device=dev).view(1, 1, W)
    x[0, 2:] = 0.5
    target = torch.as_tensor(y, device=dev).view(1, 1, D, H, W)

    loss_fn = ExponentialWallDistance(
        scale=float(os.environ.get("EXPWALL_SCALE", "16.0")),
        cap=int(os.environ.get("WALL_CAP", "32")))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()
    stage = {}
    out = model(x)
    stage["after forward"] = torch.cuda.max_memory_allocated() / 2**30
    loss = loss_fn(out, target)
    stage["after loss"] = torch.cuda.max_memory_allocated() / 2**30
    loss.backward()
    stage["after backward"] = torch.cuda.max_memory_allocated() / 2**30
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    stage["after optimizer step"] = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.synchronize()

    print(f"\nloss = {float(loss.detach()):.6f}   out {tuple(out.shape)}")
    print(f"\n{'stage':<24} {'peak GiB':>9}")
    for k, v in stage.items():
        print(f"{k:<24} {v:9.2f}")
    peak = max(stage.values())
    print(f"\npeak {peak:.2f} GiB of {total:.1f} GiB "
          f"({100 * peak / total:.1f}%)")
    # Adam holds two moment buffers, allocated on the first step; a second step
    # is where a marginal fit actually breaks.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), target).backward()
        opt.step()
    torch.cuda.synchronize()
    steady = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak after three steps (optimizer state resident): {steady:.2f} GiB "
          f"({100 * steady / total:.1f}%)")
    print(f"\nverdict: {'FITS' if steady < 0.85 * total else 'TOO TIGHT'} "
          f"on {name}")


if __name__ == "__main__":
    main()
