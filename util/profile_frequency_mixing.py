"""Bounded standalone forward/backward profile at local or bottleneck shapes.

python -m util.profile_frequency_mixing --shape 35 35 64 --modes 16 16 16 --out profile.json
CPU reports whole-process peak RSS; CUDA reports peak allocated tensor memory.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from spectral_mixing_operator import FrequencyMixingOperator, FourierMultiplier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs="+", type=int, required=True)
    parser.add_argument("--modes", nargs="+", type=int, required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads) <= 0:
        parser.error("steps, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(31)
    device = torch.device(args.device)
    if args.baseline:
        op = FourierMultiplier(args.channels, args.modes).to(device)
    else:
        op = FrequencyMixingOperator(args.channels, len(args.shape), args.modes,
                                     mixing_rank=args.rank).to(device)
    x = torch.randn(args.batch_size, args.channels, *args.shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(op.parameters(), lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        op(x).square().mean().backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step:
            times.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(shape=args.shape, modes=args.modes, channels=args.channels,
                  batch_size=args.batch_size, rank=args.rank, baseline=args.baseline,
                  device=str(device), torch_version=torch.__version__,
                  parameters=sum(p.numel() for p in op.parameters()),
                  step_seconds=times, mean_step_seconds=sum(times) / len(times),
                  cpu_process_peak_rss_mib=rss / (1024**2 if sys.platform == "darwin" else 1024),
                  cuda_peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 1024**2
                                           if device.type == "cuda" else None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
