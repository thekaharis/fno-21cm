#!/usr/bin/env python3
"""Inference throughput for trained multi-field checkpoints, on real cones.

Multi-field counterpart to viz/inference_speed_eval.py. The methodology is
deliberately identical so the numbers are comparable with the 3-D operator
matrix; what differs is that a multi-field model consumes several input fields
and emits several target fields, so throughput is reported per *cone* and
separately as input and output voxel rates.

Method, and why each part is there:

* **Real cones from the prepared cache**, cycled, not random tensors -- FFT and
  cuDNN plans depend on the true shape, and the windowed local operator chunks
  over it.
* **Warmup passes are discarded.** The first forward pass builds FFT plans and
  picks cuDNN algorithms; including it overstates cost, and unevenly across
  architectures.
* **`torch.cuda.synchronize()` around every timed pass.** CUDA launches are
  asynchronous, so timing without a sync measures queueing rather than compute.
* **Median over timed passes**, with p16/p84, so one descheduled iteration does
  not set the number.
* Forward pass only. Host transfer is excluded: it is identical across models
  and would compress the spread.

Parameter counts are **complex-aware** -- a complex weight is two real numbers,
and `numel()` reports one, which halves every FNO-family count.

Run:
    python -m viz.multifield_inference_speed \
        --checkpoints mf_cnn_fno=experiments/multifield/mf_cnn_fno_ep10_snapshot.pt \
        --out figures/3d_xhi/mf_cnn_fno/speed
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it!r}")
        name, path = it.split("=", 1)
        out[name] = path
    return out


def count_params(model):
    """(real_numbers, complex_weights) -- a complex weight is two reals."""
    real = cplx = 0
    for p in model.parameters():
        if torch.is_complex(p):
            real += p.numel() * 2
            cplx += p.numel()
        else:
            real += p.numel()
    return real, cplx


def time_model(model, samples, warmup, iters):
    device = next(model.parameters()).device
    xs = [s["x"].unsqueeze(0).to(device) for s in samples]

    with torch.no_grad():
        for i in range(warmup):
            model(xs[i % len(xs)])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for i in range(iters):
            x = xs[i % len(xs)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t = np.asarray(times, dtype=float)
    med = float(np.median(t))
    in_shape = tuple(int(v) for v in xs[0].shape[1:])
    out_shape = tuple(int(v) for v in out.shape[1:])
    spatial = int(np.prod(in_shape[1:]))
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else float("nan"))
    return {
        "ms_per_cone": med * 1e3,
        "ms_p16": float(np.percentile(t, 16)) * 1e3,
        "ms_p84": float(np.percentile(t, 84)) * 1e3,
        "cones_per_s": 1.0 / med,
        "in_voxels_per_s": in_shape[0] * spatial / med,
        "out_voxels_per_s": out_shape[0] * spatial / med,
        "peak_mib": peak,
        "n_timed": int(t.size),
        "in_shape": in_shape,
        "out_shape": out_shape,
    }


def time_native(model, dataset, rows, config, warmup, iters):
    """Whole-cone inference for native-LOS window models.

    For these models one forward pass covers a single window, so the honest
    per-cone cost is assembling the full native cone from overlapping windows
    (predict_native_cone), which is what a user of the model actually pays.
    That path copies each window's core to host memory, so unlike the full-grid
    timing this includes device-to-host transfer.
    """
    from dataset.los_windows import predict_native_cone
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(warmup):
            predict_native_cone(model, dataset, rows[i % len(rows)], config, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times, lengths = [], []
        for i in range(iters):
            row = rows[i % len(rows)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = predict_native_cone(model, dataset, row, config, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            lengths.append(int(out.shape[-1]))
    t = np.asarray(times, dtype=float)
    med = float(np.median(t))
    n_los = float(np.median(lengths))
    spatial = int(np.prod(dataset.transverse_shape)) * n_los
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else float("nan"))
    return {
        "ms_per_cone": med * 1e3,
        "ms_p16": float(np.percentile(t, 16)) * 1e3,
        "ms_p84": float(np.percentile(t, 84)) * 1e3,
        "cones_per_s": 1.0 / med,
        "in_voxels_per_s": dataset.in_channels * spatial / med,
        "out_voxels_per_s": dataset.out_channels * spatial / med,
        "peak_mib": peak,
        "n_timed": int(t.size),
        "in_shape": ("native", int(n_los)),
        "out_shape": ("native", int(n_los)),
    }


def run(checkpoints, split, n_samples, warmup, iters, device_arg):
    from fno_multifield import restore, choose_device
    device = choose_device(device_arg)
    if device.type != "cuda":
        print("WARNING: no CUDA device; numbers below are CPU and not "
              "comparable with any GPU measurement.", file=sys.stderr)
    rows = []
    for name, path in checkpoints.items():
        checkpoint, model, dataset, split_rows = restore(path, device)
        try:
            idx = [int(i) for i in split_rows[split][:n_samples]]
            real, cplx = count_params(model)
            mapping = checkpoint["metadata"]["mapping"]
            sampling = checkpoint["metadata"].get("sampling", {"mode": "full"})
            if sampling.get("mode", "full") != "full":
                from dataset.los_windows import LOSWindowConfig
                r = time_native(model, dataset, idx, LOSWindowConfig(**sampling),
                                warmup, iters)
                r["timing_mode"] = f"native whole-cone ({sampling['mode']})"
            else:
                samples = [dataset[i] for i in idx]
                r = time_model(model, samples, warmup, iters)
                r["timing_mode"] = "full-grid forward"
            r.update(name=name, checkpoint=str(path), params_real=real,
                     params_complex=cplx,
                     inputs=",".join(mapping["inputs"]),
                     targets=",".join(mapping["targets"]),
                     n_cones_timed=len(idx))
            rows.append(r)
            print(f"{name:18s} {r['ms_per_cone']:8.1f} ms/cone  "
                  f"{r['cones_per_s']:7.3f} cones/s  "
                  f"peak {r['peak_mib']:8.1f} MiB  params {real:,}")
        finally:
            dataset.close()
    return rows, device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--n-samples", type=int, default=4,
                    help="distinct cones cycled through the timed passes")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    # fno_multifield.choose_device expects the literal "auto" or a device
    # string; it does not accept None.
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows, device = run(parse_kv(args.checkpoints), args.split, args.n_samples,
                       args.warmup, args.iters, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.out / "multifield_speed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    meta = {"device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__, "platform": platform.platform(),
            "split": args.split, "warmup": args.warmup, "iters": args.iters,
            "n_samples": args.n_samples}
    (args.out / "multifield_speed_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out}/multifield_speed.csv")


if __name__ == "__main__":
    main()
