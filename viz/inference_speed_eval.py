#!/usr/bin/env python3
"""Inference throughput for trained 3-D checkpoints, on real cubes.

Distinct from `legacy/probes/bench_operator_variants.py`, which times freshly
built *2-D slice* variants at matched width to isolate the operator cost. This
script times the **actual trained matrix checkpoints** on **whole 256^3-class
cubes**, so the numbers are directly comparable with the accuracy of the same
checkpoint and with its parameter count.

Method, and why each part is there:

* **Real cubes from the cache**, cycled, not a random tensor -- FFT/cuDNN plans
  and memory layout depend on the true shape, and the windowed local operators
  chunk over it.
* **Warmup iterations are discarded.** The first forward pass of an FNO builds
  FFT plans and cuDNN algorithm choices; including it overstates cost by a
  large factor and does so unevenly across architectures.
* **`torch.cuda.synchronize()` around every timed pass.** CUDA launches are
  asynchronous; timing without a sync measures queueing, not compute.
* **Median, not mean**, over the timed passes, with p16/p84 reported -- one
  descheduled iteration should not set the number.
* Only the forward pass is timed. Host transfer is excluded: it is identical
  for every model and would compress the spread.

Parameter counts are **complex-aware** -- a complex weight is two real
numbers. `numel()` reports one, which halves every FNO-family count and makes
Fourier and Walsh models incomparable.

Run:
    python -m viz.inference_speed_eval --checkpoints name=path ... \
        --out figures/final_eval/matrix/speed
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def _parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def count_params(model) -> tuple[int, int]:
    """(real_numbers, complex_weights) -- complex counts as two reals."""
    import torch
    real = 0
    cplx = 0
    for p in model.parameters():
        if torch.is_complex(p):
            real += p.numel() * 2
            cplx += p.numel()
        else:
            real += p.numel()
    return real, cplx


def time_model(model, samples, warmup: int, iters: int) -> dict:
    import torch
    device = next(model.parameters()).device
    xs = [s["x"].unsqueeze(0).to(device) for s in samples]

    with torch.no_grad():
        for i in range(warmup):
            model(x=xs[i % len(xs)])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for i in range(iters):
            x = xs[i % len(xs)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x=x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t = np.asarray(times, dtype=float)
    n_slices = int(xs[0].shape[-1])          # LOS slices per cube
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else float("nan"))
    med = float(np.median(t))
    return {"ms_per_cube": med * 1e3,
            "ms_p16": float(np.percentile(t, 16)) * 1e3,
            "ms_p84": float(np.percentile(t, 84)) * 1e3,
            "cubes_per_s": 1.0 / med,
            "slices_per_s": n_slices / med,
            "peak_mib": peak,
            "n_timed": int(t.size),
            "cube_shape": tuple(int(v) for v in xs[0].shape[1:])}


def run(checkpoints: dict[str, str], n_samples: int, warmup: int,
        iters: int, split: str) -> dict[str, dict]:
    import torch
    from dataset import paths
    from dataset.dataset_3d import (InputFeatures, LightconeCubeCache,
                                    ParameterNormalization, resolve_split)
    from modeling import ModelConfig
    from util.run_metadata import load_run_metadata
    from viz.visualize_3d import load_model

    cache = Path(os.environ.get("CUBES_CACHE", paths.CUBES))
    first = Path(next(iter(checkpoints.values())))
    first_meta = load_run_metadata(first.parent)
    feats = InputFeatures(first_meta["input_features"]["name"]
                          if first_meta and "input_features" in first_meta
                          else "density_z_params")
    dataset = LightconeCubeCache(cache, input_features=feats)
    if first_meta and first_meta.get("parameter_normalization"):
        dataset.set_parameter_normalization(
            ParameterNormalization.from_dict(first_meta["parameter_normalization"]))

    train_idx, val_idx, test_idx, _ = resolve_split(dataset, first_meta)
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split][:n_samples]
    samples = [dataset[r] for r in rows]
    print(f"[speed] {len(samples)} cubes from {split}; "
          f"warmup {warmup}, timed {iters}")

    results = {}
    for name, ckpt in checkpoints.items():
        ck = Path(ckpt)
        meta = load_run_metadata(ck.parent)
        cfg = (ModelConfig.from_dict(meta["model_config"])
               if meta and "model_config" in meta else None)
        model = load_model(in_channels=dataset.in_channels, checkpoint=ck,
                           model_config=cfg)
        model.eval()
        real, cplx = count_params(model)
        r = time_model(model, samples, warmup, iters)
        r["params"] = real
        r["params_complex_weights"] = cplx
        results[name] = r
        print(f"[speed] {name}: {r['ms_per_cube']:.1f} ms/cube  "
              f"{r['slices_per_s']:.1f} slices/s  {real:,} params  "
              f"peak {r['peak_mib']:.0f} MiB", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--n-samples", type=int, default=4,
                    help="distinct cubes cycled through (default 4)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("figures/inference_speed_3d"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    res = run(_parse_kv(args.checkpoints), args.n_samples, args.warmup,
              args.iters, args.split)

    import torch
    env = {"device": (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu"),
           "torch": torch.__version__,
           "host": platform.node(),
           "patch_chunk_size": os.environ.get("LOCALFNO_PATCH_CHUNK_SIZE"),
           "split": args.split, "warmup": args.warmup, "iters": args.iters}
    print(f"\ndevice: {env['device']}   "
          f"LOCALFNO_PATCH_CHUNK_SIZE={env['patch_chunk_size']}")

    order = sorted(res.items(), key=lambda kv: kv[1]["ms_per_cube"])
    cols = ["model", "params", "params_complex_weights", "ms_per_cube",
            "ms_p16", "ms_p84", "cubes_per_s", "slices_per_s", "peak_mib",
            "n_timed"]
    with open(args.out / "inference_speed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, s in order:
            w.writerow([name] + [s[c] for c in cols[1:]])
    with open(args.out / "inference_speed.json", "w") as fh:
        json.dump({"env": env, "results": res}, fh, indent=2)

    print(f"\n{'rank':<5}{'model':<30}{'ms/cube':>10}{'slices/s':>10}"
          f"{'params':>14}{'peak MiB':>10}")
    for i, (name, s) in enumerate(order, 1):
        print(f"{i:<5}{name:<30}{s['ms_per_cube']:>10.1f}"
              f"{s['slices_per_s']:>10.1f}{s['params']:>14,}"
              f"{s['peak_mib']:>10.0f}")
    print(f"\nwrote {args.out}/inference_speed.csv")


if __name__ == "__main__":
    main()
