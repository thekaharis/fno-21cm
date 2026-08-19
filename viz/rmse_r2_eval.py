#!/usr/bin/env python3
"""Voxel RMSE and R^2 over the test split, for any number of checkpoints.

Neither quantity is recoverable from the training logs. `val_l2` is neuralop's
*absolute* Lp norm -- a quadrature-weighted `(h^d * sum |u-v|^2)^(1/2)`, not a
root-mean-square -- and `val_l2_rel` divides by `||truth||`, whose origin is
zero rather than the truth mean, so it is not `1 - R^2` either. Both therefore
need a fresh pass over the predictions.

Definitions, stated because R^2 on a bimodal field is easy to misread:

    RMSE   = sqrt( SS_res / N )
    R^2    = 1 - SS_res / SS_tot,   SS_tot = sum (y - ybar)^2

`ybar` is the mean over *all* evaluated voxels of *this* split, so R^2 is
measured against the constant-mean predictor. x_HI is strongly bimodal (mass at
0 and ~1), which makes that constant predictor poor and R^2 correspondingly
flattering -- it is a relative score, not an accuracy.

Accumulation is streaming (sums only), so cost is one forward pass per cone and
memory does not grow with `--n-cones`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


class Accumulator:
    """Streaming sums for RMSE and R^2, plus a per-cone R^2 spread."""

    def __init__(self) -> None:
        self.n = 0
        self.ss_res = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.per_cone_r2: list[float] = []

    def update(self, truth: np.ndarray, pred: np.ndarray) -> None:
        t = truth.astype(np.float64).ravel()
        p = pred.astype(np.float64).ravel()
        err = p - t
        self.n += t.size
        self.ss_res += float(err @ err)
        self.sum_y += float(t.sum())
        self.sum_y2 += float(t @ t)
        tbar = t.mean()
        denom = float(((t - tbar) ** 2).sum())
        if denom > 0:
            self.per_cone_r2.append(1.0 - float(err @ err) / denom)

    def stats(self) -> dict:
        if self.n == 0:
            return {}
        ybar = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * ybar * ybar     # sum (y - ybar)^2
        rmse = (self.ss_res / self.n) ** 0.5
        r2 = 1.0 - self.ss_res / ss_tot if ss_tot > 0 else float("nan")
        pc = np.asarray(self.per_cone_r2, dtype=float)
        return {"rmse": rmse, "r2": r2, "mse": self.ss_res / self.n,
                "n_voxels": self.n, "truth_mean": ybar,
                "truth_std": (ss_tot / self.n) ** 0.5,
                "r2_per_cone_mean": float(pc.mean()) if pc.size else float("nan"),
                "r2_per_cone_p16": float(np.percentile(pc, 16)) if pc.size else float("nan"),
                "r2_per_cone_p84": float(np.percentile(pc, 84)) if pc.size else float("nan"),
                "n_cones": int(pc.size)}


def _parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def run(checkpoints: dict[str, str], n_cones: int, split: str) -> dict[str, dict]:
    import torch
    from dataset import paths
    from dataset.dataset_3d import (InputFeatures, LightconeCubeCache,
                                    ParameterNormalization, resolve_split)
    from modeling import ModelConfig
    from util.run_metadata import load_run_metadata
    from viz.visualize_3d import load_model, predict_cube

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
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split][:n_cones]
    print(f"[rmse_r2] {split} split: {len(rows)} cones")

    results = {}
    for name, ckpt in checkpoints.items():
        ck = Path(ckpt)
        meta = load_run_metadata(ck.parent)
        cfg = (ModelConfig.from_dict(meta["model_config"])
               if meta and "model_config" in meta else None)
        if meta and "input_features" in meta:
            got = meta["input_features"]["name"]
            if got != feats.name:
                raise ValueError(f"{ck} expects input features {got!r}, "
                                 f"dataset uses {feats.name!r}")
        model = load_model(in_channels=dataset.in_channels, checkpoint=ck,
                           model_config=cfg)
        acc = Accumulator()
        for j, r in enumerate(rows):
            _d, truth, pred = predict_cube(model, dataset[r])
            acc.update(truth, pred)
            if (j + 1) % 50 == 0 or j + 1 == len(rows):
                print(f"[rmse_r2] {name}: {j + 1}/{len(rows)} cones", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results[name] = acc.stats()
        s = results[name]
        print(f"[rmse_r2] {name}: RMSE={s['rmse']:.6f}  R2={s['r2']:.6f}")
    return results


def _selftest() -> int:
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) > 0.4).astype(float)       # bimodal like x_HI
    acc = Accumulator()
    acc.update(y, y)
    s = acc.stats()
    assert abs(s["rmse"]) < 1e-12 and abs(s["r2"] - 1.0) < 1e-12, s
    acc = Accumulator()
    acc.update(y, np.full_like(y, y.mean()))            # constant-mean predictor
    s = acc.stats()
    assert abs(s["r2"]) < 1e-9, s                       # R^2 = 0 by definition
    acc = Accumulator()
    acc.update(y, y + 0.1)
    s = acc.stats()
    assert abs(s["rmse"] - 0.1) < 1e-9, s
    print("[selftest] perfect -> R2=1, constant-mean -> R2=0, "
          "+0.1 offset -> RMSE=0.1: OK")
    return 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--n-cones", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("figures/rmse_r2"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())
    if not args.checkpoints:
        ap.error("--checkpoints is required (or use --selftest)")

    args.out.mkdir(parents=True, exist_ok=True)
    res = run(_parse_kv(args.checkpoints), args.n_cones, args.split)

    order = sorted(res.items(), key=lambda kv: kv[1]["rmse"])
    cols = ["model", "rmse", "r2", "mse", "r2_per_cone_mean",
            "r2_per_cone_p16", "r2_per_cone_p84", "n_cones", "n_voxels",
            "truth_mean", "truth_std"]
    with open(args.out / "rmse_r2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, s in order:
            w.writerow([name] + [s[c] for c in cols[1:]])
    with open(args.out / "rmse_r2.json", "w") as fh:
        json.dump({"split": args.split, "n_cones": args.n_cones,
                   "results": res}, fh, indent=2)

    print(f"\n{'rank':<5}{'model':<32}{'RMSE':>10}{'R^2':>10}"
          f"{'R^2/cone':>12}")
    for i, (name, s) in enumerate(order, 1):
        print(f"{i:<5}{name:<32}{s['rmse']:>10.6f}{s['r2']:>10.5f}"
              f"{s['r2_per_cone_mean']:>12.5f}")
    print(f"\nwrote {args.out}/rmse_r2.csv")


if __name__ == "__main__":
    main()
