"""Optimal contrast theta (tau = 1/2) resolved against ionisation state.

Replaces four near-identical ad-hoc probes.  Measured three ways, because they
answer different questions:

  pred     the model's own output -- the practically relevant theta
  blur     truth low-passed to the model's transition width, so blur is the
           ONLY defect present: the positive control, where sharpening is
           unambiguously the right operation
  truth    the map applied to a perfect field: pure damage, the price of the
           map paid regardless of what the model does

Every gain is fitted on one set of cones and scored on another, with a
cone-level bootstrap CI.  ``edge_density`` is printed beside each gain because
near-empty bins produce large meaningless gains -- a contrast map "wins" there
by thresholding noise off a blank field, not by sharpening an edge.

    python tests/probe_contrast_theta.py --bins decile
    python tests/probe_contrast_theta.py --bins 0,0.05,0.10,...,0.36
    python tests/probe_contrast_theta.py --cache xhi_band_000_036.h5 --xhi-max 0.36
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch

from dataset import paths
from util import field_metrics as fm
from util.contrast_sweep import (DEFAULT_THETAS, best_global, held_out_theta,
                                 spearman, theta_mse)
from util.slice_eval import cone_split, gather

DEFAULT_RUN = "checkpoints/xhi2d_whno_glob_lr3e4"


def parse_bins(spec: str, xhi: torch.Tensor, n_bins: int) -> np.ndarray:
    if spec == "decile":
        q = torch.linspace(0, 1, n_bins + 1, device=xhi.device)
        e = torch.quantile(xhi, q).cpu().numpy()
        e[0] -= 1e-6
        return e
    return np.array([float(v) for v in spec.split(",")], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--cache", default=None,
                    help="slice cache filename in data/compressed (default: the "
                         "run's own training cache)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--all-slices", action="store_true",
                    help="use every row of --cache, ignoring the split "
                         "(for a purpose-built band cache that is already clean)")
    ap.add_argument("--bins", default="decile")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--xhi-max", type=float, default=None)
    ap.add_argument("--max-slices", type=int, default=None)
    ap.add_argument("--k-cut", type=float, default=0.20,
                    help="low-pass cutoff for the blur control")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache = paths.compressed(args.cache) if args.cache else None
    if args.all_slices:
        import h5py
        with h5py.File(cache) as f:
            idx = np.arange(f["x"].shape[0])
        s = gather(args.run, indices=idx, cache_file=cache,
                   max_slices=args.max_slices, device=args.device)
    else:
        s = gather(args.run, split=args.split, cache_file=cache,
                   max_slices=args.max_slices, device=args.device,
                   xhi_range=(0.0, args.xhi_max) if args.xhi_max else None)
    if args.xhi_max is not None:
        s = s.subset(s.xhi <= args.xhi_max)

    grid = DEFAULT_THETAS.to(args.device)
    blurred = fm.lowpass(s.truth, args.k_cut)
    mse_pred = theta_mse(s.pred, s.truth, grid)
    mse_blur = theta_mse(blurred, s.truth, grid)
    mse_truth = theta_mse(s.truth, s.truth, grid)
    ident = ((s.pred - s.truth) ** 2).flatten(1).mean(1)

    print(f"run   {args.run}")
    print(f"cache {cache or 'run default'}"
          f"{'  [all slices]' if args.all_slices else f'  [{args.split}]'}")
    print(f"{len(s)} slices from {len(np.unique(s.cone))} cones; "
          f"x_HI {float(s.xhi.min()):.4f}-{float(s.xhi.max()):.4f}; "
          f"z {float(s.z.min()):.2f}-{float(s.z.max()):.2f}")
    print(f"identity RMSE {float(ident.mean().sqrt()):.5f}   "
          f"width truth {float(fm.width_px(s.truth).mean()):.2f} / "
          f"pred {float(fm.width_px(s.pred).mean()):.2f} / "
          f"blurred {float(fm.width_px(blurred).mean()):.2f}\n")

    fit, held = cone_split(s.cone, seed=1)
    print(f"cone split: {int(fit.sum())} fit / {int(held.sum())} held out\n")

    edges = parse_bins(args.bins, s.xhi, args.n_bins)
    print(f"{'x_HI bin':>13} {'n':>5} {'nB':>4} {'edge_t':>7} {'w_p/w_t':>8} "
          f"{'th_blur':>8} {'th_pred':>8} | {'identity':>9} {'+theta':>9} "
          f"{'gain':>8} {'95% CI':>17} {'oracle':>8}")
    print("-" * 122)

    tot_i = tot_t = 0.0
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        inb = ((s.xhi >= lo) if b == 0 else (s.xhi > lo)) & (s.xhi <= hi)
        n = int(inb.sum())
        if n == 0:
            continue
        inb_np = inb.cpu().numpy()
        r = held_out_theta(mse_pred, grid, fit & inb_np, held & inb_np,
                           s.cone, ident, n_boot=args.boot)
        tb = float(best_global(mse_blur, grid, inb))
        ed = float(fm.edge_density(s.truth[inb]).mean())
        wr = float(fm.width_px(s.pred[inb]).mean() /
                   fm.width_px(s.truth[inb]).mean())
        if not r:
            print(f"{lo:6.3f}-{hi:6.3f} {n:5d} {0:4d} {ed:7.4f} {wr:8.2f} "
                  f"{tb:8.3f} {'--':>8} |  (too few held out)")
            continue
        print(f"{lo:6.3f}-{hi:6.3f} {n:5d} {r['n_held']:4d} {ed:7.4f} {wr:8.2f} "
              f"{tb:8.3f} {r['theta']:8.3f} | {r['identity']:9.5f} "
              f"{r['treated']:9.5f} {r['gain_pct']:+7.2f}% "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] {r['oracle_pct']:+7.2f}%")
        tot_i += r["identity"] ** 2 * r["n_held"]
        tot_t += r["treated"] ** 2 * r["n_held"]

    if tot_i:
        print("-" * 122)
        print(f"{'pooled':>13} {'':>5} {'':>4} {'':>7} {'':>8} {'':>8} {'':>8} | "
              f"{'':>9} {'':>9} {100 * ((tot_t / tot_i) ** 0.5 - 1):+7.2f}%")

    print(f"\nSpearman(theta_pred, x_HI) = {spearman(grid[mse_pred.argmin(0)], s.xhi):+.3f}"
          f"    Spearman(theta_blur, x_HI) = "
          f"{spearman(grid[mse_blur.argmin(0)], s.xhi):+.3f}")

    # global controls, for context
    print("\n--- global theta over all slices in scope ---")
    for name, mse, base in (("model", mse_pred, s.pred),
                            ("blurred truth", mse_blur, blurred),
                            ("truth (pure damage)", mse_truth, s.truth)):
        th = best_global(mse, grid)
        ref = float(((base - s.truth) ** 2).mean().sqrt())
        got = float(mse[grid == th].mean().sqrt())
        if ref > 1e-9:
            print(f"{name:<22} best theta {float(th):5.3f}   "
                  f"{ref:.5f} -> {got:.5f}  ({100 * (got / ref - 1):+.2f}%)")
        else:
            # truth vs itself: identity is exact, so report the damage curve
            worst = float(mse.max(dim=0).values.mean().sqrt())
            print(f"{name:<22} best theta {float(th):5.3f}   "
                  f"damage at theta=0.35: "
                  f"{float(mse[(grid - 0.35).abs().argmin()].mean().sqrt()):.5f}, "
                  f"worst on grid {worst:.5f}")


if __name__ == "__main__":
    main()
