"""Fit theta as a smooth function of ionisation state, usable at inference.

The per-bin sweep (NOTES-contrast-map.md 6.1) found a real held-out gain at
x_HI ~ 0.005-0.05 that decays to nothing by x_HI ~ 0.1.  A lookup table over
truth bins is not deployable; this fits a smooth schedule of the model's OWN
mean output, which it always has:

    theta(m) = theta_lo + (theta_hi - theta_lo) * sigmoid((log10(m + eps) - c) / s)

Four parameters, fitted by minimising post-map MSE directly (not by regressing
against per-bin optima, which throws away the weighting).  Fitted on one set of
cones, scored on disjoint cones.

Step 0 checks the premise: the schedule keys on mean prediction, so mean
prediction must track true x_HI.  If it does not, nothing downstream matters.

Writes the fitted parameters to contrast_theta_schedule.json for
``ContrastOutput(mode="xhi")`` to load.

Run: sbatch slurm/fit_theta_schedule.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch

from contrast import THETA_MAX, THETA_MIN, ThetaSchedule, apply_contrast
from dataset import paths
from util.contrast_sweep import DEFAULT_THETAS, bootstrap_gain, theta_mse
from util.slice_eval import cone_split, gather

DEFAULT_RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
OUT = "contrast_theta_schedule.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--cache", default="xhi_band_000_036.h5")
    ap.add_argument("--xhi-max", type=float, default=0.36)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-lo", type=float, default=0.5)
    ap.add_argument("--init-hi", type=float, default=4.0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import h5py
    cache = paths.compressed(args.cache)
    with h5py.File(cache) as f:
        idx = np.arange(f["x"].shape[0])
    s = gather(args.run, indices=idx, cache_file=cache, device=args.device)
    s = s.subset(s.xhi <= args.xhi_max)
    mean_pred = s.pred.flatten(1).mean(1)

    # ---- 0. is the key usable at inference? ---------------------------
    err = (mean_pred - s.xhi).abs()
    ss = ((s.xhi - s.xhi.mean()) ** 2).sum()
    r2 = 1.0 - ((mean_pred - s.xhi) ** 2).sum() / ss
    print(f"{len(s)} slices, {len(np.unique(s.cone))} cones, "
          f"x_HI <= {args.xhi_max}\n")
    print("=== 0. does mean prediction track true x_HI? ===")
    print(f"R^2(mean_pred -> x_HI) = {float(r2):.4f}   "
          f"MAE = {float(err.mean()):.4f}   p95 |err| = "
          f"{float(err.quantile(0.95)):.4f}")
    print("   the schedule keys on mean_pred, so this is the premise\n")

    # Which inference-time signal best identifies ionisation state?  The
    # schedule is only as good as its key, and mean_pred alone is weak.
    import h5py as _h5
    with _h5.File(cache) as f:
        par = torch.as_tensor(np.asarray(f["params"])[s.index],
                              dtype=torch.float32, device=args.device)
    par = (par - par.mean(0)) / par.std(0).clamp_min(1e-6)
    zz = ((s.z - s.z.mean()) / s.z.std()).view(-1, 1)
    mp = mean_pred.view(-1, 1)
    feats = {"mean_pred": mp, "z": zz, "z+params": torch.cat([zz, par], 1),
             "mean_pred+z": torch.cat([mp, zz], 1),
             "mean_pred+z+params": torch.cat([mp, zz, par], 1)}
    fit0, held0 = cone_split(s.cone, seed=1)
    f0 = torch.as_tensor(fit0, device=args.device)
    h0 = torch.as_tensor(held0, device=args.device)
    print("=== 1. best inference-time estimator of x_HI (held-out cones) ===")
    best_key, best_r2, best_hat = None, -9e9, None
    for name, X in feats.items():
        Xb = torch.cat([X, torch.ones(len(X), 1, device=args.device)], 1)
        sol = torch.linalg.lstsq(Xb[f0], s.xhi[f0].view(-1, 1)).solution
        hat = (Xb @ sol).view(-1).clamp(0.0, 1.0)
        r = 1 - ((hat[h0] - s.xhi[h0]) ** 2).sum() / (
            (s.xhi[h0] - s.xhi[h0].mean()) ** 2).sum()
        print(f"  {name:<20} R^2 {float(r):+.4f}   MAE "
              f"{float((hat[h0]-s.xhi[h0]).abs().mean()):.4f}")
        if float(r) > best_r2:
            best_key, best_r2, best_hat = name, float(r), hat
    print(f"  -> keying the schedule on {best_key} (R^2 {best_r2:.4f})\n")
    key = best_hat.detach()

    fit_np, held_np = cone_split(s.cone, seed=1)
    fit = torch.as_tensor(fit_np, device=args.device)
    held = torch.as_tensor(held_np, device=args.device)

    # ---- 1. fit the schedule by direct MSE minimisation ---------------
    sched = ThetaSchedule(theta_lo=args.init_lo,
                          theta_hi=args.init_hi).to(args.device)
    opt = torch.optim.Adam(sched.parameters(), lr=args.lr)
    pf, tf, mf = s.pred[fit], s.truth[fit], key[fit]
    for step in range(args.steps):
        opt.zero_grad()
        th = sched(mf).view(-1, 1, 1)
        loss = ((apply_contrast(pf, th, 0.5) - tf) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  train MSE {float(loss.detach()):.6f}  "
                  f"{sched.describe()}")

    # ---- 2. score on held-out cones -----------------------------------
    print("\n=== 2. held-out cones ===")
    with torch.no_grad():
        th_h = sched(key[held]).view(-1, 1, 1)
        mse_sched = ((apply_contrast(s.pred[held], th_h, 0.5)
                      - s.truth[held]) ** 2).flatten(1).mean(1)
    mse_id = ((s.pred[held] - s.truth[held]) ** 2).flatten(1).mean(1)
    grid = DEFAULT_THETAS.to(args.device)
    mse_grid = theta_mse(s.pred[held], s.truth[held], grid)
    best_gl = mse_grid.mean(1).argmin()
    e_id = float(mse_id.mean().sqrt())
    e_sc = float(mse_sched.mean().sqrt())
    e_gl = float(mse_grid[best_gl].mean().sqrt())
    e_or = float(mse_grid.min(0).values.mean().sqrt())
    ci = bootstrap_gain(mse_id.cpu().numpy(), mse_sched.cpu().numpy(),
                        s.cone[held_np])
    print(f"{'variant':<38} {'RMSE':>9} {'gain':>8}")
    print(f"{'identity':<38} {e_id:9.5f} {0.0:+7.2f}%")
    print(f"{'best single global theta (%.2f)' % float(grid[best_gl]):<38} "
          f"{e_gl:9.5f} {100*(e_gl/e_id-1):+7.2f}%")
    print(f"{'FITTED theta(mean_pred)':<38} {e_sc:9.5f} "
          f"{100*(e_sc/e_id-1):+7.2f}%   95% CI "
          f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]")
    print(f"{'per-slice theta oracle':<38} {e_or:9.5f} "
          f"{100*(e_or/e_id-1):+7.2f}%")

    # ---- 3. where the gain sits ---------------------------------------
    print("\n=== 3. held-out gain by true x_HI ===")
    print(f"{'x_HI bin':>13} {'n':>5} {'theta(m)':>9} {'identity':>9} "
          f"{'+sched':>9} {'gain':>8}")
    for lo, hi in ((0, .005), (.005, .01), (.01, .02), (.02, .03), (.03, .05),
                   (.05, .10), (.10, .20), (.20, .36)):
        m = (s.xhi[held] > lo) & (s.xhi[held] <= hi)
        if int(m.sum()) < 5:
            continue
        a = float(mse_id[m].mean().sqrt()); b = float(mse_sched[m].mean().sqrt())
        print(f"{lo:6.3f}-{hi:6.3f} {int(m.sum()):5d} "
              f"{float(th_h.view(-1)[m].median()):9.3f} {a:9.5f} {b:9.5f} "
              f"{100*(b/a-1):+7.2f}%")

    payload = {"form": "theta_lo + (theta_hi-theta_lo)*sigmoid((log10(m+eps)-c)/s)",
               "fitted_on": args.run, "cache": args.cache, "key": best_key,
               "held_out_gain_pct": 100 * (e_sc / e_id - 1),
               "held_out_ci": list(ci), **sched.state_dict_floats()}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}: {sched.describe()}")
    print("curve:  " + "  ".join(
        f"m={m:.3f}->{float(sched(torch.tensor([m], device=args.device))):.2f}"
        for m in (0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35)))


if __name__ == "__main__":
    main()
