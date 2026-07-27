"""Derive (theta, tau) from the ground truth instead of fitting them blind.

Two questions, deliberately separated:

A. GEOMETRY. The blind per-slice fit finds (theta, tau) by minimising RMSE
   against truth -- an opaque oracle.  Here they are read off the truth's edge
   geometry instead:
     tau_geom   = median prediction value *at the true edge location*, i.e.
                  "what does the model output where the boundary actually is".
     theta_geom = the theta that makes the prediction's transition width match
                  the truth's.
   If (theta_geom, tau_geom) recovers the blind fit's gain, we have explained
   what those parameters are.  If tau_geom also turns out to track a physical
   quantity the model can see, the inference-time wall in section 3.3 reopens.

B. PROJECTION / INFORMATION LOSS.  This separates two explanations that every
   previous experiment conflated:
     "the map cannot help"        vs  "training will not use the map".
   Band-limit the *truth* to a cutoff k_c (a stand-in for what the operator can
   synthesise) and ask how much of that loss a pointwise map buys back.  On
   band-limited truth the only error present IS blur, so if the map cannot undo
   it there, the map is useless in principle.  If it can, then the map works
   and the model's residual error must be something else -- misplaced edges,
   which no pointwise map can fix.

Run: sbatch slurm/probe_contrast_from_truth.sbatch
"""
import os
import sys

sys.path.insert(0, "/pfs/10/work/hd_id260-fno_training/fno-21cm")
os.chdir("/pfs/10/work/hd_id260-fno_training/fno-21cm")
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch

from contrast import apply_contrast
from util.field_metrics import lowpass, rmse, width_px
from util.slice_eval import gather

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
N_SLICES = 512
DEV = "cuda"
THETAS = torch.tensor(np.geomspace(0.03, 5.0, 40), dtype=torch.float32)
TAUS = torch.tensor(np.linspace(0.10, 0.90, 33), dtype=torch.float32)


def grid_mse(pred, truth):
    """(n_theta, n_tau, n_slices) MSE over the whole (theta, tau) grid."""
    mse = torch.empty(len(THETAS), len(TAUS), len(pred), device=pred.device)
    with torch.inference_mode():
        for i, th in enumerate(THETAS):
            for j, ta in enumerate(TAUS):
                g = apply_contrast(pred, th.to(pred.device), ta.to(pred.device))
                mse[i, j] = ((g - truth) ** 2).mean(dim=(-2, -1))
    return mse


def blind_fit(pred, truth):
    """Per-slice (theta, tau) minimising RMSE -- the opaque oracle."""
    mse = grid_mse(pred, truth)
    n = mse.shape[2]
    flat = mse.reshape(-1, n)
    best = flat.argmin(dim=0)
    ti, ai = best // len(TAUS), best % len(TAUS)
    return THETAS.to(pred.device)[ti], TAUS.to(pred.device)[ai], mse


def geometric_fit(pred, truth):
    """(theta, tau) read off truth edge geometry, not off the loss."""
    # tau: what the model outputs where the true boundary is.
    on_edge = (truth > 0.35) & (truth < 0.65)
    tau = torch.stack([
        (pred[i][on_edge[i]].median() if on_edge[i].any()
         else torch.tensor(0.5, device=pred.device))
        for i in range(len(pred))
    ])
    # theta: match the prediction's transition width to the truth's.
    w_true = width_px(truth)
    tau_b = tau[:, None, None]
    widths = torch.stack([
        width_px(apply_contrast(pred, th.to(pred.device), tau_b))
        for th in THETAS
    ])                                            # (n_theta, n_slices)
    theta = THETAS.to(pred.device)[(widths - w_true[None, :]).abs().argmin(0)]
    return theta, tau


def main():
    s = gather(RUN, split="test", max_slices=N_SLICES, device=DEV)
    pred, truth = s.pred, s.truth
    print(f"{len(pred)} held-out test slices, {tuple(pred.shape[-2:])} px\n")

    ident = float(rmse(pred, truth))

    # ---- A. geometry vs blind fit -------------------------------------
    th_b, ta_b, mse = blind_fit(pred, truth)
    th_g, ta_g = geometric_fit(pred, truth)

    def score(th, ta):
        return float(rmse(apply_contrast(pred, th[:, None, None],
                                         ta[:, None, None]), truth))

    gmean = mse.mean(dim=2)
    bi = gmean.argmin()
    th_gl = THETAS[bi // len(TAUS)].to(DEV)
    ta_gl = TAUS[bi % len(TAUS)].to(DEV)

    print("=== A. where does the gain come from ===")
    print(f"{'variant':<40} {'RMSE':>9} {'gain':>8}")
    rows = [
        ("identity", ident),
        (f"best GLOBAL (th={float(th_gl):.2f}, ta={float(ta_gl):.2f})",
         float(rmse(apply_contrast(pred, th_gl, ta_gl), truth))),
        ("per-slice GEOMETRIC (from truth edges)", score(th_g, ta_g)),
        ("per-slice BLIND fit (RMSE oracle)", score(th_b, ta_b)),
        ("  geometric tau + blind theta", score(th_b, ta_g)),
        ("  blind tau + geometric theta", score(th_g, ta_b)),
    ]
    for name, v in rows:
        print(f"{name:<40} {v:9.5f} {100*(v/ident-1):+7.2f}%")

    def corr(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))

    xhi = truth.mean(dim=(-2, -1))
    pmean = pred.mean(dim=(-2, -1))
    print("\n--- do the geometric params match the blind ones? ---")
    print(f"corr(tau_geom,   tau_blind)   = {corr(ta_g, ta_b):+.3f}")
    print(f"corr(log th_geom, log th_blind) = "
          f"{corr(th_g.log(), th_b.log()):+.3f}")
    print("\n--- are they physical (visible to the model)? ---")
    for nm, v in (("tau_geom", ta_g), ("tau_blind", ta_b)):
        print(f"corr({nm:<9}, mean x_HI truth) = {corr(v, xhi):+.3f}    "
              f"corr({nm:<9}, mean pred) = {corr(v, pmean):+.3f}")
    print(f"tau_geom  median={float(ta_g.median()):.3f} "
          f"10-90%={float(ta_g.quantile(0.1)):.3f}-{float(ta_g.quantile(0.9)):.3f}")
    print(f"tau_blind median={float(ta_b.median()):.3f} "
          f"10-90%={float(ta_b.quantile(0.1)):.3f}-{float(ta_b.quantile(0.9)):.3f}")

    # ---- B. projection / information loss ------------------------------
    print("\n=== B. can the map undo pure band-limiting? ===")
    print("Band-limit the TRUTH, then try to sharpen it back. The only error")
    print("present is blur, so this is the map's best possible case.\n")
    print(f"{'k_cut':>7} {'blurred':>9} {'+global':>9} {'+perslice':>10} "
          f"{'recovered':>10} {'width':>7}")
    for k_cut in (0.02, 0.03, 0.05, 0.08, 0.12, 0.20):
        y = lowpass(truth, k_cut)
        e0 = float(rmse(y, truth))
        m = grid_mse(y, truth)
        gm = m.mean(dim=2)
        b = gm.argmin()
        e_gl = float(gm.reshape(-1)[b].sqrt())
        e_ps = float(m.reshape(-1, m.shape[2]).min(dim=0).values.mean().sqrt())
        rec = 100 * (1 - e_ps / e0) if e0 > 0 else 0.0
        print(f"{k_cut:7.3f} {e0:9.5f} {e_gl:9.5f} {e_ps:10.5f} "
              f"{rec:9.1f}% {float(width_px(y).mean()):7.2f}")

    print(f"\nfor comparison, the real model:")
    e_ps_model = float(mse.reshape(-1, mse.shape[2]).min(dim=0).values.mean().sqrt())
    print(f"{'':7} {ident:9.5f} "
          f"{float(rmse(apply_contrast(pred, th_gl, ta_gl), truth)):9.5f} "
          f"{e_ps_model:10.5f} {100*(1-e_ps_model/ident):9.1f}% "
          f"{float(width_px(pred).mean()):7.2f}")
    print(f"{'truth':>7} {'--':>9} {'--':>9} {'--':>10} {'--':>10} "
          f"{float(width_px(truth).mean()):7.2f}")


if __name__ == "__main__":
    main()
