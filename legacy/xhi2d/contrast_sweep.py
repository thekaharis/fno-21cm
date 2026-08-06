"""Grid sweeps and held-out scoring for the (theta, tau) contrast map.

The map itself lives in ``contrast.py``; this is the measurement machinery that
every probe reuses -- sweep a grid, pick a best parameter on one set of cones,
score it on another, and put an interval on the result.

Findings the callers depend on (see NOTES-contrast-map.md):
  * a global theta never beats the identity on a converged model;
  * per-slice gains rest on tau, which is not predictable at inference;
  * gains in near-empty bins are thresholding artefacts, not sharpening --
    always report edge density alongside a gain.
"""

from __future__ import annotations

import numpy as np
import torch

from contrast import apply_contrast

# Geometric in theta (it acts like a scale) and linear in tau.
DEFAULT_THETAS = torch.tensor(np.geomspace(0.05, 5.0, 64), dtype=torch.float32)
DEFAULT_TAUS = torch.tensor(np.linspace(0.10, 0.90, 33), dtype=torch.float32)


def theta_mse(src: torch.Tensor, truth: torch.Tensor, thetas=None,
              tau: float = 0.5) -> torch.Tensor:
    """(n_theta, n_slices) MSE of ``g_theta(src)`` against ``truth``."""
    thetas = DEFAULT_THETAS if thetas is None else thetas
    out = torch.empty(len(thetas), len(src), device=src.device)
    with torch.inference_mode():
        for i, th in enumerate(thetas):
            g = apply_contrast(src, th.to(src.device), tau)
            out[i] = ((g - truth) ** 2).flatten(1).mean(1)
    return out


def theta_tau_mse(src: torch.Tensor, truth: torch.Tensor, thetas=None,
                  taus=None) -> torch.Tensor:
    """(n_theta, n_tau, n_slices) MSE over the full grid."""
    thetas = DEFAULT_THETAS if thetas is None else thetas
    taus = DEFAULT_TAUS if taus is None else taus
    out = torch.empty(len(thetas), len(taus), len(src), device=src.device)
    with torch.inference_mode():
        for i, th in enumerate(thetas):
            for j, ta in enumerate(taus):
                g = apply_contrast(src, th.to(src.device), ta.to(src.device))
                out[i, j] = ((g - truth) ** 2).flatten(1).mean(1)
    return out


def best_global(mse: torch.Tensor, grid: torch.Tensor, mask=None) -> torch.Tensor:
    """The single grid value minimising mean MSE over ``mask``."""
    sub = mse if mask is None else mse[:, mask]
    return grid.to(mse.device)[sub.mean(dim=1).argmin()]


def rmse_of(mse_row: torch.Tensor) -> float:
    return float(mse_row.mean().sqrt())


def bootstrap_gain(mse_identity, mse_treated, cone, n: int = 2000,
                   seed: int = 0, pct=(2.5, 97.5)) -> tuple[float, float]:
    """Percentile CI on the RMSE gain, resampling whole CONES.

    Slices from one lightcone are correlated, so a slice-level bootstrap gives
    intervals that are far too tight.
    """
    mi = np.asarray(mse_identity, dtype=np.float64)
    mt = np.asarray(mse_treated, dtype=np.float64)
    cone = np.asarray(cone)
    rng = np.random.default_rng(seed)
    uc = np.unique(cone)
    groups = [np.flatnonzero(cone == c) for c in uc]
    out = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        m = np.concatenate([groups[i] for i in pick])
        out[k] = 100.0 * ((mt[m].mean() / mi[m].mean()) ** 0.5 - 1.0)
    return tuple(np.percentile(out, pct))


def held_out_theta(mse: torch.Tensor, grid: torch.Tensor, fit_mask,
                   held_mask, cone, identity_mse: torch.Tensor,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """Fit one theta on ``fit_mask``, score it on ``held_mask``.

    Returns the fitted value, held-out RMSE before/after, the gain and its CI,
    and the per-slice oracle on the same held-out slices as a ceiling.
    """
    dev = mse.device
    fit_t = torch.as_tensor(fit_mask, device=dev)
    held_t = torch.as_tensor(held_mask, device=dev)
    if not bool(fit_t.any()) or not bool(held_t.any()):
        return {}
    theta = best_global(mse, grid, fit_t)
    row = mse[grid.to(dev) == theta].flatten()
    mi = identity_mse[held_t].cpu().numpy()
    mt = row[held_t].cpu().numpy()
    e_i, e_t = float(mi.mean() ** 0.5), float(mt.mean() ** 0.5)
    e_o = float(mse[:, held_t].min(dim=0).values.mean().sqrt())
    lo, hi = bootstrap_gain(mi, mt, np.asarray(cone)[np.asarray(held_mask)],
                            n=n_boot, seed=seed)
    return {
        "theta": float(theta), "n_fit": int(fit_t.sum()), "n_held": int(held_t.sum()),
        "identity": e_i, "treated": e_t, "oracle": e_o,
        "gain_pct": 100.0 * (e_t / e_i - 1.0),
        "ci": (lo, hi), "oracle_pct": 100.0 * (e_o / e_i - 1.0),
        "improved_frac": float((mt < mi).mean()),
    }


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar, br = ar - ar.mean(), br - br.mean()
    return float((ar * br).sum() / (ar.norm() * br.norm()).clamp_min(1e-12))
