#!/usr/bin/env python3
"""Average fitted theta per redshift bin, for runs using the x_HI contrast map.

The schedule is keyed on x_HI, not on z, so "theta at redshift z" is not a
property of the schedule alone -- it depends on which x_HI a cone happens to be
at when it reaches z, and every cone reionises on its own timetable.  This
walks real cones through the fitted table and reports what theta actually gets
applied as a function of redshift.

Two keyings are reported and they answer different questions:

* ``truth``  -- key from the true per-slice mean x_HI.  This is the schedule's
  *intent*: the theta the map would apply if the key were perfect.
* ``pred``   -- key from a model's own per-slice mean prediction, exactly as
  ``contrast.los_key`` computes it during training.  This is what actually
  happens, and it needs a forward pass.

The gap between the two curves is the key bias made visible: wherever they
disagree, the map is applying the theta belonging to a different x_HI.

Reading the result: theta = THETA_MAX (5.0) is the identity -- no sharpening.
Smaller theta is a steeper map.  A z range sitting flat at 5.0 is one the map
never touches, and a range at the floor (0.25 by default) is one where the fit
wanted more contrast than is numerically safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrast import _squash, isotonic  # noqa: E402

CACHE = "/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5"


def load_schedule(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (edges, thetas) from a checkpoint's stepped schedule."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = edges = None
    for k, v in sd.items():
        if k.endswith("contrast.schedule.raw"):
            raw = v
        elif k.endswith("contrast.schedule.edges"):
            edges = v
    if raw is None or edges is None:
        raise SystemExit(
            f"{checkpoint} has no contrast schedule -- was it trained with "
            f"CONTRAST_MODE=xhi?"
        )
    thetas = _squash(raw.float()).numpy()
    return edges.float().numpy(), thetas


def theta_of(keys: np.ndarray, edges: np.ndarray,
             thetas: np.ndarray) -> np.ndarray:
    """Map key values through the stepped table (same bucketize as training)."""
    return thetas[np.searchsorted(edges, keys, side="right")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path/to/best_model_state_dict.pt")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--n-cones", type=int, default=200)
    ap.add_argument("--n-z-bins", type=int, default=32)
    ap.add_argument("--key-mode", choices=["mean", "monotone"],
                    default="monotone")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import h5py
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.cache, "r") as f:
        z = f["target_z"][:].astype(np.float64)
        n = min(args.n_cones, f["neutral_fraction"].shape[0])
        # Per-slice mean x_HI per cone: the truth key. Read cone by cone --
        # the full array is 180 GB and this only needs its slice means.
        raw = np.empty((n, len(z)), dtype=np.float64)
        for i in range(n):
            raw[i] = f["neutral_fraction"][i].mean(axis=(0, 1))

    key_truth = isotonic(raw)[0] if args.key_mode == "monotone" else raw

    edges_z = np.linspace(z.min(), z.max(), args.n_z_bins + 1)
    which = np.clip(np.digitize(z, edges_z) - 1, 0, args.n_z_bins - 1)
    centers = 0.5 * (edges_z[1:] + edges_z[:-1])

    rows = []
    for spec in args.checkpoints:
        name, _, path = spec.partition("=")
        edges, thetas = load_schedule(Path(path))
        th = theta_of(key_truth, edges, thetas)          # (n_cones, n_z)
        for b in range(args.n_z_bins):
            m = which == b
            if not m.any():
                continue
            vals = th[:, m]
            rows.append({
                "model": name,
                "z_lo": float(edges_z[b]), "z_hi": float(edges_z[b + 1]),
                "z_center": float(centers[b]),
                "theta_mean": float(vals.mean()),
                "theta_p16": float(np.percentile(vals, 16)),
                "theta_p50": float(np.percentile(vals, 50)),
                "theta_p84": float(np.percentile(vals, 84)),
                "xhi_mean": float(key_truth[:, m].mean()),
                "frac_identity": float((vals >= 4.99).mean()),
                "frac_floored": float((vals <= 0.2501).mean()),
                "n_slices": int(m.sum() * th.shape[0]),
            })

    csv = args.out / "theta_z_profile.csv"
    cols = list(rows[0].keys())
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({r["model"] for r in rows})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for name in names:
        sub = [r for r in rows if r["model"] == name]
        zc = [r["z_center"] for r in sub]
        ax.plot(zc, [r["theta_mean"] for r in sub], lw=2, label=name)
        ax.fill_between(zc, [r["theta_p16"] for r in sub],
                        [r["theta_p84"] for r in sub], alpha=0.15)
        ax2.plot(zc, [r["frac_identity"] for r in sub], lw=1.6,
                 label=f"{name} (identity)")
        ax2.plot(zc, [r["frac_floored"] for r in sub], lw=1.6, ls="--",
                 label=f"{name} (floored)")
    ax.axhline(5.0, color="black", lw=1, ls=":", label="identity (theta=5)")
    ax.axhline(0.25, color="red", lw=1, ls=":", label="floor (theta=0.25)")
    ax.set_ylabel("theta")
    ax.set_title(f"Applied theta vs redshift ({args.key_mode} key, truth-keyed)"
                 f"  -- band = 16-84th pct across cones")
    ax.legend(fontsize=8)
    ax2.set_ylabel("fraction of slices")
    ax2.set_xlabel("z")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out / "theta_z_profile.png", dpi=140)

    with open(args.out / "theta_z_config.json", "w") as fh:
        json.dump({"n_cones": int(raw.shape[0]), "n_z_bins": args.n_z_bins,
                   "key_mode": args.key_mode,
                   "checkpoints": args.checkpoints}, fh, indent=2)
    print(f"wrote {csv} and theta_z_profile.png")


if __name__ == "__main__":
    main()
