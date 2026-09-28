"""Side-by-side transverse slices: ground truth vs prediction, per field.

A deliberately plain companion to viz/multifield_detailed.py -- no spectra, no
statistics, just the fields as images so the morphology can be compared by eye.

Redshifts are chosen where the field is actually varying (peak transverse
standard deviation) and spread across that active band, because a uniform pick
across z = 5..25 spends most of its panels on quiescent gas where every model
looks perfect.

``--orientation los`` instead shows line-of-sight (x, z) planes at several
transverse positions. Those are cropped to the active redshift band by default:
the cache spans z = 5..25 but the ionization front occupies a narrow strip of
it, and an uncropped panel is mostly blank gas. ``--full-z`` disables the crop.

  python -m viz.multifield_slices --prediction experiments/.../cone_541.h5 \
      --n-slices 5 --out-dir figures/3d_xhi/mf_cnn_fno/slices
  python -m viz.multifield_slices --prediction ... --orientation los
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from viz._zaxis import edges, transverse_edges

BOX_MPC = 200.0
CMAPS = {"neutral_fraction": "magma", "brightness_temp": "inferno",
         "density": "viridis", "los_velocity": "coolwarm"}
#: Fields whose physical range should be used verbatim rather than percentiles.
BOUNDS = {"neutral_fraction": (0.0, 1.0)}


def active_slices(truth, n):
    """n redshift indices spread over the band where the field varies."""
    sd = truth.std(axis=(0, 1))
    if not np.any(sd > 0):
        return np.linspace(0, truth.shape[2] - 1, n).astype(int)
    peak = sd.max()
    active = np.flatnonzero(sd > 0.15 * peak)
    lo, hi = active[0], active[-1]
    # Bias the picks toward the peak: the front is where the models differ.
    return np.unique(np.linspace(lo, hi, n).astype(int))


def active_band(truth, pad=12):
    """Index range over which the field varies, with a little padding."""
    sd = truth.std(axis=(0, 1))
    if not np.any(sd > 0):
        return 0, truth.shape[2] - 1
    active = np.flatnonzero(sd > 0.05 * sd.max())
    return max(0, active[0] - pad), min(truth.shape[2] - 1, active[-1] + pad)


def los_figure(field, truth, pred, z, cone, units, n_slices, out_path, full_z=False):
    """Truth vs prediction on (x, z) planes at evenly spaced transverse rows."""
    lo, hi = (0, truth.shape[2] - 1) if full_z else active_band(truth)
    zc = z[lo:hi + 1]
    ys = np.linspace(0.12, 0.88, n_slices) * truth.shape[1]
    ys = np.unique(ys.astype(int))
    n = len(ys)
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n + 1.8, 10.2),
                             gridspec_kw=dict(hspace=0.2, wspace=0.08))
    if n == 1:
        axes = axes[:, None]
    cmap = CMAPS.get(field, "viridis")
    xe, ze = transverse_edges(truth.shape[0], BOX_MPC), edges(zc)
    if field in BOUNDS:
        v0, v1 = BOUNDS[field]
    else:
        band = truth[:, :, lo:hi + 1]
        v0, v1 = np.percentile(band, [0.5, 99.5])

    for col, y in enumerate(ys):
        tt = truth[:, y, lo:hi + 1]
        pp = pred[:, y, lo:hi + 1]
        last = None
        for row, (label, data) in enumerate((("truth", tt), ("prediction", pp))):
            ax = axes[row, col]
            last = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=v0, vmax=v1,
                                 shading="flat", rasterized=True)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(f"{label}\nredshift", fontsize=11)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(f"y = {y * BOX_MPC / truth.shape[1]:.0f} Mpc", fontsize=11)
        if col == n - 1:
            fig.colorbar(last, ax=axes[:2, :].ravel().tolist(), fraction=0.022,
                         pad=0.02, label=units)
        ax = axes[2, col]
        err = pp - tt
        lim = np.percentile(np.abs(err), 99) or 1.0
        im = ax.pcolormesh(xe, ze, err.T, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0.0, -lim, lim), shading="flat",
                           rasterized=True)
        if col == 0:
            ax.set_ylabel("pred - truth\nredshift", fontsize=11)
        else:
            ax.set_yticks([])
        ax.set_xlabel(f"x [Mpc]\nRMSE {np.sqrt((err ** 2).mean()):.3g}", fontsize=9)
        if col == n - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=units)

    crop = "full z range" if full_z else f"cropped to z = {zc[0]:.1f}-{zc[-1]:.1f}"
    scale = "shared colour scale" if field in BOUNDS else "shared percentile scale"
    fig.suptitle(f"{field} — cone {cone}   "
                 f"(line-of-sight slices, {crop}, {scale})", fontsize=15, y=0.965)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure(field, truth, pred, z, cone, units, n_slices, out_path):
    idx = active_slices(truth, n_slices)
    n = len(idx)
    fig, axes = plt.subplots(3, n, figsize=(3.05 * n + 1.6, 9.6),
                             gridspec_kw=dict(hspace=0.16, wspace=0.06))
    if n == 1:
        axes = axes[:, None]
    cmap = CMAPS.get(field, "viridis")
    ext = [0, BOX_MPC, 0, BOX_MPC]

    for col, j in enumerate(idx):
        t, p = truth[:, :, j], pred[:, :, j]
        # Shared scale within a column so truth and prediction are comparable.
        # Bounded fields use their physical range: percentile-clipping x_HI to
        # [0.5, 99.5] renders a field that is mostly 0 or 1 as pure black/yellow
        # and hides the partially ionized voxels that carry the morphology.
        if field in BOUNDS:
            v0, v1 = BOUNDS[field]
        else:
            v0, v1 = np.percentile(t, [0.5, 99.5])
            if v0 == v1:
                v0, v1 = t.min(), max(t.max(), t.min() + 1e-6)
        last = None
        for row, (label, data) in enumerate((("truth", t), ("prediction", p))):
            ax = axes[row, col]
            last = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=v0, vmax=v1, extent=ext)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
            if row == 0:
                ax.set_title(f"z = {z[j]:.2f}", fontsize=12)
        # A bounded field shares one scale across all columns, so it needs a
        # single colorbar; an unbounded one is scaled per column (brightness
        # temperature spans ~140 mK at the front and ~0 at high z, and a shared
        # scale would flatten every panel but the front) and needs one each.
        if field not in BOUNDS:
            fig.colorbar(last, ax=axes[:2, col].tolist(), fraction=0.046,
                         pad=0.02, label=units if col == n - 1 else "")
        elif col == n - 1:
            fig.colorbar(last, ax=axes[:2, :].ravel().tolist(), fraction=0.023,
                         pad=0.02, label=units)
        ax = axes[2, col]
        err = p - t
        lim = np.percentile(np.abs(err), 99) or 1.0
        im = ax.imshow(err.T, origin="lower", cmap="RdBu_r",
                       norm=TwoSlopeNorm(0.0, -lim, lim), extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("pred - truth", fontsize=12)
        ax.set_xlabel(f"RMSE {np.sqrt((err**2).mean()):.3g}", fontsize=9)
        if col == n - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=units)

    scale_note = ("shared colour scale" if field in BOUNDS
                  else "colour scale set per column")
    fig.suptitle(f"{field} — cone {cone}   "
                 f"(200 x 200 Mpc transverse slices, {scale_note})",
                 fontsize=15, y=0.965)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-slices", type=int, default=5)
    ap.add_argument("--orientation", choices=("transverse", "los"), default="transverse")
    ap.add_argument("--full-z", action="store_true",
                    help="LOS mode: keep the whole z = 5..25 range instead of cropping")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.prediction:
        with h5py.File(path, "r") as f:
            z = f["target_z"][:]
            cone = int(f.attrs["cone_id"])
            for field in f["target"]:
                suffix = "los" if args.orientation == "los" else "slices"
                out = args.out_dir / f"cone{cone}_{field}_{suffix}.png"
                truth = f[f"target/{field}"][:].astype(np.float32)
                pred = f[f"prediction/{field}"][:].astype(np.float32)
                units = f[f"target/{field}"].attrs.get("units", "")
                if args.orientation == "los":
                    los_figure(field, truth, pred, z, cone, units,
                               args.n_slices, out, args.full_z)
                else:
                    figure(field, truth, pred, z, cone, units, args.n_slices, out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
