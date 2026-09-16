"""Lightcone strips: full line-of-sight extent, one transverse row, stacked.

The conventional 21 cm lightcone rendering -- redshift runs along the long
horizontal axis, one transverse direction fills the short vertical axis, and
strips are stacked so several simulations can be compared at a glance.

Per cone a group of three strips is drawn (truth, prediction, difference), and
the groups are stacked down the page. One figure per field, because the two
fields need different colour scales.

  python -m viz.multifield_lightcone_strips --prediction pred/*.h5 \
      --out-dir figures/3d_xhi/mf_cnn_fno/strips
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

BOX_MPC = 200.0
CMAPS = {"neutral_fraction": "magma", "brightness_temp": "inferno",
         "density": "viridis", "los_velocity": "coolwarm"}
#: Fields whose physical range is used verbatim instead of percentiles.
BOUNDS = {"neutral_fraction": (0.0, 1.0)}


def z_ticks(z, n=9):
    """Tick positions in index space with redshift labels."""
    idx = np.linspace(0, len(z) - 1, n).astype(int)
    return idx, [f"{z[i]:.1f}" for i in idx]


def active_band(truth, pad=8):
    sd = truth.std(axis=(0, 1))
    if not np.any(sd > 0):
        return 0, truth.shape[2] - 1
    a = np.flatnonzero(sd > 0.05 * sd.max())
    return max(0, a[0] - pad), min(truth.shape[2] - 1, a[-1] + pad)


def build(field, entries, out_path, y_frac=0.5, strip_h=0.85, crop=False,
          show_error=True, note=""):
    """entries: list of (cone_id, truth, pred, z, units)."""
    # A common scale across cones, so strips are comparable down the page --
    # the whole point of stacking them.
    if field in BOUNDS:
        v0, v1 = BOUNDS[field]
    else:
        pool = np.concatenate([t[:, :, :].ravel()[::97] for _, t, _, _, _ in entries])
        v0, v1 = np.percentile(pool, [0.5, 99.5])
    elim = np.percentile(
        np.abs(np.concatenate([(p - t).ravel()[::97] for _, t, p, _, _ in entries])), 99)

    per_cone = 3 if show_error else 2
    rows = per_cone * len(entries)
    fig, axes = plt.subplots(rows, 1, figsize=(16.5, strip_h * rows + 1.7),
                             gridspec_kw=dict(hspace=0.09))
    if rows == 1:
        axes = [axes]
    cmap = CMAPS.get(field, "viridis")
    units = entries[0][4]
    last_img = last_err = None

    for ci, (cone, truth, pred, z, _) in enumerate(entries):
        lo, hi = active_band(truth) if crop else (0, truth.shape[2] - 1)
        y = int(y_frac * truth.shape[1])
        tt = truth[:, y, lo:hi + 1]
        pp = pred[:, y, lo:hi + 1]
        zc = z[lo:hi + 1]
        panels = [("truth", tt, False), ("prediction", pp, False)]
        if show_error:
            panels.append(("pred - truth", pp - tt, True))
        for pi, (label, data, is_err) in enumerate(panels):
            ax = axes[ci * per_cone + pi]
            if is_err:
                last_err = ax.imshow(data.T[::-1] if False else data, aspect="auto",
                                     origin="lower", cmap="RdBu_r",
                                     norm=TwoSlopeNorm(0.0, -elim, elim))
            else:
                last_img = ax.imshow(data, aspect="auto", origin="lower",
                                     cmap=cmap, vmin=v0, vmax=v1)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center",
                          labelpad=6)
            bottom = pi == len(panels) - 1
            if bottom and ci == len(entries) - 1:
                idx, labs = z_ticks(zc)
                ax.set_xticks(idx); ax.set_xticklabels(labs, fontsize=9)
                ax.set_xlabel("redshift", fontsize=11)
            else:
                ax.set_xticks([])
            if pi == 0:
                rmse = float(np.sqrt(((pp - tt) ** 2).mean()))
                ax.text(1.004, 0.5, f"cone {cone}\nRMSE {rmse:.3g}",
                        transform=ax.transAxes, fontsize=8.5, va="center", ha="left")

    # Leave a clear gutter between the per-cone annotations (drawn just past the
    # axes) and the colour bars, or the two overlap.
    fig.subplots_adjust(left=0.075, right=0.845, top=0.945, bottom=0.075)
    cax = fig.add_axes([0.935, 0.52, 0.011, 0.36])
    fig.colorbar(last_img, cax=cax, label=units)
    if show_error:
        cax2 = fig.add_axes([0.935, 0.10, 0.011, 0.32])
        fig.colorbar(last_err, cax=cax2, label=f"error [{units}]")
    span = "active band" if crop else "z = 5 to 25"
    fig.suptitle(f"{field} lightcone strips — transverse row y = "
                 f"{y_frac * BOX_MPC:.0f} Mpc, {span}{note}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--y-frac", type=float, default=0.5)
    ap.add_argument("--strip-height", type=float, default=0.85)
    ap.add_argument("--crop", action="store_true", help="limit to the active band")
    ap.add_argument("--no-error", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_field = {}
    for path in sorted(args.prediction):
        with h5py.File(path, "r") as f:
            cone = int(f.attrs["cone_id"])
            z = f["target_z"][:]
            for field in f["target"]:
                by_field.setdefault(field, []).append(
                    (cone, f[f"target/{field}"][:].astype(np.float32),
                     f[f"prediction/{field}"][:].astype(np.float32), z,
                     f[f"target/{field}"].attrs.get("units", "")))
    for field, entries in by_field.items():
        entries.sort(key=lambda e: e[0])
        suffix = "_active" if args.crop else ""
        out = args.out_dir / f"{field}_strips{suffix}.png"
        build(field, entries, out, args.y_frac, args.strip_height,
              args.crop, not args.no_error, args.note)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
