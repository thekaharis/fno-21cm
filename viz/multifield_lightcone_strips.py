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

from viz._zaxis import edges, transverse_edges

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
    # sharex: every strip is drawn on its true redshift cell edges, so cones
    # with different native LOS lengths still line up in redshift.
    fig, axes = plt.subplots(rows, 1, figsize=(16.5, strip_h * rows + 1.7),
                             sharex=True, gridspec_kw=dict(hspace=0.09))
    if rows == 1:
        axes = [axes]
    cmap = CMAPS.get(field, "viridis")
    units = entries[0][4]
    last_img = last_err = None
    zspan = [np.inf, -np.inf]

    # Crop to the UNION of the cones' active bands, in redshift. Cropping each
    # cone to its own band and drawing in index space (as an earlier version
    # did) leaves every strip but the bottom one mislabelled on a shared axis.
    if crop:
        bands = []
        for _, truth, _, z, _ in entries:
            lo, hi = active_band(truth)
            bands.append((z[lo], z[hi]))
        zlo, zhi = min(b[0] for b in bands), max(b[1] for b in bands)

    for ci, (cone, truth, pred, z, _) in enumerate(entries):
        keep = (z >= zlo) & (z <= zhi) if crop else np.ones(len(z), bool)
        y = int(y_frac * truth.shape[1])
        tt = truth[:, y, keep]
        pp = pred[:, y, keep]
        zc = z[keep]
        ze = edges(zc)
        xe = transverse_edges(tt.shape[0], BOX_MPC)
        zspan = [min(zspan[0], ze[0]), max(zspan[1], ze[-1])]
        panels = [("truth", tt, False), ("prediction", pp, False)]
        if show_error:
            panels.append(("pred - truth", pp - tt, True))
        for pi, (label, data, is_err) in enumerate(panels):
            ax = axes[ci * per_cone + pi]
            if is_err:
                last_err = ax.pcolormesh(ze, xe, data, cmap="RdBu_r",
                                         norm=TwoSlopeNorm(0.0, -elim, elim),
                                         shading="flat", rasterized=True)
            else:
                last_img = ax.pcolormesh(ze, xe, data, cmap=cmap, vmin=v0, vmax=v1,
                                         shading="flat", rasterized=True)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center",
                          labelpad=6)
            if ci == len(entries) - 1 and pi == len(panels) - 1:
                ax.set_xlabel("redshift", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if pi == 0:
                rmse = float(np.sqrt(((pp - tt) ** 2).mean()))
                ax.text(1.004, 0.5, f"cone {cone}\nRMSE {rmse:.3g}",
                        transform=ax.transAxes, fontsize=8.5, va="center", ha="left")

    axes[-1].set_xlim(*zspan)
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
    ap.add_argument("--per-figure", type=int, default=4,
                    help="cones per figure; more cones are split across numbered figures")
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
        groups = [entries[i:i + args.per_figure]
                  for i in range(0, len(entries), max(1, args.per_figure))]
        for gi, group in enumerate(groups, start=1):
            part = f"_part{gi}" if len(groups) > 1 else ""
            out = args.out_dir / f"{field}_strips{suffix}{part}.png"
            build(field, group, out, args.y_frac, args.strip_height,
                  args.crop, not args.no_error, args.note)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
