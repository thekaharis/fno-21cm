"""Merge the lightcone sharpness survey and produce the report and figures.

Consumes the per-shard npz files from dataset/survey_truth_sharpness.py.

The headline quantities:

  partial fraction  share of voxels strictly between fully ionized and fully
                    neutral, at three thresholds -- how much of the field is a
                    front rather than one of the two phases
  width_px          transverse transition width, band pixels per unit of total
                    variation. Mean |grad| is deliberately not used: blurring a
                    monotonic step spreads the same total variation over more
                    pixels, so it barely moves
  TV_xy vs TV_z     transverse against line-of-sight total variation, i.e. how
                    anisotropic the fronts are in the raw lightcone

Run: python -m viz.report_truth_survey
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("figures/shared/diagnostics/truth_survey")
OUT_MD = "figures/shared/diagnostics/truth_survey/REPORT.md"
OUT_FIG = "figures/shared/diagnostics/truth_survey/truth_sharpness_survey.png"

SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de"
# Validated all-pairs in light mode (worst CVD dE 9.2, normal-vision 24.0).
HUES = ["#2a78d6", "#eb6834", "#1baf7a"]


def merge() -> dict:
    files = sorted(SRC.glob("shard*.npz"))
    if not files:
        sys.exit(f"no shards in {SRC}")
    acc = None
    for f in files:
        d = np.load(f)
        if acc is None:
            acc = {k: np.array(d[k]) for k in d.files}
        else:
            for k in ("n_planes", "sum_partial", "sum_tv_xy", "sum_tv_z",
                      "sum_band", "sum_width", "sum_peak", "value_hist",
                      "n_cones", "n_planes_total"):
                acc[k] = acc[k] + d[k]
    acc["n_shards"] = len(files)
    return acc


def main() -> None:
    a = merge()
    nb = int(a["n_xhi_bins"])
    n = a["n_planes"].astype(np.float64)
    centres = (np.arange(nb) + 0.5) / nb
    ok = n > 0
    mean = lambda k: np.where(ok, a[k] / np.maximum(n, 1), np.nan)

    partial = np.stack([np.where(ok, a["sum_partial"][b] / np.maximum(n, 1), np.nan)
                        for b in range(a["sum_partial"].shape[0])])
    width, tv_xy, tv_z = mean("sum_width"), mean("sum_tv_xy"), mean("sum_tv_z")
    peak, band = mean("sum_peak"), mean("sum_band")

    vh = a["value_hist"].astype(np.float64)
    nvb = int(a["n_value_bins"])
    vtot = vh.sum()
    edges = np.linspace(0, 1, nvb + 1)
    frac_ion = vh[0] / vtot
    frac_neu = vh[-1] / vtot
    frac_mid = vh[1:-1].sum() / vtot

    print(f"{int(a['n_cones'])} lightcones, {int(a['n_planes_total']):,} "
          f"transverse planes, {vtot:,.0f} voxels "
          f"({a['n_shards']} shards)\n")
    print(f"voxel population:")
    print(f"  fully ionized  (x_HI < {edges[1]:.3f}) : {100*frac_ion:6.3f}%")
    print(f"  fully neutral  (x_HI > {edges[-2]:.3f}) : {100*frac_neu:6.3f}%")
    print(f"  partially ionized (in between)  : {100*frac_mid:6.3f}%\n")

    bands = a["partial_bands"]
    print(f"{'x_HI':>7} {'planes':>10} " +
          " ".join(f"{f'part {lo:g}-{hi:g}':>13}" for lo, hi in bands) +
          f" {'width px':>9} {'TV_xy':>8} {'TV_z':>8} {'peak|g|':>8}")
    print("-" * 104)
    for i in range(nb):
        if n[i] < 100:
            continue
        print(f"{centres[i]:7.3f} {int(n[i]):10,} " +
              " ".join(f"{100*partial[b][i]:12.3f}%" for b in range(len(bands))) +
              f" {width[i]:9.3f} {tv_xy[i]:8.4f} {tv_z[i]:8.4f} {peak[i]:8.4f}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    ax = axes[0][0]
    c = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(c, 100 * vh / vtot, width=1.0 / nvb, color=HUES[0], zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("voxel $x_{HI}$", fontsize=10, color=INK_2)
    ax.set_ylabel("% of all voxels (log)", fontsize=10, color=INK_2)
    ax.set_title(f"Voxel population: {100*frac_mid:.2f}% partially ionized",
                 fontsize=11, color=INK, loc="left")

    ax = axes[0][1]
    for b, (lo, hi) in enumerate(bands):
        ax.plot(centres[ok], 100 * partial[b][ok], "-", lw=2, color=HUES[b],
                label=f"${lo:g} < x_{{HI}} < {hi:g}$", zorder=3)
    ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
    ax.set_ylabel("% of voxels in the slice", fontsize=10, color=INK_2)
    ax.set_title("Partially ionized fraction vs ionization state",
                 fontsize=11, color=INK, loc="left")
    leg = ax.legend(frameon=True, fontsize=9, facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax = axes[1][0]
    ax.plot(centres[ok], width[ok], "-", lw=2, color=HUES[0], zorder=3)
    ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
    ax.set_ylabel("transition width (px)", fontsize=10, color=INK_2)
    ax.set_title("Front width vs ionization state", fontsize=11, color=INK,
                 loc="left")

    ax = axes[1][1]
    ax.plot(centres[ok], tv_xy[ok], "-", lw=2, color=HUES[0],
            label="transverse", zorder=3)
    ax.plot(centres[ok], tv_z[ok], "-", lw=2, color=HUES[1],
            label="line of sight", zorder=3)
    ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
    ax.set_ylabel("total variation per voxel", fontsize=10, color=INK_2)
    ax.set_title("Front strength, transverse vs LOS", fontsize=11, color=INK,
                 loc="left")
    leg = ax.legend(frameon=True, fontsize=9, facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(f"21cmFAST ground truth: ionization-front survey over "
                 f"{int(a['n_cones'])} lightcones "
                 f"({int(a['n_planes_total']):,} transverse planes, raw 2340-step LOS)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160, facecolor=SURFACE, bbox_inches="tight")

    # ------------------------------------------------------------------- md
    with open(OUT_MD, "w") as f:
        f.write("# Ionization-front survey of the 21cmFAST ground truth\n\n")
        f.write(f"{int(a['n_cones'])} lightcones, "
                f"{int(a['n_planes_total']):,} transverse planes, "
                f"{vtot:,.0f} voxels. Read from the **raw** lightcones at their "
                f"native 2340-step line of sight -- `cubes_3d.h5` interpolates "
                f"that to 256, so it cannot be used for a sharpness "
                f"measurement.\n\n")
        f.write("## Voxel population\n\n")
        f.write(f"| phase | share |\n|---|---:|\n")
        f.write(f"| fully ionized (x_HI < {edges[1]:.3f}) | {100*frac_ion:.3f}% |\n")
        f.write(f"| fully neutral (x_HI > {edges[-2]:.3f}) | {100*frac_neu:.3f}% |\n")
        f.write(f"| **partially ionized** | **{100*frac_mid:.3f}%** |\n\n")
        f.write("## Accumulated against slice mean x_HI\n\n")
        f.write("| x_HI | planes | " +
                " | ".join(f"partial {lo:g}-{hi:g}" for lo, hi in bands) +
                " | width px | TV transverse | TV LOS | peak grad |\n")
        f.write("|---:|---:|" + "---:|" * (len(bands) + 4) + "\n")
        for i in range(nb):
            if n[i] < 100:
                continue
            f.write(f"| {centres[i]:.3f} | {int(n[i]):,} | " +
                    " | ".join(f"{100*partial[b][i]:.3f}%" for b in range(len(bands))) +
                    f" | {width[i]:.3f} | {tv_xy[i]:.4f} | {tv_z[i]:.4f} | "
                    f"{peak[i]:.4f} |\n")
    json.dump({"n_cones": int(a["n_cones"]),
               "n_planes": int(a["n_planes_total"]),
               "frac_ionized": frac_ion, "frac_neutral": frac_neu,
               "frac_partial": frac_mid},
              open("figures/shared/diagnostics/truth_survey/summary.json", "w"), indent=2)
    print(f"\nwrote {OUT_MD}, {OUT_FIG}")


if __name__ == "__main__":
    main()
