"""Parameters vs inference throughput for every operator pairing.

Reads figures/operator_variant_benchmark.json (tests/bench_operator_variants.py).

Colour encodes the *local* operator, because that is the finding: within each
local family the three global operators differ by under 1% in speed, so the
local branch alone sets the cost. Only three hues are used -- a scatter puts
every pair on screen at once, and the validated categorical palette clears the
all-pairs colour-vision floors for three slots, not more. Everything outside the
3x3 matrix (the plain-FNO and U-FNO baselines, sirenfno, cnn) is folded into
neutral ink with its own marker rather than given a fourth and fifth hue.

Every point is directly labelled, which is also what discharges the palette
validator's contrast WARN on the aqua slot.

Run: python -m viz.plot_operator_benchmark
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "figures/operator_variant_benchmark.json"
OUT = "figures/operator_params_vs_throughput.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
# Validated for all-pairs in light mode (worst CVD dE 9.2, normal-vision 24.0).
LOCAL_HUE = {"fourier": "#2a78d6", "wavelet": "#eb6834", "hadamard": "#1baf7a",
             "siren_hadamard": "#9d4edd", "cnn": "#d4a017",
             "siren_fourier": "#00a6a6"}
LOCAL_NAME = {"fourier": "local FNO", "wavelet": "local WNO",
              "hadamard": "local WHNO", "siren_hadamard": "local SWHNO",
              "cnn": "local CNN (U-Net path)", "siren_fourier": "local SirenFNO"}
TRAINED = "local fno / global whno"      # the configuration actually trained

# (dx, dy) in points, per label, to keep the dense cluster legible.
NUDGE = {
    "FNO (plain)": (0, 13), "U-FNO": (0, -20),
    "local fno / global fno": (10, 9), "local fno / global wno": (-12, -19),
    "local fno / global whno": (12, -19),
    "local wno / global fno": (12, 11), "local wno / global wno": (-13, 11),
    "local wno / global whno": (13, 11),
    "local whno / global fno": (11, 10), "local whno / global wno": (-14, -19),
    "local whno / global whno": (12, -19),
    "local sirenfno / global sirenfno": (-2, -21),
    "local cnn / global fno": (0, 13),
}
SHORT = {"FNO (plain)": "FNO", "U-FNO": "U-FNO",
         "local sirenfno / global sirenfno": "SirenFNO"}
DEFAULT_NUDGE = (0, 12)      # any variant not in NUDGE still gets a label
# The matrix is the square local x global sweep. cnn and siren_fourier have
# their own hue (they are distinct local families) but sit outside it, so
# membership drives the marker, not the colour.
MATRIX_OPS = ("fourier", "wavelet", "hadamard", "siren_hadamard")


def short(v: str) -> str:
    if v in SHORT:
        return SHORT[v]
    if "global " not in v:
        return v
    g = v.split("global ")[1]
    if v.startswith("local cnn"):
        return f"CNN / {g}"                # cnn has its own hue but is the
    return f"…/ {g}"                       # local family is already the colour


def main() -> None:
    meta = json.load(open(SRC))
    rows = meta["rows"]

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    seen = set()
    for r in rows:
        loc = r["local"]
        matrix = loc in MATRIX_OPS and r["global"] in MATRIX_OPS
        colour = LOCAL_HUE.get(loc, INK_MUTED)
        marker = "o" if matrix else "s"
        lbl = None
        if loc in LOCAL_NAME and loc not in seen:
            lbl = LOCAL_NAME[loc]
            seen.add(loc)
        elif loc not in LOCAL_NAME and "other" not in seen:
            lbl = "outside the matrix"
            seen.add("other")
        is_trained = r["variant"] == TRAINED
        ax.plot(r["params_real"], r["slices_per_s"], marker, ms=13 if is_trained else 10,
                color=colour, mec=SURFACE, mew=2, label=lbl, zorder=3,
                linestyle="none")
        if is_trained:                      # ring the configuration in use
            ax.plot(r["params_real"], r["slices_per_s"], "o", ms=22, mfc="none",
                    mec=colour, mew=1.6, alpha=0.55, zorder=2, linestyle="none")
        dx, dy = NUDGE.get(r["variant"], DEFAULT_NUDGE)
        ax.annotate(short(r["variant"]), (r["params_real"], r["slices_per_s"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=8.5, color=INK_2, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters  (complex weights counted as two, "
                  "as in the training logs)", fontsize=10, color=INK_2)
    ax.set_ylabel("Inference throughput  (slices / s)", fontsize=10, color=INK_2)
    ax.set_title("Local/global operator pairings: cost vs size",
                 fontsize=13, color=INK, pad=34, loc="left")
    ax.text(0.0, 1.012,
            f"{meta['device']}, batch {meta['batch']} x {meta['in_channels']} x "
            f"{meta['resolution']}², median of {meta['repeats']} forward passes"
            "\ncolour is the LOCAL operator; square = outside the local x global matrix",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")

    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)

    # Name the ringed marker, otherwise the ring is unexplained.
    trained = next(r for r in rows if r["variant"] == TRAINED)
    ax.annotate("the configuration trained\n(whno_glob)",
                xy=(trained["params_real"], trained["slices_per_s"]),
                xytext=(34, -46), textcoords="offset points",
                fontsize=8.5, color=INK_2, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))

    ax.annotate("smaller and faster", xy=(0.055, 0.955), xycoords="axes fraction",
                xytext=(0.20, 0.955), textcoords="axes fraction",
                fontsize=9, color=INK_MUTED, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.2))

    ax.set_ylim(440, 820)
    ax.set_xlim(2.2e5, 4.2e7)

    # Fixed legend order: the three matrix families, then the fold-in.
    handles, labels = ax.get_legend_handles_labels()
    by_label = {lab: h for h, lab in zip(handles, labels)}
    order = ["local FNO", "local WNO", "local WHNO", "outside the 3x3 matrix"]
    leg = ax.legend([by_label[l] for l in order if l in by_label],
                    [l for l in order if l in by_label],
                    loc="lower right", frameon=True, fontsize=9.5,
                    facecolor=SURFACE, edgecolor=GRID, borderpad=0.8)
    for txt in leg.get_texts():
        txt.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
