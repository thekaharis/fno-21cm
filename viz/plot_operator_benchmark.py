"""Parameters vs inference throughput for every operator pairing.

Reads figures/summary/operator_variant_benchmark.json (tests/bench_operator_variants.py).

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

SRC = "figures/summary/operator_variant_benchmark.json"
OUT = "figures/summary/operator_params_vs_throughput.png"

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

SHORT = {"FNO (plain)": "FNO", "U-FNO": "U-FNO",
         "local sirenfno / global sirenfno": "SirenFNO"}
# The matrix is the square local x global sweep. cnn and siren_fourier have
# their own hue (they are distinct local families) but sit outside it, so
# membership drives the marker, not the colour.
MATRIX_OPS = ("fourier", "wavelet", "hadamard", "siren_hadamard")


def short(v: str) -> str:
    """Label text. Colour already encodes the local operator, so inside the
    matrix only the global slot needs naming."""
    if v in SHORT:
        return SHORT[v]
    if "global " not in v:
        return v
    return v.split("global ")[1]


def offsets(rows) -> dict[str, tuple[float, float]]:
    """Alternate labels above/below within each local family.

    Throughput varies under 1% across the global slot, so a family's four
    points form a near-horizontal band and a fixed offset stacks every label on
    its neighbour. Sorting by x and flipping the sign staggers them, which is
    what the old hand-fitted NUDGE table did by hand for 13 points and could
    not do for 23.
    """
    out, groups = {}, {}
    for r in rows:
        groups.setdefault(r["local"], []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: r["params_real"])
        for i, r in enumerate(members):
            out[r["variant"]] = (0, 11) if i % 2 == 0 else (0, -19)
    return out


def main() -> None:
    meta = json.load(open(SRC))
    rows = meta["rows"]

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    nudge = offsets(rows)
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
        ax.plot(r["params_real"], r["slices_per_s"], marker, ms=10,
                color=colour, mec=SURFACE, mew=2, label=lbl, zorder=3,
                linestyle="none")
        dx, dy = nudge[r["variant"]]
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

    ax.annotate("smaller and faster", xy=(0.055, 0.955), xycoords="axes fraction",
                xytext=(0.20, 0.955), textcoords="axes fraction",
                fontsize=9, color=INK_MUTED, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.2))

    ax.set_ylim(440, 820)
    ax.set_xlim(2.2e5, 4.2e7)

    # Legend order is derived, not hardcoded: the previous fixed list silently
    # dropped every family added after it was written.
    handles, labels = ax.get_legend_handles_labels()
    by_label = {lab: h for h, lab in zip(handles, labels)}
    order = [LOCAL_NAME[k] for k in
             ("fourier", "wavelet", "hadamard", "siren_hadamard",
              "siren_fourier", "cnn")] + ["outside the matrix"]
    order += [l for l in by_label if l not in order]       # never lose one
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
