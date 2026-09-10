#!/usr/bin/env python3
"""Initial vs trained waveform, for every initialization preset.

The per-run reports from `viz.learned_waveforms` show one checkpoint's bank in
detail. This asks the cross-run question instead: **does the learned waveform
move away from where it started?**

If training pulls every preset toward one shape, the basis is finding something
in the data and the initialization is only a starting point. If each stays near
its own preset, the bank is barely training and the "learned" basis is really a
fixed basis chosen by hand -- which is what the flat 1.2% spread across presets
(0.1101-0.1114 val_l2) hints at.

Rows are presets; the left column is the global (bottleneck) bank, the right the
local (encoder0) bank. Dashed grey is the initialization, solid is the trained
table, both mean-subtracted and unit-normalized so shape is comparable rather
than scale. Cosine similarity to the init is printed per panel.

Note the random presets cannot be reproduced exactly -- `random` and
`smooth_random` draw from the global RNG at build time -- so their dashed curve
is a fresh draw from the same distribution, shown for shape reference only, and
their similarity number is omitted.

Run: python -m viz.plot_waveform_init_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from learned_waveform_operator import WAVEFORM_INITIALIZATIONS, initial_waveform

CKPT = Path("checkpoints")
OUT = Path("figures/waveforms/init_comparison.png")
RUN = "xhi2d_lwf_both_ph_{init}"
BANKS = [("global (bottleneck)", "fno.bottleneck.0.spectral.bank", 31),
         ("local (encoder0)", "fno.encoder0.spectral.bank", 15)]
STOCHASTIC = {"random", "smooth_random"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
INIT_C = "#8a8880"
TRAINED_C = "#2a78d6"


def unit(t: torch.Tensor) -> torch.Tensor:
    t = t - t.mean()
    return t / torch.linalg.vector_norm(t).clamp_min(1e-12)


def trained_tables(init: str, bank_prefix: str) -> dict[str, torch.Tensor]:
    path = CKPT / RUN.format(init=init) / "final_model_state_dict.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.split(".tables.")[1]: value.float()
            for key, value in state.items()
            if key.startswith(f"{bank_prefix}.tables.")}


def main() -> None:
    inits = list(WAVEFORM_INITIALIZATIONS)
    fig, axes = plt.subplots(len(inits), len(BANKS),
                             figsize=(10.5, 2.05 * len(inits)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    torch.manual_seed(0)
    for row, init in enumerate(inits):
        for col, (label, prefix, bins) in enumerate(BANKS):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            tables = trained_tables(init, prefix)
            start = unit(initial_waveform(bins, init))
            x = torch.arange(bins) + 0.5
            ax.plot(x, start, "--", color=INIT_C, lw=1.4, zorder=2,
                    label="initialization")
            sims = []
            for axis, table in sorted(tables.items()):
                t = unit(table)
                ax.plot(x, t, "-", color=TRAINED_C, lw=1.8, alpha=0.85, zorder=3,
                        label="trained" if axis == "0" else None)
                sims.append(float(torch.dot(t, start)))
            ax.axhline(0, color=GRID, lw=0.8, zorder=1)
            note = ("" if init in STOCHASTIC
                    else "  cos=" + ", ".join(f"{s:+.2f}" for s in sims))
            ax.set_title(f"{init} - {label}{note}", fontsize=9,
                         color=INK, loc="left")
            ax.set_xlim(0, bins)
            ax.tick_params(colors=INK_2, labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            if row == len(inits) - 1:
                ax.set_xlabel("bin", fontsize=9, color=INK_2)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Learned waveform: initialization (dashed) vs trained (solid)",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.997)
    fig.text(0.01, 0.001,
             "lwf/lwf with phase mixing, 100 epochs, 2-D x_HI. Both axes of each "
             "bank overplotted. Curves mean-subtracted and unit-normalized; "
             "random/smooth_random dashed curves are fresh draws, not the actual init.",
             fontsize=7.5, color=INK_2, ha="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.018, 1, 0.985))
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}")

    print(f"\n{'init':16s}{'bank':22s}{'cos(trained, init)':>22}")
    for init in inits:
        for label, prefix, bins in BANKS:
            start = unit(initial_waveform(bins, init))
            sims = [float(torch.dot(unit(t), start))
                    for _, t in sorted(trained_tables(init, prefix).items())]
            shown = "n/a (stochastic init)" if init in STOCHASTIC else \
                ", ".join(f"{s:+.3f}" for s in sims)
            print(f"{init:16s}{label:22s}{shown:>22}")


if __name__ == "__main__":
    main()
