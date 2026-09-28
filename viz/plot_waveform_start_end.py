"""Illustration: learned mother waveform at initialization vs after training.

Uses the identity-ablation runs, where the local branch is removed and the
learned-waveform global slot is the only spectral operator, so the waveform the
model settles on is not confounded by a local operator compensating for it.

Both curves are shown zero-mean and unit-RMS, and the converged curve's sign is
chosen to match the initial one. Neither changes the basis the operator builds:
candidate columns are normalized to unit norm, so overall scale is invisible to
the model, and a sign flip of the mother waveform flips every candidate, which
spans the same space.

  python -m viz.plot_waveform_start_end --out figures/shared/diagnostics/waveforms/start_to_converged.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RUNS = [("sine", "xhi2d_id_lwf_grid_s0"),
        ("square", "xhi2d_id_lwf_grid_square_s0"),
        ("sawtooth", "xhi2d_id_lwf_grid_sawtooth_s0"),
        ("random", "xhi2d_id_lwf_grid_random_s0")]
KEY = "fno.bottleneck.0.spectral.bank.tables.{}"


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    rms = np.sqrt((v ** 2).mean())
    return v / rms if rms > 0 else v


def corr(a, b):
    return float(np.dot(normalize(a), normalize(b)) / len(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("checkpoints/2d_xhi/id_lwf"))
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/shared/diagnostics/waveforms/start_to_converged.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(RUNS), figsize=(4.1 * len(RUNS), 3.6), sharey=True)
    for ax, (label, run) in zip(axes, RUNS):
        d = args.root / run
        init = torch.load(d / "waveform_initial_tables.pt", map_location="cpu",
                          weights_only=False)[KEY.format(args.axis)].numpy()
        final = torch.load(d / "final_model_state_dict.pt", map_location="cpu",
                           weights_only=True)[KEY.format(args.axis)].numpy()
        a, b = normalize(init), normalize(final)
        if np.dot(a, b) < 0:
            b = -b
        n = len(a)
        edges = np.linspace(0.0, 1.0, n + 1)
        sine = np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
        ax.stairs(a, edges, color="0.55", lw=1.8, ls="--", label="initial")
        ax.stairs(b, edges, color="#c0392b", lw=2.4, label="converged")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(f"{label} start", fontsize=13)
        ax.text(0.03, 0.04,
                f"similarity to a sine\n{abs(corr(init, sine)):.3f}  →  {abs(corr(final, sine)):.4f}",
                transform=ax.transAxes, fontsize=9.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85"))
        ax.set_xlabel("position within one period")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("waveform amplitude\n(zero mean, unit RMS)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Learned basis waveform: initialization vs after training", fontsize=14,
                 x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
