#!/usr/bin/env python3
"""Learned waveform shape at EQUAL epochs, across waveform learning-rate settings.

Comparing a ratio-0.1 table at epoch 99 with a ratio-1.0 table at epoch 50 would
confound step size with training time, so every panel holds the epoch fixed and
varies only the schedule. Columns are epochs, rows are preset x bank; within a
panel each series is one training setting, drawn against the initialization.

Checkpoint for (run, epoch) is resolved in this order:
  1. snapshots/epNNN_model_state_dict.pt   (copies of periodic saves -- the
     periodic model_state_dict.pt is overwritten every save interval)
  2. the file named by manifest.pt's "model" field, when its "epoch" matches.
     The manifest is per directory and records WHICH state dict it describes:
     after the final save it names final_model_state_dict.pt (epoch 99), while
     model_state_dict.pt still holds the last periodic save (epoch 75). Pairing
     the manifest epoch with model_state_dict.pt regardless of that field would
     load epoch-75 weights under an epoch-99 label.
  3. final_model_state_dict.pt for the last epoch
A setting without a checkpoint at that epoch is left out of that panel and
listed in the printed table, never substituted with a nearby epoch.

Curves are mean-subtracted and unit-normalized: the QR orthonormalizes the
sampled candidates, so the table's overall scale does not reach the basis --
only its shape does. Axis 0 is solid, axis 1 dotted. The initialization is read
from `waveform_initial_tables.pt` when the run saved it, otherwise recomputed
from the deterministic preset formula (so this script refuses stochastic presets
without a saved init).

Run:
    python -m viz.plot_waveform_lr_comparison --presets sine square --epochs 75 99
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from learned_waveform_operator import initial_waveform

CKPT = Path("checkpoints")
SERIES = [  # label, run-directory template, colour
    ("ratio 0.1, joint", "xhi2d_lwf_both_ph_{preset}", "#8a8880"),
    ("ratio 1.0, joint", "xhi2d_lwf_both_ph_{preset}_lr1", "#2a78d6"),
    ("ratio 0.1, alternating 1:5", "xhi2d_lwf_both_ph_{preset}_altall", "#eb6834"),
]
BANKS = [("global (bottleneck)", "fno.bottleneck.0.spectral.bank", 31),
         ("local (encoder0)", "fno.encoder0.spectral.bank", 15)]
STOCHASTIC = {"random", "smooth_random"}
LAST_EPOCH = 99

SURFACE, INK, INK_2, GRID, INIT_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#0b0b0b"


def unit(t: torch.Tensor) -> torch.Tensor:
    t = t.double() - t.double().mean()
    return t / torch.linalg.vector_norm(t).clamp_min(1e-12)


def resolve(run: Path, epoch: int) -> Path | None:
    snap = run / "snapshots" / f"ep{epoch:03d}_model_state_dict.pt"
    if snap.is_file():
        return snap
    manifest = run / "manifest.pt"
    if manifest.is_file():
        saved = torch.load(manifest, map_location="cpu", weights_only=False)
        named = run / str(saved.get("model", ""))
        if int(saved.get("epoch", -1)) == epoch and named.is_file():
            return named
    final = run / "final_model_state_dict.pt"
    if epoch == LAST_EPOCH and final.is_file():
        return final
    return None


def tables(state: dict, prefix: str) -> dict[str, torch.Tensor]:
    return {k.split(".tables.")[1]: v for k, v in state.items()
            if k.startswith(f"{prefix}.tables.")}


def init_tables(run: Path, preset: str, prefix: str, bins: int) -> dict[str, torch.Tensor] | None:
    saved = run / "waveform_initial_tables.pt"
    if saved.is_file():
        blob = torch.load(saved, map_location="cpu", weights_only=False)
        blob = blob.get("initial_tables", blob) if isinstance(blob, dict) else blob
        found = tables(blob, prefix)
        if found:
            return found
    if preset in STOCHASTIC:
        return None     # the random draw was not saved, so there is no reference to compare to
    return {"0": initial_waveform(bins, preset), "1": initial_waveform(bins, preset)}


SHORT = {"ratio 0.1, joint": "r0.1", "ratio 1.0, joint": "r1.0",
         "ratio 0.1, alternating 1:5": "alt"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--presets", nargs="+", default=["sine", "square"])
    ap.add_argument("--epochs", nargs="+", type=int, default=[75, LAST_EPOCH])
    ap.add_argument("--out", type=Path, default=Path("figures/shared/diagnostics/waveforms/lr_comparison.png"))
    args = ap.parse_args(argv)

    rows = [(p, b) for p in args.presets for b in BANKS]
    fig, axes = plt.subplots(len(rows), len(args.epochs), squeeze=False,
                             figsize=(max(5.6 * len(args.epochs), 10.0), 2.6 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    report = []

    for r, (preset, (bank_label, prefix, bins)) in enumerate(rows):
        for c, epoch in enumerate(args.epochs):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            x = torch.arange(bins) + 0.5
            init_drawn, cos_note = False, []
            for label, template, colour in SERIES:
                run = CKPT / template.format(preset=preset)
                ckpt = resolve(run, epoch) if run.is_dir() else None
                if ckpt is None:
                    report.append((preset, bank_label, epoch, label, None, None, "no checkpoint"))
                    cos_note.append(f"{SHORT.get(label, label)} n/a")
                    continue
                start = init_tables(run, preset, prefix, bins)
                if start is None:
                    report.append((preset, bank_label, epoch, label, None, None, "no saved init"))
                    cos_note.append(f"{SHORT.get(label, label)} no init")
                    continue
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                trained = tables(state, prefix)
                if not init_drawn:
                    ax.plot(x, unit(start["0"]), "--", color=INIT_C, lw=1.1, alpha=0.55, zorder=2)
                    init_drawn = True
                cos = {}
                for axis, style in (("0", "-"), ("1", ":")):
                    if axis not in trained:
                        continue
                    t, t0 = trained[axis], start[axis]
                    cos[axis] = float(torch.dot(unit(t), unit(t0)))
                    dist = float(torch.linalg.vector_norm(t.double() - t0.double())
                                 / torch.linalg.vector_norm(t0.double()).clamp_min(1e-12))
                    ax.plot(x, unit(t), style, color=colour, lw=1.8, zorder=3)
                    report.append((preset, bank_label, epoch, label, axis, (cos[axis], dist), ckpt.name))
                cos_note.append(f"{SHORT.get(label, label)} {cos.get('0', float('nan')):.3f}/"
                                f"{cos.get('1', float('nan')):.3f}")
            ax.axhline(0, color=GRID, lw=0.8, zorder=1)
            ax.set_title(f"{preset} - {bank_label} - epoch {epoch}", fontsize=9.5,
                         color=INK, loc="left", pad=15)
            # Cosines live in a subtitle, not a legend, so nothing is drawn over the curves.
            ax.text(0.0, 1.02, "cos to init x/y:  " + "   ".join(cos_note),
                    transform=ax.transAxes, fontsize=7.4, color=INK_2, va="bottom")
            ax.set_xlim(0, bins)
            ax.tick_params(colors=INK_2, labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            if r == len(rows) - 1:
                ax.set_xlabel("bin", fontsize=9, color=INK_2)

    handles = [Line2D([], [], color=INIT_C, lw=1.1, ls="--", alpha=0.55)]
    labels = ["initialization"]
    for label, _, colour in SERIES:
        handles.append(Line2D([], [], color=colour, lw=2.0))
        labels.append(f"{label} ({SHORT.get(label, label)})")
    handles += [Line2D([], [], color=INK_2, lw=1.4, ls="-"), Line2D([], [], color=INK_2, lw=1.4, ls=":")]
    labels += ["axis 0 (x): solid", "axis 1 (y): dotted"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.962))
    fig.suptitle("Learned waveform at equal epochs, by waveform learning rate",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.997)
    fig.text(0.01, 0.002, "Mean-subtracted, unit-normalized tables (scale does not reach "
             "the QR basis). lwf/lwf with phase mixing, 2-D x_HI, base lr 3e-4.",
             fontsize=7.5, color=INK_2, ha="left", va="bottom")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.915))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}\n")
    print(f"{'preset':8s}{'bank':22s}{'ep':>4}  {'setting':28s}{'axis':>5}{'cos->init':>10}{'rel dist':>10}  source")
    for preset, bank, epoch, label, axis, vals, src in report:
        if vals is None:
            print(f"{preset:8s}{bank:22s}{epoch:>4}  {label:28s}{'-':>5}{'-':>10}{'-':>10}  {src}")
        else:
            print(f"{preset:8s}{bank:22s}{epoch:>4}  {label:28s}{axis:>5}{vals[0]:>10.4f}{vals[1]:>10.4f}  {src}")


if __name__ == "__main__":
    main()
