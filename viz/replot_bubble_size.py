#!/usr/bin/env python3
"""Re-render BSD distribution figures from a saved bubble_size_results.npz.

The evaluator's own figure draws a 16-84 percentile band per model, which turns
into an unreadable pile of overlapping shadows once more than a couple of models
share a panel. This script replots the same arrays with the mean curves only.

Examples
--------
Default five-model comparison::

    python -m viz.replot_bubble_size \
      --npz ../../Figures/final_eval/matrix/bsd/bubble_size_results.npz \
      --out ../../Figures/final_eval/matrix/bsd/bsd_clean

Pick models and labels explicitly::

    python -m viz.replot_bubble_size --npz results.npz --out fig \
      --models ufno_plain_gnorm="U-FNO" fno_whno_plain="FNO / WHNO"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# local/global operator pairs as written by slurm/train_3d_matrix.sbatch
DEFAULT_MODELS = [
    ("ufno_plain_gnorm", "U-FNO"),
    ("cnn_swhno_plain", "CNN / SWHNO"),
    ("sfno_swhno_bw48om60", "SFNO / SWHNO"),
    ("swhno_swhno_bw48om60", "SWHNO / SWHNO"),
    ("fno_whno_plain", "FNO / WHNO"),
]

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
          "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def stage_title(label: str) -> str:
    """Turn the stored ASCII stage label into a math-set axis title."""
    label = str(label)
    if label.startswith("xbar_HI"):
        return r"$\bar{x}_{\rm HI}$" + label[len("xbar_HI"):]
    if label.startswith("active"):
        return "all active stages" + label[len("active"):]
    return label


def parse_models(specs: list[str] | None, available: set[str]):
    if not specs:
        missing = [name for name, _ in DEFAULT_MODELS if name not in available]
        if missing:
            raise SystemExit(
                f"npz is missing default models {missing}; "
                f"available: {sorted(available)}"
            )
        return list(DEFAULT_MODELS)
    models = []
    for spec in specs:
        name, _, label = spec.partition("=")
        if name not in available:
            raise SystemExit(f"unknown model {name!r}; available: {sorted(available)}")
        models.append((name, label or name))
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output stem; .png and .pdf are written")
    parser.add_argument("--models", nargs="+", default=None,
                        help="name[=label] entries, in plot order")
    parser.add_argument("--stages", nargs="+", type=int, default=None,
                        help="stage indices to draw (default: all)")
    parser.add_argument("--stat", default="mean", choices=["mean", "med"],
                        help="which stored central curve to draw")
    parser.add_argument("--no-markers", action="store_true",
                        help="drop the underflow/censored edge markers")
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--summary", type=Path, default=None,
                        help="also write the Wasserstein/bias bar summary to this stem")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    available = {key.split("/", 1)[0] for key in data.files}
    models = parse_models(args.models, available)

    first = models[0][0]
    labels = [str(s) for s in data[f"{first}/stage_labels"]]
    centers = data[f"{first}/radius_centers_mpc"]
    edges = data[f"{first}/radius_edges_mpc"]
    stages = args.stages if args.stages is not None else list(range(len(labels)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = min(args.ncols, len(stages))
    n_rows = int(np.ceil(len(stages) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.9 * n_rows),
                             squeeze=False)
    axes = axes.ravel()

    for panel, stage_index in enumerate(stages):
        ax = axes[panel]
        ax.plot(centers, data[f"{first}/truth_mass_{args.stat}"][stage_index],
                color="black", linewidth=1.2, label="21cmFAST truth", zorder=5)
        if not args.no_markers:
            ax.scatter(edges[0],
                       data[f"{first}/truth_underflow_fraction_{args.stat}"][stage_index],
                       color="black", marker="v", s=26, zorder=5)
            ax.scatter(edges[-1],
                       data[f"{first}/truth_censored_fraction_{args.stat}"][stage_index],
                       color="black", marker="^", s=26, zorder=5)
        for model_index, (name, label) in enumerate(models):
            color = COLORS[model_index % len(COLORS)]
            ax.plot(centers, data[f"{name}/pred_mass_{args.stat}"][stage_index],
                    color=color, linewidth=0.85, label=label)
            if not args.no_markers:
                ax.scatter(edges[0],
                           data[f"{name}/pred_underflow_fraction_{args.stat}"][stage_index],
                           color=color, marker="v", s=20)
                ax.scatter(edges[-1],
                           data[f"{name}/pred_censored_fraction_{args.stat}"][stage_index],
                           color=color, marker="^", s=20)
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.set(title=stage_title(labels[stage_index]),
               xlabel="MFP distance $R$ [cMpc]",
               ylabel="probability per log bin")
        ax.grid(alpha=0.2)

    for ax in axes[len(stages):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=9, framealpha=0.9)
    if not args.no_markers:
        axes[0].text(0.02, 0.72,
                     "$\\nabla$: no ionized pixels\n$\\Delta$: censored at box length",
                     transform=axes[0].transAxes, va="top", fontsize=7.5, color="0.35")
    fig.suptitle("Transverse mean-free-path ionized-bubble size distributions")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        path = args.out.with_suffix(suffix)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    for path in written:
        print(f"wrote {path}")

    if args.summary is not None:
        plot_summary(data, models, labels, stages, args.summary, args.dpi)


def plot_summary(data, models, labels, stages, out_stem: Path, dpi: int):
    """Median restricted-Wasserstein and mean-MFP bias, same model selection."""
    import matplotlib.pyplot as plt

    x = np.arange(len(stages))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for model_index, (name, label) in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        color = COLORS[model_index % len(COLORS)]
        axes[0].bar(x + offset,
                    data[f"{name}/restricted_wasserstein_mpc_med"][stages],
                    width, label=label, color=color)
        axes[1].bar(x + offset,
                    100 * data[f"{name}/relative_mean_bias_med"][stages],
                    width, label=label, color=color)
    axes[0].set(ylabel="restricted Wasserstein distance [cMpc]",
                title="BSD distribution error")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="restricted mean MFP bias [%]", title="Bubble-size bias")
    for ax in axes:
        ax.set_xticks(x, [stage_title(labels[i]) for i in stages],
                      rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
