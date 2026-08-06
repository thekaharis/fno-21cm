"""6x6 panel: representative slices down the rows, models across the columns.

Columns are TRUTH followed by the five losses under comparison; rows are six
representative test slices picked at redshift quantiles spanning reionization
(the same selection ``visualize_xhi2d_representative`` uses), so the grid shows
how each loss behaves from early to late times rather than on one lucky slice.

Each panel is annotated with its transition width and, for predictions, the
slice RMSE -- the two numbers that disagree with each other in this comparison:
the L2-free runs are wider apart on RMSE and much closer to the truth on width.

Run: sbatch slurm/compare_wall_models_grid.sbatch
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legacy.xhi2d import field_metrics as fm
from legacy.xhi2d.slice_eval import gather, open_run, split_indices
from viz.compare_xhi2d_models import representative_indices
from viz.visualize_xhi2d_representative import DEFAULT_QUANTILES

# Dropped from the first version: expwall s=4 (collapsed to uniform fields,
# RMSE 0.511) and the H1 seminorm (strictly dominated by the L2 baseline on
# RMSE, wall placement and width alike). Replaced by the two alternating-refit
# contrast runs, whose final fitted schedules ride along in their checkpoints.
MODELS = [
    ("L2 baseline", "checkpoints/xhi2d_whno_glob_lr3e4"),
    ("expwall s=16", "checkpoints/xhi2d_whno_glob_expwall16"),
    ("expwall s=8", "checkpoints/xhi2d_whno_glob_expwall8"),
    ("ctr-refit (L2)", "checkpoints/xhi2d_whno_glob_ctrrefit"),
    ("ctr-refit +SWD", "checkpoints/xhi2d_whno_glob_ctrrefit_swd"),
]
OUT = "figures/xhi2d_wall_models_grid_v2.png"
CMAP = "magma"


def load(run, indices):
    for ck in ("final_model_state_dict.pt", "model_state_dict.pt"):
        try:
            return gather(run, indices=indices, checkpoint=ck), ck
        except Exception:                                       # noqa: BLE001
            continue
    return None, None


def main() -> None:
    ref = MODELS[0][1]
    _, meta, cache = open_run(ref, device="cuda")
    test_idx = split_indices(meta, cache, "test")
    chosen = representative_indices(cache, test_idx, DEFAULT_QUANTILES)
    idx = np.asarray(chosen)
    print(f"representative slices: {idx.tolist()}")

    panels, labels = [], []
    truth = z = xhi = None
    for name, run in MODELS:
        got, ck = load(run, idx)
        if got is None:
            print(f"  {name}: unavailable, skipped")
            continue
        panels.append(got.pred.cpu().numpy())
        labels.append(f"{name}\n({'final' if ck.startswith('final') else 'ckpt'})")
        if truth is None:
            truth = got.truth.cpu().numpy()
            z, xhi = got.z.cpu().numpy(), got.xhi.cpu().numpy()
        print(f"  {name}: {ck}")

    cols = [("TRUTH", truth)] + list(zip(labels, panels))
    nrow, ncol = len(idx), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.5 * nrow),
                             squeeze=False)
    for r in range(nrow):
        for c, (title, field) in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(field[r], vmin=0.0, vmax=1.0, cmap=CMAP,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            w = float(fm.width_px(torch.as_tensor(field[r])[None]).mean())
            note = f"w={w:.2f}"
            if c > 0:
                note += f"  rmse={np.sqrt(((field[r]-truth[r])**2).mean()):.3f}"
            ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top",
                    fontsize=7, color="white",
                    bbox=dict(fc="black", alpha=0.45, pad=1.4, lw=0))
        axes[r][0].set_ylabel(f"z={z[r]:.1f}\n$x_{{HI}}$={xhi[r]:.2f}",
                              fontsize=8)

    fig.suptitle("2-D $x_{HI}$: representative test slices by loss function "
                 "(w = transition width in px; truth is the reference)",
                 fontsize=11)
    fig.colorbar(axes[0][0].images[0], ax=axes, fraction=0.012, pad=0.01,
                 label="$x_{HI}$")
    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}  ({nrow}x{ncol})")


if __name__ == "__main__":
    main()
