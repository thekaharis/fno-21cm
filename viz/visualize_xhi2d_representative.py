#!/usr/bin/env python3
"""Render representative held-out redshift slices for one 2-D x_HI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from viz.compare_xhi2d_models import load_run, predict, representative_indices


DEFAULT_QUANTILES = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]


def render_slices(
    truth: np.ndarray,
    prediction: np.ndarray,
    slice_info: list[dict],
    run_label: str,
    output: Path,
) -> None:
    error = prediction - truth
    error_limit = max(0.05, float(np.quantile(np.abs(error), 0.99)))
    figure, axes = plt.subplots(
        len(slice_info), 3,
        figsize=(12.5, 3.25 * len(slice_info)),
        constrained_layout=True,
        squeeze=False,
    )
    field_image = error_image = None
    for row, info in enumerate(slice_info):
        rmse = float(np.sqrt(np.mean(error[row] ** 2)))
        mae = float(np.mean(np.abs(error[row])))
        panels = (
            (truth[row], "viridis", 0.0, 1.0),
            (prediction[row], "viridis", 0.0, 1.0),
            (error[row], "RdBu_r", -error_limit, error_limit),
        )
        for column, (image, cmap, vmin, vmax) in enumerate(panels):
            rendered = axes[row, column].imshow(
                image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            if column < 2:
                field_image = rendered
            else:
                error_image = rendered
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            f"q={info['quantile']:.2f}  z={info['z']:.2f}\n"
            f"cone={info['cone_id']}  mean={info['xhi_mean']:.2f}",
            fontsize=9,
        )
        axes[row, 1].text(
            0.02, 0.98, f"RMSE={rmse:.3f}\nMAE={mae:.3f}",
            transform=axes[row, 1].transAxes, va="top", ha="left",
            fontsize=8.5, color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )

    axes[0, 0].set_title("Truth $x_{HI}$")
    axes[0, 1].set_title("Prediction $x_{HI}$")
    axes[0, 2].set_title("Prediction - truth")
    assert field_image is not None and error_image is not None
    figure.colorbar(
        field_image, ax=axes[:, :2], shrink=0.8, label="$x_{HI}$"
    )
    figure.colorbar(
        error_image, ax=axes[:, 2], shrink=0.8, label="signed error"
    )
    figure.suptitle(
        f"{run_label}: representative held-out redshift slices", fontsize=14
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/xhi2d_representative"),
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=DEFAULT_QUANTILES,
    )
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()
    if not args.quantiles or any(not 0 <= value <= 1 for value in args.quantiles):
        parser.error("--quantiles must contain values in [0, 1]")

    run = load_run(args.run.name, args.run)
    if run is None:
        raise SystemExit(f"completed run artifacts not found in {args.run}")
    metadata = run.metadata
    normalization = metadata.get("parameter_normalization")
    cache = SliceCache(
        metadata["dataset"]["cache_file"],
        input_features=metadata["input_features"]["name"],
        parameter_normalization=(
            ParameterNormalization.from_dict(normalization)
            if normalization is not None else None
        ),
    )
    test_cones = np.asarray(metadata["split"]["test_cone_ids"], dtype=np.int64)
    test_indices = np.flatnonzero(np.isin(cache.cone_id, test_cones))
    indices = representative_indices(cache, test_indices, args.quantiles)
    samples = [cache[index] for index in indices]
    inputs = torch.stack([sample["x"] for sample in samples])
    truth = torch.stack([sample["y"] for sample in samples]).numpy()[:, 0]
    prediction = predict(run, inputs, args.device)

    slice_info = [
        {
            "quantile": float(quantile),
            "global_index": int(index),
            "cone_id": int(cache.cone_id[index]),
            "z": float(cache.z[index]),
            "xhi_mean": float(cache.xHI_mean[index]),
            "rmse": float(np.sqrt(np.mean(
                (prediction[row] - truth[row]) ** 2
            ))),
            "mae": float(np.mean(np.abs(prediction[row] - truth[row]))),
        }
        for row, (quantile, index) in enumerate(zip(args.quantiles, indices))
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    figure_path = args.output / "representative_z_slices.png"
    render_slices(truth, prediction, slice_info, run.label, figure_path)
    (args.output / "selected_slices.json").write_text(
        json.dumps(slice_info, indent=2) + "\n"
    )
    (args.output / "run_summary.json").write_text(json.dumps({
        "run": str(args.run.resolve()),
        "model_kind": run.kind,
        "final_report": run.report,
        "figure": str(figure_path),
    }, indent=2) + "\n")
    print(f"Representative slice visualization: {figure_path}")


if __name__ == "__main__":
    main()
