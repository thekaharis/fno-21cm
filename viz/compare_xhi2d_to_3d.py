#!/usr/bin/env python3
"""Compare 2-D LocalWNO slices with 3-D UFNO and LocalFNO predictions.

Slices are selected from physical cones common to every model's chosen split
(--cone-split: train | test | heldout) and span ionized through neutral
stages. Note the models were trained with independent splits, so only the
intersection keeps all three on equal footing. The 3-D predictions are linearly
interpolated onto each native 2-D slice redshift before computing RMSE and R^2.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.dataset import SliceCache
from dataset.dataset_3d import (
    InputFeatures,
    LightconeCubeCache,
    ParameterNormalization,
)
from modeling import ModelConfig, TrainerModel, build_3d_model, load_checkpoint
from util.run_metadata import load_run_metadata
from viz.compare_xhi2d_models import build_model as build_2d_model, load_run


STAGES = (
    ("ionized", 0.03, 0.00, 0.10),
    ("mostly ionized", 0.20, 0.10, 0.35),
    ("mid reionization", 0.50, 0.35, 0.65),
    ("mostly neutral", 0.80, 0.65, 0.90),
    ("neutral", 0.97, 0.90, 1.000001),
)
MODEL_ORDER = ("localwno", "ufno", "localfno")
MODEL_LABELS = {
    "localwno": "LocalWNO (2-D)",
    "ufno": "UFNO (3-D)",
    "localfno": "LocalFNO (3-D)",
}
MODEL_COLORS = {
    "localwno": "#168aad",
    "ufno": "#7b2cbf",
    "localfno": "#f4a261",
}


def strict_load(model: torch.nn.Module, checkpoint: Path) -> None:
    result = load_checkpoint(model, checkpoint)
    if result.missing or result.unexpected or result.matched != result.total:
        raise RuntimeError(
            f"incomplete checkpoint load for {checkpoint}: "
            f"matched={result.matched}/{result.total}, "
            f"missing={result.missing}, unexpected={result.unexpected}"
        )


def select_stage_indices(
    cone_ids: np.ndarray,
    redshifts: np.ndarray,
    means: np.ndarray,
    eligible_cones: set[int],
    slices_per_stage: int = 1,
) -> list[int]:
    if slices_per_stage <= 0:
        raise ValueError("slices_per_stage must be positive")
    eligible = np.flatnonzero(
        np.isin(cone_ids, np.fromiter(eligible_cones, dtype=np.int64))
        & (redshifts >= 5.0) & (redshifts <= 25.0)
    )
    if len(eligible) < len(STAGES) * slices_per_stage:
        raise ValueError("not enough aligned slices in the common cone set")
    selected = []
    used_cones: set[int] = set()
    for name, target, lower, upper in STAGES:
        stage = eligible[(means[eligible] >= lower) & (means[eligible] < upper)]
        if len(stage) < slices_per_stage:
            raise ValueError(
                f"stage {name!r} has only {len(stage)} eligible slices; "
                f"need {slices_per_stage}"
            )
        stage = stage[np.argsort(redshifts[stage], kind="stable")]
        # Equal-redshift strata prevent one densely sampled redshift from
        # dominating a stage. Within each stratum, choose the neutral fraction
        # nearest the stage target on a previously unused cone.
        for stratum in np.array_split(stage, slices_per_stage):
            candidates = [
                int(candidate) for candidate in stratum
                if int(cone_ids[candidate]) not in used_cones
            ]
            if not candidates:
                candidates = [
                    int(candidate) for candidate in stage
                    if int(cone_ids[candidate]) not in used_cones
                ]
            if not candidates:
                raise ValueError(f"cannot select distinct cones for stage {name!r}")
            index = min(candidates, key=lambda value: abs(means[value] - target))
            selected.append(index)
            used_cones.add(int(cone_ids[index]))
    return selected


def interpolate_plane(
    cube: np.ndarray,
    target_z: np.ndarray,
    redshift: float,
) -> np.ndarray:
    if redshift < target_z[0] or redshift > target_z[-1]:
        raise ValueError(
            f"slice redshift {redshift} is outside cube grid "
            f"[{target_z[0]}, {target_z[-1]}]"
        )
    upper = int(np.searchsorted(target_z, redshift, side="left"))
    upper = min(upper, len(target_z) - 1)
    lower = max(upper - 1, 0)
    if lower == upper or target_z[upper] == target_z[lower]:
        return cube[..., upper]
    weight = float(
        (redshift - target_z[lower])
        / (target_z[upper] - target_z[lower])
    )
    return (1.0 - weight) * cube[..., lower] + weight * cube[..., upper]


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.square(prediction - truth).sum(dtype=np.float64))
    centered = truth - float(truth.mean(dtype=np.float64))
    total = float(np.square(centered).sum(dtype=np.float64))
    return float("nan") if total <= 1e-12 else 1.0 - residual / total


def score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(
            np.square(prediction - truth), dtype=np.float64
        ))),
        "r2": r2_score(truth, prediction),
    }


def load_3d_metadata(checkpoint: Path) -> dict:
    metadata = load_run_metadata(checkpoint.parent)
    if metadata is None:
        raise FileNotFoundError(f"run_metadata.json not found beside {checkpoint}")
    return metadata


@torch.inference_mode()
def predict_2d_slices(
    run,
    inputs: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("2-D batch size must be positive")
    model = build_2d_model(run.metadata["model_config"])
    strict_load(model, run.checkpoint)
    model = model.to(device).eval()
    predictions = []
    for start in range(0, len(inputs), batch_size):
        predictions.append(
            model(inputs[start:start + batch_size].to(device))[:, 0]
            .float().cpu().numpy()
        )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(predictions)


@torch.inference_mode()
def predict_3d_slices(
    checkpoint: Path,
    metadata: dict,
    cube_cache: LightconeCubeCache,
    selected: list[dict],
    device: str,
    patch_chunk_size: int,
) -> np.ndarray:
    config = ModelConfig.from_dict(metadata["model_config"])
    if config.kind == "localfno":
        config = replace(
            config, localfno_patch_chunk_size=int(patch_chunk_size)
        )
    model = TrainerModel(build_3d_model(config, cube_cache.in_channels))
    strict_load(model, checkpoint)
    model = model.to(device).eval()
    row_by_cone = {
        int(cone_id): row
        for row, cone_id in enumerate(cube_cache.cone_ids)
    }
    predictions = []
    for item in selected:
        row = row_by_cone[item["cone_id"]]
        sample = cube_cache[row]
        cube_prediction = model(sample["x"][None].to(device))[0, 0]
        cube_prediction = cube_prediction.float().cpu().numpy()
        predictions.append(interpolate_plane(
            cube_prediction, cube_cache.target_z, item["z"]
        ))
        del cube_prediction
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.stack(predictions)


def build_metric_rows(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: list[dict],
) -> list[dict]:
    rows = []
    for index, item in enumerate(selected):
        row = {
            "stage": item["stage"],
            "cone_id": item["cone_id"],
            "z": item["z"],
            "truth_mean_xhi": item["xhi_mean"],
        }
        for kind in MODEL_ORDER:
            metrics = score(truth[index], predictions[kind][index])
            row[f"{kind}_rmse"] = metrics["rmse"]
            row[f"{kind}_r2"] = metrics["r2"]
        rows.append(row)
    overall = {
        "stage": "overall",
        "cone_id": "",
        "z": "",
        "truth_mean_xhi": float(truth.mean(dtype=np.float64)),
    }
    for kind in MODEL_ORDER:
        metrics = score(truth, predictions[kind])
        overall[f"{kind}_rmse"] = metrics["rmse"]
        overall[f"{kind}_r2"] = metrics["r2"]
    rows.append(overall)
    return rows


def build_stage_metric_rows(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: list[dict],
) -> list[dict]:
    rows = []
    for stage, _target, _lower, _upper in STAGES:
        indices = [
            index for index, item in enumerate(selected)
            if item["stage"] == stage
        ]
        row = {
            "stage": stage,
            "n_slices": len(indices),
            "truth_mean_xhi": float(truth[indices].mean(dtype=np.float64)),
        }
        for kind in MODEL_ORDER:
            metrics = score(truth[indices], predictions[kind][indices])
            row[f"{kind}_rmse"] = metrics["rmse"]
            row[f"{kind}_r2"] = metrics["r2"]
        rows.append(row)
    overall = {
        "stage": "overall",
        "n_slices": len(selected),
        "truth_mean_xhi": float(truth.mean(dtype=np.float64)),
    }
    for kind in MODEL_ORDER:
        metrics = score(truth, predictions[kind])
        overall[f"{kind}_rmse"] = metrics["rmse"]
        overall[f"{kind}_r2"] = metrics["r2"]
    rows.append(overall)
    return rows


def plot_predictions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: list[dict],
    output: Path,
) -> None:
    columns = ("truth", *MODEL_ORDER)
    figure, axes = plt.subplots(
        len(selected), len(columns), figsize=(16, 3.25 * len(selected)),
        constrained_layout=True, squeeze=False,
    )
    image = None
    for row, item in enumerate(selected):
        panels = [truth[row]] + [predictions[kind][row] for kind in MODEL_ORDER]
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                panel, origin="lower", cmap="viridis", vmin=0, vmax=1,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column > 0:
                metrics = score(truth[row], panel)
                axis.text(
                    0.02, 0.98,
                    f"RMSE={metrics['rmse']:.3f}\nR^2={metrics['r2']:.3f}",
                    transform=axis.transAxes, va="top", ha="left",
                    color="white", fontsize=8.5,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
                )
        axes[row, 0].set_ylabel(
            f"{item['stage']}\ncone {item['cone_id']}, z={item['z']:.2f}\n"
            f"mean $x_{{HI}}$={item['xhi_mean']:.2f}",
            fontsize=9,
        )
    axes[0, 0].set_title("Truth")
    for column, kind in enumerate(MODEL_ORDER, start=1):
        axes[0, column].set_title(MODEL_LABELS[kind])
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.72, label="$x_{HI}$")
    figure.suptitle(
        "Common training cones across reionization stages", fontsize=15
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_prediction_pages(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: list[dict],
    output_dir: Path,
    page_size: int,
) -> None:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected), page_size):
        end = min(start + page_size, len(selected))
        plot_predictions(
            truth[start:end],
            {kind: values[start:end] for kind, values in predictions.items()},
            selected[start:end],
            output_dir / f"slices_{start + 1:03d}_{end:03d}.png",
        )


def plot_metrics(rows: list[dict], output: Path) -> None:
    labels = [str(row["stage"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for offset, kind in enumerate(MODEL_ORDER):
        position = x + (offset - 1) * width
        axes[0].bar(
            position, [row[f"{kind}_rmse"] for row in rows], width,
            label=MODEL_LABELS[kind], color=MODEL_COLORS[kind],
        )
        axes[1].bar(
            position, [row[f"{kind}_r2"] for row in rows], width,
            label=MODEL_LABELS[kind], color=MODEL_COLORS[kind],
        )
    for axis, title in zip(axes, ("RMSE", "R^2")):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_metric_distributions(rows: list[dict], output: Path) -> None:
    slice_rows = [row for row in rows if row["stage"] != "overall"]
    stage_names = [stage[0] for stage in STAGES]
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("rmse", "r2"), ("RMSE", "R^2")):
        positions = []
        values = []
        colors = []
        tick_positions = []
        for stage_index, stage in enumerate(stage_names):
            center = stage_index * 4
            tick_positions.append(center)
            stage_rows = [row for row in slice_rows if row["stage"] == stage]
            for model_index, kind in enumerate(MODEL_ORDER):
                positions.append(center + model_index - 1)
                values.append([row[f"{kind}_{metric}"] for row in stage_rows])
                colors.append(MODEL_COLORS[kind])
        boxes = axis.boxplot(
            values, positions=positions, widths=0.75, patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        axis.set_xticks(tick_positions, stage_names, rotation=25, ha="right")
        axis.set_title(f"Per-slice {title} distributions")
        axis.grid(axis="y", alpha=0.25)
        if metric == "r2":
            axis.axhline(0, color="black", linewidth=0.8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[kind])
        for kind in MODEL_ORDER
    ]
    axes[0].legend(handles, [MODEL_LABELS[kind] for kind in MODEL_ORDER])
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_parity(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    output: Path,
) -> None:
    flat_truth = truth.ravel()
    rng = np.random.default_rng(42)
    count = min(300_000, len(flat_truth))
    indices = rng.choice(len(flat_truth), size=count, replace=False)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, kind in zip(axes, MODEL_ORDER):
        rendered = axis.hexbin(
            flat_truth[indices], predictions[kind].ravel()[indices],
            gridsize=75, bins="log", mincnt=1, cmap="magma",
        )
        axis.plot([0, 1], [0, 1], "w--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("truth $x_{HI}$")
        axis.set_ylabel("prediction $x_{HI}$")
        axis.set_title(MODEL_LABELS[kind])
        figure.colorbar(rendered, ax=axis, label="log count")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wno-run", type=Path, required=True)
    parser.add_argument("--ufno-checkpoint", type=Path, required=True)
    parser.add_argument("--localfno-checkpoint", type=Path, required=True)
    parser.add_argument("--cube-cache", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/xhi2d_vs_best_3d"),
    )
    parser.add_argument(
        "--cone-split", choices=("train", "test", "heldout"), default="train",
        help="which cone pool to draw the comparison slices from, using cones "
             "common to all three models' splits. 'train' (default) is "
             "in-sample for every model; 'test' and 'heldout' (val+test) "
             "measure generalization.",
    )
    parser.add_argument("--patch-chunk-size", type=int, default=32)
    parser.add_argument("--slices-per-stage", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--batch-size-2d", type=int, default=16)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()
    if args.slices_per_stage <= 0:
        parser.error("--slices-per-stage must be positive")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")

    wno = load_run(args.wno_run.name, args.wno_run)
    if wno is None or wno.kind != "localwno":
        raise SystemExit(f"completed LocalWNO run not found in {args.wno_run}")
    ufno_metadata = load_3d_metadata(args.ufno_checkpoint)
    localfno_metadata = load_3d_metadata(args.localfno_checkpoint)
    if ufno_metadata["model_config"]["kind"] != "ufno":
        raise ValueError("--ufno-checkpoint metadata is not UFNO")
    if localfno_metadata["model_config"]["kind"] != "localfno":
        raise ValueError("--localfno-checkpoint metadata is not LocalFNO")
    if (
        ufno_metadata["input_features"] != localfno_metadata["input_features"]
        or ufno_metadata["parameter_normalization"]
        != localfno_metadata["parameter_normalization"]
    ):
        raise ValueError("3-D checkpoints do not share the same input contract")

    # Cones common to all three models' chosen split. The models were trained
    # with independent splits, so only the intersection keeps every model on
    # equal footing (all in-sample, or all out-of-sample).
    def cone_pool(metadata) -> set[int]:
        split = metadata["split"]
        keys = (("train",) if args.cone_split == "train"
                else ("test",) if args.cone_split == "test"
                else ("val", "test"))
        pool: set[int] = set()
        for key in keys:
            pool |= set(map(int, split.get(f"{key}_cone_ids", [])))
        return pool

    common_train = (cone_pool(wno.metadata) & cone_pool(ufno_metadata)
                    & cone_pool(localfno_metadata))
    if not common_train:
        raise ValueError(
            f"the three models have no common '{args.cone_split}' cones"
        )
    print(f"[cone-split] {args.cone_split}: {len(common_train)} common cones")

    wno_normalization = ParameterNormalization.from_dict(
        wno.metadata["parameter_normalization"]
    )
    slice_cache = SliceCache(
        wno.metadata["dataset"]["cache_file"],
        input_features=wno.metadata["input_features"]["name"],
        parameter_normalization=wno_normalization,
    )
    indices = select_stage_indices(
        slice_cache.cone_id, slice_cache.z, slice_cache.xHI_mean, common_train,
        slices_per_stage=args.slices_per_stage,
    )
    selected = []
    for stage_index, (stage, target, lower, upper) in enumerate(STAGES):
        start = stage_index * args.slices_per_stage
        for stage_rank, index in enumerate(
            indices[start:start + args.slices_per_stage], start=1
        ):
            selected.append({
                "stage": stage,
                "stage_target": target,
                "stage_range": [lower, upper],
                "stage_rank": stage_rank,
                "slice_index": int(index),
                "cone_id": int(slice_cache.cone_id[index]),
                "z": float(slice_cache.z[index]),
                "xhi_mean": float(slice_cache.xHI_mean[index]),
            })
    samples = [slice_cache[index] for index in indices]
    inputs_2d = torch.stack([sample["x"] for sample in samples])
    truth = torch.stack([sample["y"] for sample in samples]).numpy()[:, 0]

    normalization_3d = ParameterNormalization.from_dict(
        ufno_metadata["parameter_normalization"]
    )
    cube_cache = LightconeCubeCache(
        args.cube_cache,
        input_features=InputFeatures(ufno_metadata["input_features"]["name"]),
        parameter_normalization=normalization_3d,
    )
    absent = common_train - set(map(int, cube_cache.cone_ids))
    if absent:
        raise ValueError(
            f"{len(absent)} common {args.cone_split} cones are absent from cache"
        )

    predictions = {
        "localwno": predict_2d_slices(
            wno, inputs_2d, args.device, args.batch_size_2d
        ),
        "ufno": predict_3d_slices(
            args.ufno_checkpoint, ufno_metadata, cube_cache, selected,
            args.device, args.patch_chunk_size,
        ),
        "localfno": predict_3d_slices(
            args.localfno_checkpoint, localfno_metadata, cube_cache, selected,
            args.device, args.patch_chunk_size,
        ),
    }
    slice_rows = build_metric_rows(truth, predictions, selected)
    stage_rows = build_stage_metric_rows(truth, predictions, selected)

    args.output.mkdir(parents=True, exist_ok=True)
    summary_indices = [
        stage * args.slices_per_stage + args.slices_per_stage // 2
        for stage in range(len(STAGES))
    ]
    plot_predictions(
        truth[summary_indices],
        {kind: values[summary_indices] for kind, values in predictions.items()},
        [selected[index] for index in summary_indices],
        args.output / "prediction_comparison.png",
    )
    plot_prediction_pages(
        truth, predictions, selected, args.output / "prediction_pages",
        args.page_size,
    )
    plot_metrics(stage_rows, args.output / "rmse_r2_comparison.png")
    plot_metric_distributions(
        slice_rows, args.output / "rmse_r2_distributions.png"
    )
    plot_parity(truth, predictions, args.output / "parity_comparison.png")
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slice_rows[0]))
        writer.writeheader()
        writer.writerows(slice_rows)
    with (args.output / "stage_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    summary = {
        "selection": {
            "criterion": (
                "equal-redshift strata within fixed truth mean x_HI stages; "
                "nearest stage target in each stratum on distinct cones"
            ),
            "cone_split": args.cone_split,
            "common_cones": len(common_train),
            "slices_per_stage": args.slices_per_stage,
            "total_slices": len(selected),
            "slices": selected,
        },
        "models": {
            "localwno": str(wno.checkpoint.resolve()),
            "ufno": str(args.ufno_checkpoint.resolve()),
            "localfno": str(args.localfno_checkpoint.resolve()),
        },
        "alignment": "linear interpolation of 3-D predictions to native slice z",
        "stage_metrics": stage_rows,
        "slice_metrics": slice_rows,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Cross-model comparison: {args.output}")


if __name__ == "__main__":
    main()
