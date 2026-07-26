#!/usr/bin/env python3
"""Compare 2-D x_HI LocalWNO, LocalFNO, and UFNO runs.

The best run of each architecture is selected by validation RMSE. Predictions
then use identical held-out cache rows near representative redshift quantiles.
Model construction is driven entirely by each run's metadata so architecture
sweeps do not require matching environment variables at visualization time.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from modeling import LOCAL_GLOBAL_KINDS, TrainerModel, load_checkpoint
from models_zre_2d import LocalFNO2d, UFNO2d


ARCHITECTURE_ORDER = ("localwno", "localfno", "ufno", "localop")
ARCHITECTURE_NAMES = {
    "localwno": "LocalWNO",
    "localfno": "LocalFNO",
    "ufno": "UFNO",
    # Freely-paired local/global operator slots (operators.py); covers e.g.
    # the Walsh-Hadamard sweep, which all share kind="localop" and are
    # distinguished only by local_operator/global_operator in run_metadata.
    "localop": "LocalOp",
}
ARCHITECTURE_COLORS = {
    "localop": "#e07a5f",
    "localwno": "#168aad",
    "localfno": "#f4a261",
    "ufno": "#7b2cbf",
}
REPORT_FIELDS = (
    "val_rmse",
    "test_rmse",
    "test_gradient_rmse",
    "test_mean_xhi_mae",
    "test_high_k_power_ratio",
    "test_high_k_cross_correlation",
)


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    metadata: dict
    report: dict

    @property
    def kind(self) -> str:
        return str(self.metadata["model_config"]["kind"])

    @property
    def checkpoint(self) -> Path:
        return self.path / "final_model_state_dict.pt"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use LABEL=CHECKPOINT_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("runs must use LABEL=CHECKPOINT_DIR")
    return label, Path(raw_path)


def load_run(label: str, path: Path) -> Run | None:
    required = (
        path / "run_metadata.json",
        path / "final_report.json",
        path / "final_model_state_dict.pt",
    )
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        print(f"[skip] {label}: missing {', '.join(missing)} in {path}")
        return None
    metadata_mtime = required[0].stat().st_mtime
    stale = [
        item.name for item in required[1:]
        if item.stat().st_mtime < metadata_mtime
    ]
    if stale:
        print(
            f"[skip] {label}: {', '.join(stale)} predates run metadata "
            "and may be stale"
        )
        return None
    metadata = json.loads(required[0].read_text())
    report = json.loads(required[1].read_text())
    kind = metadata.get("model_config", {}).get("kind")
    if kind not in ARCHITECTURE_ORDER:
        print(f"[skip] {label}: unsupported model kind {kind!r}")
        return None
    absent_metrics = [name for name in REPORT_FIELDS if name not in report]
    if absent_metrics:
        print(f"[skip] {label}: report lacks {', '.join(absent_metrics)}")
        return None
    return Run(label, path, metadata, report)


def build_model(config: dict) -> TrainerModel:
    kind = str(config["kind"])
    contrast_mode = str(config.get("contrast_mode", "off"))
    in_channels = int(config["in_channels"])
    out_channels = int(config.get("out_channels", 1))
    if kind == "ufno":
        modes = tuple(int(value) for value in config["n_modes"])
        inner = UFNO2d(
            modes1=modes[0],
            modes2=modes[1],
            width=int(config["ufno_width"]),
            in_channels=in_channels,
            out_channels=out_channels,
            sigmoid=True,
            norm=str(config["ufno_norm"]),
        )
    elif kind in LOCAL_GLOBAL_KINDS or kind == "localop":
        # Runs predating the operator registry record only their kind, so fall
        # back to the pair that kind is shorthand for.
        local, global_ = LOCAL_GLOBAL_KINDS.get(kind, ("fourier", "fourier"))
        inner = LocalFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            base_width=int(config["localfno_base_width"]),
            local_window=tuple(config["localfno_window"]),
            local_modes=tuple(config.get("localfno_modes", (6, 6))),
            global_modes=tuple(config["localfno_global_modes"]),
            spectral_rank=int(config["localfno_spectral_rank"]),
            patch_chunk_size=int(config["localfno_patch_chunk_size"]),
            output_sigmoid=True,
            local_operator=config.get("local_operator", local),
            global_operator=config.get("global_operator", global_),
            local_operator_kwargs=config.get("local_operator_kwargs"),
            global_operator_kwargs=config.get("global_operator_kwargs"),
            local_windowed=config.get("local_windowed"),
            wavelet_levels=int(config.get("localwno_levels", 2)),
        )
    else:
        raise ValueError(f"unsupported model kind {kind!r}")
    if contrast_mode != "off":
        from contrast import ContrastComposed
        inner = ContrastComposed(inner, contrast_mode)
    return TrainerModel(inner)


def best_by_architecture(runs: list[Run]) -> dict[str, Run]:
    selected = {}
    for kind in ARCHITECTURE_ORDER:
        candidates = [run for run in runs if run.kind == kind]
        if not candidates:
            raise ValueError(f"no completed {ARCHITECTURE_NAMES[kind]} run found")
        selected[kind] = min(
            candidates, key=lambda run: float(run.report["val_rmse"])
        )
    return selected


def validate_data_contract(runs: list[Run]) -> None:
    reference = runs[0].metadata
    contract_fields = (
        "dataset", "input_features", "parameter_normalization", "split"
    )
    expected = {
        field: reference.get(field) for field in contract_fields
    }
    for run in runs[1:]:
        observed = {
            field: run.metadata.get(field) for field in contract_fields
        }
        if observed != expected:
            raise ValueError(
                f"{run.label} does not use the same dataset, input features, "
                "parameter normalization, and split"
            )


def representative_indices(
    cache: SliceCache,
    test_indices: np.ndarray,
    quantiles: list[float],
) -> list[int]:
    z = cache.z[test_indices]
    means = cache.xHI_mean[test_indices]
    targets = np.quantile(z, quantiles)
    chosen: list[int] = []
    used: set[int] = set()
    local_count = max(32, len(test_indices) // 100)
    for target in targets:
        nearest = np.argsort(np.abs(z - target), kind="stable")[:local_count]
        local_median = float(np.median(means[nearest]))
        ranked = nearest[np.argsort(
            np.abs(means[nearest] - local_median), kind="stable"
        )]
        selected = next(
            int(test_indices[index])
            for index in ranked
            if int(test_indices[index]) not in used
        )
        chosen.append(selected)
        used.add(selected)
    return chosen


@torch.inference_mode()
def predict(
    run: Run,
    inputs: torch.Tensor,
    device: str,
) -> np.ndarray:
    model = build_model(run.metadata["model_config"])
    result = load_checkpoint(model, run.checkpoint)
    if result.missing or result.unexpected or result.matched != result.total:
        raise RuntimeError(
            f"incomplete checkpoint load for {run.label}: "
            f"matched={result.matched}/{result.total}, "
            f"missing={result.missing}, unexpected={result.unexpected}"
        )
    model = model.to(device).eval()
    prediction = model(inputs.to(device)).cpu().numpy()[:, 0]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return prediction


def metrics_rows(runs: list[Run]) -> list[dict]:
    rows = []
    for run in runs:
        config = run.metadata["model_config"]
        training = run.metadata.get("training", {})
        row = {
            "label": run.label,
            "kind": run.kind,
            "checkpoint_dir": str(run.path.resolve()),
            "epochs": training.get("epochs", run.report.get("n_epochs")),
            "learning_rate": training.get("learning_rate"),
            "base_width": config.get("localfno_base_width", config.get("ufno_width")),
            "spectral_rank": config.get("localfno_spectral_rank"),
            "wavelet_levels": config.get("localwno_levels"),
        }
        row.update({name: float(run.report[name]) for name in REPORT_FIELDS})
        rows.append(row)
    return rows


def write_metrics(rows: list[dict], selected: dict[str, Run], output: Path) -> None:
    fields = list(rows[0])
    with (output / "metrics_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "selection_metric": "val_rmse",
        "selected_runs": {
            kind: run.label for kind, run in selected.items()
        },
        "runs": rows,
    }
    (output / "metrics_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def plot_metrics(rows: list[dict], output: Path) -> None:
    panels = (
        ("val_rmse", "Validation RMSE", False),
        ("test_rmse", "Test RMSE", False),
        ("test_gradient_rmse", "Test gradient RMSE", False),
        ("test_mean_xhi_mae", "Mean $x_{HI}$ MAE", False),
        ("test_high_k_power_ratio", "High-k power ratio", True),
        ("test_high_k_cross_correlation", "High-k correlation", False),
    )
    labels = [row["label"] for row in rows]
    colors = [ARCHITECTURE_COLORS[row["kind"]] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for axis, (field, title, ideal_one) in zip(axes.flat, panels):
        values = [row[field] for row in rows]
        axis.bar(x, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if ideal_one:
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    fig.savefig(output / "metrics_comparison.png", dpi=160)
    plt.close(fig)


def plot_predictions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: dict[str, Run],
    slice_info: list[dict],
    output: Path,
) -> None:
    kinds = list(ARCHITECTURE_ORDER)
    columns = ["Truth"] + [
        f"{ARCHITECTURE_NAMES[kind]}\n{selected[kind].label}"
        for kind in kinds
    ]
    fig, axes = plt.subplots(
        len(slice_info), len(columns), figsize=(14, 3.25 * len(slice_info)),
        constrained_layout=True, squeeze=False,
    )
    rendered = None
    for row, info in enumerate(slice_info):
        images = [truth[row]] + [predictions[kind][row] for kind in kinds]
        for col, image in enumerate(images):
            axis = axes[row, col]
            rendered = axis.imshow(
                image, origin="lower", cmap="viridis", vmin=0, vmax=1
            )
            if row == 0:
                axis.set_title(columns[col], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(
            f"z={info['z']:.2f}\nmean $x_{{HI}}$={info['xhi_mean']:.2f}",
            fontsize=10,
        )
    assert rendered is not None
    fig.colorbar(rendered, ax=axes, shrink=0.75, label="$x_{HI}$")
    fig.suptitle("Held-out representative redshift slices", fontsize=14)
    fig.savefig(output / "representative_z_predictions.png", dpi=170)
    plt.close(fig)

    errors = {
        kind: predictions[kind] - truth for kind in kinds
    }
    limit = max(
        0.05,
        float(np.quantile(np.abs(np.concatenate(
            [errors[kind].ravel() for kind in kinds]
        )), 0.99)),
    )
    fig, axes = plt.subplots(
        len(slice_info), len(kinds), figsize=(11, 3.25 * len(slice_info)),
        constrained_layout=True, squeeze=False,
    )
    rendered = None
    for row, info in enumerate(slice_info):
        for col, kind in enumerate(kinds):
            axis = axes[row, col]
            rendered = axis.imshow(
                errors[kind][row], origin="lower", cmap="RdBu_r",
                vmin=-limit, vmax=limit,
            )
            if row == 0:
                axis.set_title(ARCHITECTURE_NAMES[kind])
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(f"z={info['z']:.2f}")
    assert rendered is not None
    fig.colorbar(rendered, ax=axes, shrink=0.75, label="prediction - truth")
    fig.suptitle("Signed errors on representative redshift slices", fontsize=14)
    fig.savefig(output / "representative_z_errors.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", type=parse_run, required=True,
        help="candidate run as LABEL=CHECKPOINT_DIR; repeat for every run",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/xhi2d_wno_sweep_comparison"),
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()
    if not args.quantiles or any(not 0 <= value <= 1 for value in args.quantiles):
        parser.error("--quantiles must contain values in [0, 1]")

    runs = [
        run
        for label, path in args.run
        if (run := load_run(label, path)) is not None
    ]
    if not runs:
        raise SystemExit("no completed runs were found")
    selected = best_by_architecture(runs)
    validate_data_contract(runs)

    reference = selected["localwno"].metadata
    normalization = reference.get("parameter_normalization")
    cache = SliceCache(
        reference["dataset"]["cache_file"],
        input_features=reference["input_features"]["name"],
        parameter_normalization=(
            ParameterNormalization.from_dict(normalization)
            if normalization is not None else None
        ),
    )
    test_cones = np.asarray(reference["split"]["test_cone_ids"], dtype=np.int64)
    test_indices = np.flatnonzero(np.isin(cache.cone_id, test_cones))
    indices = representative_indices(cache, test_indices, args.quantiles)
    samples = [cache[index] for index in indices]
    inputs = torch.stack([sample["x"] for sample in samples])
    truth = torch.stack([sample["y"] for sample in samples]).numpy()[:, 0]
    slice_info = [
        {
            "global_index": int(index),
            "cone_id": int(cache.cone_id[index]),
            "z": float(cache.z[index]),
            "xhi_mean": float(cache.xHI_mean[index]),
        }
        for index in indices
    ]

    predictions = {}
    for kind in ARCHITECTURE_ORDER:
        run = selected[kind]
        print(
            f"[predict] {ARCHITECTURE_NAMES[kind]}: {run.label} "
            f"(val_rmse={float(run.report['val_rmse']):.6f})"
        )
        predictions[kind] = predict(run, inputs, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    rows = metrics_rows(runs)
    write_metrics(rows, selected, args.output)
    plot_metrics(rows, args.output)
    plot_predictions(truth, predictions, selected, slice_info, args.output)
    (args.output / "selected_slices.json").write_text(
        json.dumps(slice_info, indent=2) + "\n"
    )
    print(f"[done] comparison written to {args.output}")


if __name__ == "__main__":
    main()
