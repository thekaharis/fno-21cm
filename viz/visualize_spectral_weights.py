#!/usr/bin/env python3
"""Plot Fourier-layer weight magnitudes throughout training."""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modeling import ModelConfig
from util.spectral_weights import HISTORY_FILENAME, HISTORY_FORMAT_VERSION


AXES = ("x", "y", "z", "shell")
AXIS_TITLES = {
    "x": "|k_x|",
    "y": "|k_y|",
    "z": "k_z (rFFT)",
    "shell": "radial shell |k|",
}


def load_history(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No spectral history at {path}. New training runs create this "
            "file automatically; old best/final checkpoints cannot recover "
            "the missing intermediate epochs."
        )
    with np.load(path, allow_pickle=False) as saved:
        required = {"epochs", "layers", *AXES}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Spectral history is missing keys: {sorted(missing)}")
        layers = saved["layers"].astype(str)
        version = (
            int(saved["format_version"])
            if "format_version" in saved.files
            else 1
        )
        if version != HISTORY_FORMAT_VERSION and any(
            layer.startswith("body.conv") for layer in layers
        ):
            raise ValueError(
                "This U-FNO history predates the corrected positive/negative "
                "quadrant mapping and cannot be repaired from its aggregated "
                "profiles. Regenerate it with a new training run."
            )
        return {key: saved[key] for key in required}


def _epoch_labels(epochs: np.ndarray) -> list[str]:
    return ["init" if int(epoch) < 0 else str(int(epoch)) for epoch in epochs]


def _tick_indices(length: int, maximum: int = 8) -> np.ndarray:
    count = min(length, maximum)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def _log_limits(
    history: dict[str, np.ndarray],
    axes: tuple[str, ...] = AXES,
) -> tuple[float, float]:
    positive = np.concatenate(
        [history[axis][history[axis] > 0] for axis in axes]
    )
    if not positive.size:
        return -12.0, 0.0
    log_values = np.log10(positive)
    return (
        float(np.percentile(log_values, 1)),
        float(np.percentile(log_values, 99)),
    )


def _z_only_layer_grid(layer_count: int) -> tuple[int, int]:
    """Return a compact layer grid, using 3 columns for six-layer U-FNO."""
    columns = min(3, max(1, layer_count))
    rows = math.ceil(layer_count / columns)
    return rows, columns


def plot_evolution(
    history: dict[str, np.ndarray],
    output: Path,
    axes_to_plot: tuple[str, ...] = AXES,
) -> None:
    epochs = history["epochs"]
    layers = history["layers"].astype(str)
    vmin, vmax = _log_limits(history, axes_to_plot)
    image = None
    epoch_ticks = _tick_indices(len(epochs))
    labels = _epoch_labels(epochs)

    if axes_to_plot == ("z",):
        rows, columns = _z_only_layer_grid(len(layers))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.2 * columns, 3.3 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for layer_index, layer in enumerate(layers):
            axis = axes.flat[layer_index]
            values = history["z"][:, layer_index, :]
            image = axis.imshow(
                np.log10(np.maximum(values, 1e-12)),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            axis.set_title(layer)
            axis.set_xlabel("LOS mode index k_z")
            axis.set_ylabel("epoch")
            axis.set_yticks(
                epoch_ticks,
                [labels[index] for index in epoch_ticks],
            )
        for index in range(len(layers), rows * columns):
            axes.flat[index].set_visible(False)
    else:
        figure, axes = plt.subplots(
            len(layers),
            len(axes_to_plot),
            figsize=(4.2 * len(axes_to_plot), 2.7 * len(layers)),
            squeeze=False,
            constrained_layout=True,
        )
        for layer_index, layer in enumerate(layers):
            for column, axis_name in enumerate(axes_to_plot):
                axis = axes[layer_index, column]
                values = history[axis_name][:, layer_index, :]
                image = axis.imshow(
                    np.log10(np.maximum(values, 1e-12)),
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
                axis.set_title(AXIS_TITLES[axis_name])
                axis.set_xlabel("absolute mode index")
                axis.set_yticks(
                    epoch_ticks,
                    [labels[index] for index in epoch_ticks],
                )
                if column == 0:
                    axis.set_ylabel(f"{layer}\nepoch")
                else:
                    axis.set_ylabel("epoch")

    assert image is not None
    figure.colorbar(
        image,
        ax=axes,
        label="log10 RMS |Fourier weight|",
        shrink=0.75,
    )
    title = (
        "LOS Fourier-weight evolution"
        if axes_to_plot == ("z",)
        else "Fourier-weight evolution by layer and mode"
    )
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _selected_epoch_indices(length: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(4, length), dtype=int))


def plot_profiles(
    history: dict[str, np.ndarray],
    output: Path,
    axes_to_plot: tuple[str, ...] = AXES,
) -> None:
    epochs = history["epochs"]
    layers = history["layers"].astype(str)
    selected = _selected_epoch_indices(len(epochs))
    labels = _epoch_labels(epochs)
    if axes_to_plot == ("z",):
        rows, columns = _z_only_layer_grid(len(layers))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.2 * columns, 3.6 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for layer_index, layer in enumerate(layers):
            axis = axes.flat[layer_index]
            for epoch_index in selected:
                values = history["z"][epoch_index, layer_index]
                axis.plot(
                    np.arange(values.size),
                    values,
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    label=labels[epoch_index],
                )
            axis.set_yscale("log")
            axis.set_title(layer)
            axis.set_xlabel("LOS mode index k_z")
            axis.set_ylabel("RMS |weight|")
            axis.grid(alpha=0.25)
        for index in range(len(layers), rows * columns):
            axes.flat[index].set_visible(False)
        axes.flat[0].legend(title="epoch", fontsize=8)
    else:
        figure, axes = plt.subplots(
            len(layers),
            len(axes_to_plot),
            figsize=(4.2 * len(axes_to_plot), 2.7 * len(layers)),
            squeeze=False,
            constrained_layout=True,
        )
        for layer_index, layer in enumerate(layers):
            for column, axis_name in enumerate(axes_to_plot):
                axis = axes[layer_index, column]
                for epoch_index in selected:
                    values = history[axis_name][epoch_index, layer_index]
                    axis.plot(
                        np.arange(values.size),
                        values,
                        marker="o",
                        markersize=2.5,
                        linewidth=1.2,
                        label=labels[epoch_index],
                    )
                axis.set_yscale("log")
                axis.set_title(AXIS_TITLES[axis_name])
                axis.set_xlabel("absolute mode index")
                axis.grid(alpha=0.25)
                if column == 0:
                    axis.set_ylabel(f"{layer}\nRMS |weight|")
                else:
                    axis.set_ylabel("RMS |weight|")
                if layer_index == 0 and column == len(axes_to_plot) - 1:
                    axis.legend(title="epoch", fontsize=8)

    title = (
        "LOS Fourier-weight profiles"
        if axes_to_plot == ("z",)
        else "Fourier-weight profiles at selected epochs"
    )
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def high_low_ratio(values: np.ndarray) -> np.ndarray:
    """RMS(last quarter of modes) / RMS(first quarter of modes)."""
    values = np.asarray(values)
    result = np.full(values.shape[:-1], np.nan, dtype=np.float32)
    for index in np.ndindex(result.shape):
        profile = values[index]
        profile = profile[np.isfinite(profile)]
        if profile.size == 0:
            continue
        band = max(1, profile.size // 4)
        low = np.sqrt(np.mean(np.square(profile[:band])))
        high = np.sqrt(np.mean(np.square(profile[-band:])))
        result[index] = high / max(low, 1e-12)
    return result


def plot_cutoff_ratios(
    history: dict[str, np.ndarray],
    output: Path,
    axes_to_plot: tuple[str, ...] = AXES,
) -> None:
    epochs = history["epochs"]
    layers = history["layers"].astype(str)
    labels = _epoch_labels(epochs)
    x = np.arange(len(epochs))
    figure, axes = plt.subplots(
        1,
        len(axes_to_plot),
        figsize=(
            (8.0, 4.8)
            if axes_to_plot == ("z",)
            else (4.5 * len(axes_to_plot), 4.2)
        ),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, axis_name in zip(axes[0], axes_to_plot):
        ratios = high_low_ratio(history[axis_name])
        for layer_index, layer in enumerate(layers):
            axis.plot(x, ratios[:, layer_index], marker="o", label=layer)
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(AXIS_TITLES[axis_name])
        axis.set_xlabel("epoch")
        axis.set_ylabel("high-mode / low-mode RMS")
        ticks = _tick_indices(len(epochs))
        axis.set_xticks(ticks, [labels[index] for index in ticks])
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8, title="layer")
    title = (
        "LOS cutoff diagnostic"
        if axes_to_plot == ("z",)
        else "Cutoff diagnostic: outer-quarter versus inner-quarter weights"
    )
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_csv(
    history: dict[str, np.ndarray],
    output: Path,
    axes_to_write: tuple[str, ...] = AXES,
) -> None:
    epochs = history["epochs"]
    layers = history["layers"].astype(str)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "layer", "axis", "mode", "rms_weight"))
        for epoch_index, epoch in enumerate(epochs):
            for layer_index, layer in enumerate(layers):
                for axis_name in axes_to_write:
                    for mode, value in enumerate(
                        history[axis_name][epoch_index, layer_index]
                    ):
                        writer.writerow(
                            (int(epoch), layer, axis_name, mode, float(value))
                        )


def parse_args() -> argparse.Namespace:
    default_checkpoint_dir = str(
        ModelConfig.from_env().default_checkpoint_dir
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            os.environ.get("CHECKPOINT_DIR", default_checkpoint_dir)
        )
        / HISTORY_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (
        Path("figures") / f"spectral-weights_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_evolution(history, output_dir / "spectral_weight_evolution.png")
    plot_profiles(history, output_dir / "spectral_weight_profiles.png")
    plot_cutoff_ratios(history, output_dir / "spectral_weight_cutoff_ratio.png")
    write_csv(history, output_dir / "spectral_weight_history.csv")
    print(f"Wrote spectral-weight diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
