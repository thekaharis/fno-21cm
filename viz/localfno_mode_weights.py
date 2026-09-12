#!/usr/bin/env python3
"""Per-branch Fourier mode-weight diagnostic for trained LocalFNO checkpoints.

Counterpart of ``viz/visualize_spectral_weights.py`` (the SIREN/FNO/U-FNO
history diagnostic) that works directly from a checkpoint instead of the
training-time ``spectral_weight_history.npz``: it loads a trained LocalFNO
or LocalSirenFNO (3-D lightcone or 2-D z_re variant), walks all six
spectral branches (``encoder0``, ``encoder1``, ``bottleneck.0``,
``bottleneck.1``, ``decoder1``, ``decoder0``) and maps their retained
quadrant weights to

  * RMS-per-absolute-mode profiles along each axis and radial shells,
    in the same style as the history-based plots,
  * centered signed-``k_x``/``k_y`` mode-plane heatmaps per branch (the
    quadrant tensors reassembled into one plane, log10 RMS over channels),
  * outer-quarter / inner-quarter cutoff ratios per branch and axis,
  * a CSV with every profile value.

Local branches act on Hann windows (encoder0/decoder0 at full resolution,
encoder1/decoder1 at 1/2 resolution) while the bottleneck acts on the whole
1/4-resolution volume, so mode index ``k`` refers to each branch's own
transform length; panel titles carry the window/downsampling context.

Usage (from the project root, same env contract as training):

    python -m viz.localfno_mode_weights --task 3d
    python -m viz.localfno_mode_weights --task zre

CHECKPOINT_DIR / CHECKPOINT / CHECKPOINT_KIND select the weights file as in
the other viz entry points.  The architecture geometry (mode counts, rank,
widths, channels) is inferred from the checkpoint's weight shapes, so no
LOCALFNO_* switches are required; only the Hann window extent -- which is
cosmetic here (panel labels) -- is read from run_metadata.json when the run
recorded it, else from LOCALFNO_WINDOW_*.  LocalSirenFNO checkpoints encode
no mode counts in their shapes (the weights are SIREN-generated), so their
retained bands come from run_metadata.json when recorded, else from
LOCALFNO_MODES_* / LOCALFNO_GLOBAL_MODES_* (z_re) or N_MODES_* (3-D).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from util.run_metadata import load_run_metadata, resolve_checkpoint
from util.spectral_weights import (
    _channel_power,
    _profiles_from_power_grids,
)
from viz.visualize_spectral_weights import high_low_ratio


DEFAULT_CHECKPOINT_DIRS = {
    "3d": Path("checkpoints") / "checkpoints_3d_localfno",
    "zre": Path("checkpoints") / "checkpoints_zre_localfno",
}
# Resolution of each branch relative to the input grid (the U-Net has two
# pooling levels); used only for labeling.
BRANCH_DOWNSAMPLE = {
    "encoder0": 1,
    "encoder1": 2,
    "bottleneck.0": 4,
    "bottleneck.1": 4,
    "decoder1": 2,
    "decoder0": 1,
}


@dataclass(frozen=True)
class BranchWeights:
    """Quadrant powers and axis profiles of one spectral branch."""

    name: str
    label: str
    n_modes: tuple[int, ...]
    windowed: bool
    quadrant_powers: list[np.ndarray]
    profiles: dict[str, np.ndarray]

    @property
    def ndim(self) -> int:
        return len(self.n_modes)


def _quadrant_tensors(conv: nn.Module) -> list[torch.Tensor] | None:
    # LocalSirenFNO branches generate their quadrant weights from SIREN
    # networks; materialize them (extract_branches runs under no_grad).
    materialize = getattr(conv, "spectral_weight_tensors", None)
    if callable(materialize):
        return materialize()
    if all(hasattr(conv, name) for name in ("weights1", "weights2",
                                            "weights3", "weights4")):
        return [conv.weights1, conv.weights2, conv.weights3, conv.weights4]
    if all(hasattr(conv, name) for name in ("weights1", "weights2")):
        return [conv.weights1, conv.weights2]
    return None


def _channel_power_any(weight: torch.Tensor) -> np.ndarray:
    """``util.spectral_weights._channel_power`` extended to 2-D weights."""
    if weight.ndim == 4:
        return _channel_power(weight.unsqueeze(-1))[..., 0]
    return _channel_power(weight)


def _branch_profiles(
    powers: list[np.ndarray],
    n_modes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Map quadrant powers to RMS profiles by absolute mode index.

    Follows ``util.spectral_weights._ufno_quadrant_profiles``: positive
    slices ``:m`` are frequencies ``0..m-1``; negative slices ``-m:`` are
    ``-m..-1``, i.e. absolute order ``m..1``. 2-D branches reuse the 3-D
    grouping through a singleton z axis (rFFT y takes the z role of "last,
    non-negative axis"; the ``z`` profile is dropped for them).
    """
    if len(n_modes) == 3:
        mx, my, mz = n_modes
        positive_x, negative_x = np.arange(mx), np.arange(mx, 0, -1)
        positive_y, negative_y = np.arange(my), np.arange(my, 0, -1)
        kz = np.arange(mz)
        coordinates = (
            (positive_x, positive_y, kz),
            (negative_x, positive_y, kz),
            (positive_x, negative_y, kz),
            (negative_x, negative_y, kz),
        )
        entries = [
            (power, kx, ky, z)
            for power, (kx, ky, z) in zip(powers, coordinates, strict=True)
        ]
        axes = ("x", "y", "z", "shell")
    else:
        mx, my = n_modes
        positive_x, negative_x = np.arange(mx), np.arange(mx, 0, -1)
        ky = np.arange(my)
        singleton = np.zeros(1)
        entries = [
            (powers[0][:, :, None], positive_x, ky, singleton),
            (powers[1][:, :, None], negative_x, ky, singleton),
        ]
        axes = ("x", "y", "shell")
    x, y, z, shell = _profiles_from_power_grids(entries)
    values = {"x": x, "y": y, "z": z, "shell": shell}
    return {axis: values[axis] for axis in axes}


def assemble_plane(powers: list[np.ndarray]) -> np.ndarray:
    """Reassemble quadrant powers into one centered signed-frequency plane.

    3-D (four quadrants, shape ``(mx, my, mz)`` each) returns
    ``(2*mx, 2*my, mz)`` with axes ``k_x, k_y = -m .. m-1``.  2-D (two
    signed-x blocks, shape ``(mx, my)``) returns ``(2*mx, my)`` with
    ``k_x = -mx .. mx-1`` and non-negative rFFT ``k_y``.
    """
    if len(powers) == 4:
        mx, my = powers[0].shape[:2]
        plane = np.full((2 * mx, 2 * my, *powers[0].shape[2:]), np.nan)
        plane[mx:, my:] = powers[0]        # (+x, +y)
        plane[:mx, my:] = powers[1]        # (-x, +y)
        plane[mx:, :my] = powers[2]        # (+x, -y)
        plane[:mx, :my] = powers[3]        # (-x, -y)
        return plane
    mx = powers[0].shape[0]
    plane = np.full((2 * mx, *powers[0].shape[1:]), np.nan)
    plane[mx:] = powers[0]                 # (+x)
    plane[:mx] = powers[1]                 # (-x)
    return plane


@torch.no_grad()
def extract_branches(model: nn.Module) -> list[BranchWeights]:
    """Collect quadrant powers and profiles from every LocalFNO branch."""
    branches: list[BranchWeights] = []
    for name, module in model.named_modules():
        conv = getattr(module, "spectral", None)
        if conv is None or not hasattr(module, "window_grid"):
            continue
        tensors = _quadrant_tensors(conv)
        if tensors is None:
            continue
        short = name.split("fno.", 1)[-1]
        n_modes = tuple(int(mode) for mode in conv.n_modes)
        windowed = module.window_grid is not None
        if windowed:
            window = "x".join(
                str(size) for size in module.window_grid.window_size
            )
            context = f"window {window}"
        else:
            context = "global"
        factor = BRANCH_DOWNSAMPLE.get(short)
        resolution = f" @1/{factor}res" if factor and factor > 1 else ""
        modes = "x".join(str(mode) for mode in n_modes)
        powers = [_channel_power_any(weight) for weight in tensors]
        branches.append(
            BranchWeights(
                name=short,
                label=f"{short} ({context}{resolution}, modes {modes})",
                n_modes=n_modes,
                windowed=windowed,
                quadrant_powers=powers,
                profiles=_branch_profiles(powers, n_modes),
            )
        )
    if not branches:
        raise ValueError(
            "No LocalFNO spectral branches (quadrant weight tensors) found; "
            "is the checkpoint really a LocalFNO run?"
        )
    return branches


AXIS_TITLES = {
    "x": "|k_x|",
    "y": "|k_y|",
    "z": "k_z (rFFT)",
    "shell": "radial shell |k|",
}


def _log_limits(values: list[np.ndarray]) -> tuple[float, float]:
    positive = np.concatenate(
        [value[np.isfinite(value) & (value > 0)].reshape(-1)
         for value in values]
    )
    if not positive.size:
        return -12.0, 0.0
    logs = np.log10(positive)
    return float(np.percentile(logs, 1)), float(np.percentile(logs, 99))


def plot_profiles(branches: list[BranchWeights], output: Path) -> None:
    axes_names = list(branches[0].profiles)
    figure, axes = plt.subplots(
        len(branches),
        len(axes_names),
        figsize=(4.2 * len(axes_names), 2.7 * len(branches)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, branch in enumerate(branches):
        for column, axis_name in enumerate(axes_names):
            axis = axes[row, column]
            values = branch.profiles[axis_name]
            axis.plot(
                np.arange(values.size),
                values,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                color="tab:blue",
            )
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(AXIS_TITLES[axis_name])
            if row == len(branches) - 1:
                axis.set_xlabel("absolute mode index")
            if column == 0:
                axis.set_ylabel(f"{branch.label}\nRMS |weight|", fontsize=8)
            else:
                axis.set_ylabel("RMS |weight|")
    figure.suptitle("LocalFNO mode-weight profiles by branch", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_planes(branches: list[BranchWeights], output: Path) -> None:
    three_dimensional = branches[0].ndim == 3
    columns = 2 if three_dimensional else 1
    figure, axes = plt.subplots(
        len(branches),
        columns,
        figsize=(5.0 * columns + 1.5, 3.4 * len(branches)),
        squeeze=False,
        constrained_layout=True,
    )
    planes: list[tuple[int, int, np.ndarray, str]] = []
    for row, branch in enumerate(branches):
        plane = assemble_plane(branch.quadrant_powers)
        rms = np.sqrt(plane)
        if three_dimensional:
            planes.append((row, 0, rms[:, :, 0], "k_z = 0"))
            planes.append(
                (row, 1, np.sqrt(plane.mean(axis=-1)), "RMS over k_z")
            )
        else:
            planes.append((row, 0, rms, "rFFT k_y"))
    vmin, vmax = _log_limits([values for _, _, values, _ in planes])
    image = None
    for row, column, values, subtitle in planes:
        axis = axes[row, column]
        branch = branches[row]
        mx = branch.n_modes[0]
        my = branch.n_modes[1]
        extent = (
            (-my - 0.5, my - 0.5) if branch.ndim == 3 else (-0.5, my - 0.5)
        ) + (-mx - 0.5, mx - 0.5)
        image = axis.imshow(
            np.log10(np.maximum(values, 1e-12)),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            extent=extent,
        )
        axis.set_title(f"{branch.label}\n{subtitle}", fontsize=9)
        axis.set_xlabel("k_y")
        axis.set_ylabel("k_x (signed)")
        axis.axhline(-0.5, color="white", linewidth=0.5, alpha=0.5)
        if branch.ndim == 3:
            axis.axvline(-0.5, color="white", linewidth=0.5, alpha=0.5)
    assert image is not None
    figure.colorbar(
        image,
        ax=axes,
        label="log10 RMS |Fourier weight|",
        shrink=0.75,
    )
    figure.suptitle(
        "LocalFNO mode-plane weight maps (quadrants reassembled)",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_cutoff_ratios(branches: list[BranchWeights], output: Path) -> None:
    axes_names = list(branches[0].profiles)
    positions = np.arange(len(branches))
    width = 0.8 / len(axes_names)
    figure, axis = plt.subplots(
        figsize=(1.9 * len(branches) + 3.0, 4.6),
        constrained_layout=True,
    )
    for index, axis_name in enumerate(axes_names):
        ratios = [
            float(high_low_ratio(branch.profiles[axis_name]))
            for branch in branches
        ]
        axis.bar(
            positions + (index - (len(axes_names) - 1) / 2) * width,
            ratios,
            width=width,
            label=AXIS_TITLES[axis_name],
        )
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yscale("log")
    axis.set_xticks(positions, [branch.name for branch in branches])
    axis.set_ylabel("high-mode / low-mode RMS")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=8)
    axis.set_title(
        "Cutoff diagnostic: outer-quarter versus inner-quarter weights"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_csv(branches: list[BranchWeights], output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("branch", "axis", "mode", "rms_weight"))
        for branch in branches:
            for axis_name, values in branch.profiles.items():
                for mode, value in enumerate(values):
                    writer.writerow(
                        (branch.name, axis_name, mode, float(value))
                    )


def _shape_of(state: dict, suffix: str) -> tuple[int, ...]:
    for key, value in state.items():
        if key.endswith(suffix):
            return tuple(int(size) for size in value.shape)
    raise KeyError(
        f"No '*{suffix}' key in the checkpoint; is this a LocalFNO "
        "state dict?"
    )


def _window_from_env(task: str) -> tuple[int, ...]:
    window = (
        int(os.environ.get("LOCALFNO_WINDOW_X", "16")),
        int(os.environ.get("LOCALFNO_WINDOW_Y", "16")),
    )
    if task == "3d":
        window += (int(os.environ.get("LOCALFNO_WINDOW_Z", "32")),)
    return window


def _siren_modes(
    task: str,
    model_config: dict,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Retained local/global bands of a LocalSirenFNO checkpoint.

    SIREN-generated weights leave no mode-count trace in the checkpoint
    shapes, so the bands come from run_metadata.json (3-D runs record the
    full ModelConfig) or from the same LOCALFNO_*/N_MODES_* switches the
    training scripts read.
    """
    local = tuple(
        int(mode) for mode in model_config.get("localfno_modes") or ()
    )
    global_ = tuple(
        int(mode) for mode in model_config.get("modes") or ()
    )
    if task == "3d":
        local = local or (
            int(os.environ.get("LOCALFNO_MODES_X", "6")),
            int(os.environ.get("LOCALFNO_MODES_Y", "6")),
            int(os.environ.get("LOCALFNO_MODES_Z", "12")),
        )
        global_ = global_ or (
            int(os.environ.get("N_MODES_X", "16")),
            int(os.environ.get("N_MODES_Y", "16")),
            int(os.environ.get("N_MODES_Z", "16")),
        )
    else:
        local = local or (
            int(os.environ.get("LOCALFNO_MODES_X", "6")),
            int(os.environ.get("LOCALFNO_MODES_Y", "6")),
        )
        global_ = global_ or (
            int(os.environ.get("LOCALFNO_GLOBAL_MODES_X", "16")),
            int(os.environ.get("LOCALFNO_GLOBAL_MODES_Y", "16")),
        )
    return local, global_


def _siren_kwargs_from_shapes(state: dict) -> dict:
    """Structural SIREN settings encoded in the checkpoint weight shapes.

    Omega and the Fourier-feature sigma are learnable tensors, so their
    trained values load from the checkpoint; only the hidden/feature widths
    and the trunk depth must match the run's architecture.
    """
    first = _shape_of(
        state, "encoder0.spectral.real_weight.first.linear.weight"
    )
    hidden_layers = {
        key.rsplit("hidden.", 1)[-1].split(".", 1)[0]
        for key in state
        if "encoder0.spectral.real_weight.hidden." in key
    }
    return {
        "siren_hidden_dim": first[0],
        "siren_feature_dim": first[1],
        "siren_n_hidden": 1 + len(hidden_layers),
    }


def _build_model(task: str, checkpoint: Path) -> nn.Module:
    """Reconstruct the LocalFNO whose geometry the checkpoint encodes.

    Mode counts, spectral rank, base width, and channel counts are read
    from the weight shapes, so the LOCALFNO_* switches do not need to
    match the training run. LocalSirenFNO checkpoints are the exception:
    their SIREN-generated weights encode no mode count, so the retained
    bands come from run_metadata.json when recorded (3-D runs), otherwise
    from the LOCALFNO_MODES_*/N_MODES_* switches. The Hann window extent
    also leaves no trace in the parameters; it only affects panel labels
    here and comes from run_metadata.json or LOCALFNO_WINDOW_*.
    """
    from modeling import TrainerModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lifting = _shape_of(state, "lifting.weight")
    base_width, in_channels = lifting[0], lifting[1]
    out_channels = _shape_of(state, "projection.2.weight")[0]

    metadata = load_run_metadata(checkpoint.parent) or {}
    model_config = metadata.get("model_config") or {}
    window = tuple(
        int(size) for size in model_config.get("localfno_window") or ()
    ) or _window_from_env(task)

    fourier = any(
        key.endswith("encoder0.spectral.weights1") for key in state
    )
    siren = any(
        "encoder0.spectral.real_weight." in key for key in state
    )
    if not (fourier or siren):
        # Only Fourier branches have per-mode weights to profile. A wavelet,
        # Walsh-Hadamard, or convolutional local slot needs its own diagnostic.
        raise ValueError(
            "This checkpoint's local branches are not Fourier "
            f"(local operator: {model_config.get('local_operator', 'unknown')}"
            "). Mode-weight panels only apply to localfno/localsirenfno runs."
        )
    if siren:
        rank = _shape_of(state, "encoder0.in_projection.weight")[0]
        local_modes, global_modes = _siren_modes(task, model_config)
        siren_kwargs = _siren_kwargs_from_shapes(state)
    else:
        local = _shape_of(state, "encoder0.spectral.weights1")
        global_ = _shape_of(state, "bottleneck.0.spectral.weights1")
        expected = 5 if task == "3d" else 4
        if len(local) != expected:
            raise ValueError(
                f"--task {task} expects {expected}-dimensional quadrant "
                f"weights, but the checkpoint holds shape {local}; wrong "
                "--task for this checkpoint?"
            )
        rank, local_modes, global_modes = local[0], local[2:], global_[2:]
        siren_kwargs = {}

    try:
        if task == "3d":
            from local_fno_3d import LocalFNO3d as LocalFNO
        else:
            from models_zre_2d import LocalFNO2d as LocalFNO
        model = LocalFNO(
            in_channels=in_channels,
            out_channels=out_channels,
            base_width=base_width,
            local_window=window,
            local_modes=local_modes,
            global_modes=global_modes,
            spectral_rank=rank,
            siren=siren,
            **siren_kwargs,
        )
    except ValueError as error:
        source = (
            "run metadata or the environment" if siren else "the checkpoint"
        )
        raise ValueError(
            f"{error} -- the retained modes come from {source}, so this "
            "window cannot be the training run's; export the LOCALFNO_* "
            "values that run used"
        ) from error
    variant = "LocalSirenFNO" if siren else "LocalFNO"
    print(
        f"Model ({task}, {variant}): window={window} "
        f"local-modes={tuple(local_modes)} "
        f"global-modes={tuple(global_modes)} base-width={base_width} "
        f"rank={rank} in/out-channels={in_channels}/{out_channels}"
    )
    return TrainerModel(model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=("3d", "zre"),
        default="3d",
        help="which LocalFNO pipeline the checkpoint belongs to",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="weights file (default: CHECKPOINT / CHECKPOINT_KIND resolution "
        "inside CHECKPOINT_DIR, as in the other viz entry points)",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    checkpoint_dir = Path(
        os.environ.get(
            "CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIRS[args.task])
        )
    )
    checkpoint = args.checkpoint or resolve_checkpoint(checkpoint_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    from modeling import load_checkpoint

    model = _build_model(args.task, checkpoint)
    report = load_checkpoint(model, checkpoint)
    print(
        f"Loaded {checkpoint} ({report.transform}): "
        f"{report.matched}/{report.total} tensors matched"
    )
    if report.missing or report.unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly match the configured LocalFNO "
            f"(missing={list(report.missing)[:4]}, "
            f"unexpected={list(report.unexpected)[:4]}); align the "
            "LOCALFNO_* environment switches with the training run"
        )

    branches = extract_branches(model)
    print(f"Found {len(branches)} spectral branches:")
    for branch in branches:
        print(f"  {branch.label}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (
        Path("figures") / ("3d_xhi" if args.task == "3d" else "zre") / "diagnostics"
        / f"localfno-mode-weights-{args.task}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_profiles(branches, output_dir / "mode_weight_profiles.png")
    plot_planes(branches, output_dir / "mode_weight_planes.png")
    plot_cutoff_ratios(branches, output_dir / "mode_weight_cutoff_ratio.png")
    write_csv(branches, output_dir / "mode_weight_profiles.csv")
    print(f"Wrote LocalFNO mode-weight diagnostics to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
