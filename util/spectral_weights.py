"""Extract and persist human-readable Fourier-weight summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HISTORY_FILENAME = "spectral_weight_history.npz"
HISTORY_FORMAT_VERSION = 2


@dataclass(frozen=True)
class SpectralWeightProfile:
    layer: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    shell: np.ndarray


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "fno"):
        model = model.fno
    return model


def _channel_power(weight: torch.Tensor) -> np.ndarray:
    """Mean squared complex magnitude over input/output channels."""
    if weight.ndim != 5 or not torch.is_complex(weight):
        raise ValueError(
            "Expected complex spectral weights with shape "
            "(in_channels, out_channels, modes_x, modes_y, modes_z)"
        )
    return (
        weight.detach()
        .abs()
        .square()
        .mean(dim=(0, 1))
        .float()
        .cpu()
        .numpy()
    )


def _grouped_rms(powers: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coordinate_max = int(coordinates.max(initial=0))
    profile = np.empty(coordinate_max + 1, dtype=np.float32)
    for index in range(coordinate_max + 1):
        selected = powers[coordinates == index]
        profile[index] = np.sqrt(selected.mean()) if selected.size else np.nan
    return profile


def _profiles_from_power_grids(
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    powers_by_axis = {axis: [] for axis in ("x", "y", "z", "shell")}
    coordinates_by_axis = {axis: [] for axis in powers_by_axis}

    for power, kx, ky, kz in entries:
        grid_x, grid_y, grid_z = np.meshgrid(kx, ky, kz, indexing="ij")
        shell = np.floor(
            np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        ).astype(np.int64)
        grids = {
            "x": grid_x,
            "y": grid_y,
            "z": grid_z,
            "shell": shell,
        }
        for axis, grid in grids.items():
            powers_by_axis[axis].append(power.reshape(-1))
            coordinates_by_axis[axis].append(grid.reshape(-1))

    return tuple(
        _grouped_rms(
            np.concatenate(powers_by_axis[axis]),
            np.concatenate(coordinates_by_axis[axis]),
        )
        for axis in ("x", "y", "z", "shell")
    )


def _centered_fno_profiles(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    power = _channel_power(weight)
    nx, ny, nz = power.shape
    kx = np.abs(np.arange(nx) - nx // 2)
    ky = np.abs(np.arange(ny) - ny // 2)
    kz = np.arange(nz)
    return _profiles_from_power_grids([(power, kx, ky, kz)])


def _ufno_quadrant_profiles(
    weights: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map U-FNO's four unshifted FFT quadrants to absolute frequencies.

    For a retained width ``m``, a positive slice ``:m`` maps tensor indices
    to frequencies ``0, ..., m-1``. A negative slice ``-m:`` maps them to
    ``-m, ..., -1``, so its absolute-frequency order is ``m, ..., 1``.
    """
    powers = [_channel_power(weight) for weight in weights]
    nx, ny, nz = powers[0].shape
    if any(power.shape != (nx, ny, nz) for power in powers):
        raise ValueError("U-FNO quadrant weights must have matching shapes")

    positive_x = np.arange(nx)
    negative_x = np.arange(nx, 0, -1)
    positive_y = np.arange(ny)
    negative_y = np.arange(ny, 0, -1)
    kz = np.arange(nz)
    coordinates = (
        (positive_x, positive_y, kz),
        (negative_x, positive_y, kz),
        (positive_x, negative_y, kz),
        (negative_x, negative_y, kz),
    )
    return _profiles_from_power_grids(
        [
            (power, kx, ky, z)
            for power, (kx, ky, z) in zip(powers, coordinates)
        ]
    )


@torch.no_grad()
def extract_spectral_weight_profiles(
    model: nn.Module,
) -> list[SpectralWeightProfile]:
    """Summarize each FNO/U-FNO/SirenFNO layer by absolute mode index."""
    profiles: list[SpectralWeightProfile] = []
    root = _unwrap_model(model)

    for name, module in root.named_modules():
        generated_weights = getattr(module, "spectral_weight_tensors", None)
        if callable(generated_weights):
            tensors = generated_weights()
            x, y, z, shell = _ufno_quadrant_profiles(tensors)
            profiles.append(SpectralWeightProfile(name, x, y, z, shell))
            continue

        # NeuralOperator SpectralConv. Dense and factorized tensors both
        # expose to_tensor(), so the analysis does not depend on storage form.
        weight = getattr(module, "weight", None)
        if (
            weight is not None
            and hasattr(weight, "to_tensor")
            and hasattr(module, "n_modes")
        ):
            x, y, z, shell = _centered_fno_profiles(weight.to_tensor())
            profiles.append(SpectralWeightProfile(name, x, y, z, shell))
            continue

        # Wen et al. U-FNO stores four tensors for the +/- X/Y quadrants.
        quadrant_names = ("weights1", "weights2", "weights3", "weights4")
        if all(hasattr(module, attr) for attr in quadrant_names):
            tensors = [getattr(module, attr) for attr in quadrant_names]
            x, y, z, shell = _ufno_quadrant_profiles(tensors)
            profiles.append(SpectralWeightProfile(name, x, y, z, shell))

    if not profiles:
        raise ValueError("No FNO, U-FNO, or SirenFNO spectral layers were found")
    return profiles


class SpectralWeightHistory:
    """Store compact per-epoch spectral profiles in one atomic NPZ file."""

    def __init__(
        self,
        path: str | Path,
        model: nn.Module,
        *,
        reset: bool = False,
    ):
        self.path = Path(path)
        self.model = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.unlink(missing_ok=True)

    def record(self, epoch: int) -> None:
        profiles = extract_spectral_weight_profiles(self.model)
        layers = np.asarray([profile.layer for profile in profiles], dtype=str)
        current = {
            axis: np.stack(
                [getattr(profile, axis) for profile in profiles], axis=0
            ).astype(np.float32)
            for axis in ("x", "y", "z", "shell")
        }

        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as saved:
                epochs = saved["epochs"].astype(np.int64)
                saved_layers = saved["layers"].astype(str)
                if not np.array_equal(saved_layers, layers):
                    raise ValueError(
                        "Spectral layer layout changed within one training run"
                    )
                history = {
                    axis: saved[axis].astype(np.float32)
                    for axis in current
                }
        else:
            epochs = np.empty(0, dtype=np.int64)
            history = {
                axis: np.empty((0, *values.shape), dtype=np.float32)
                for axis, values in current.items()
            }

        duplicate = np.flatnonzero(epochs == int(epoch))
        if duplicate.size:
            index = int(duplicate[-1])
            for axis, values in current.items():
                history[axis][index] = values
        else:
            epochs = np.append(epochs, int(epoch))
            for axis, values in current.items():
                history[axis] = np.concatenate(
                    [history[axis], values[None, ...]], axis=0
                )

        order = np.argsort(epochs)
        payload = {
            "format_version": np.asarray(HISTORY_FORMAT_VERSION),
            "epochs": epochs[order],
            "layers": layers,
            **{axis: values[order] for axis, values in history.items()},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(self.path)
