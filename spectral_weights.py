"""Extract and persist human-readable Fourier-weight summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HISTORY_FILENAME = "spectral_weight_history.npz"


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


def _channel_rms(weights: list[torch.Tensor]) -> np.ndarray:
    """RMS complex magnitude over input/output channels and weight quadrants."""
    powers = []
    for weight in weights:
        if weight.ndim != 5 or not torch.is_complex(weight):
            raise ValueError(
                "Expected complex spectral weights with shape "
                "(in_channels, out_channels, modes_x, modes_y, modes_z)"
            )
        powers.append(weight.detach().abs().square().mean(dim=(0, 1)))
    mean_power = torch.stack(powers).mean(dim=0)
    return mean_power.sqrt().float().cpu().numpy()


def _grouped_rms(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coordinate_max = int(coordinates.max(initial=0))
    profile = np.empty(coordinate_max + 1, dtype=np.float32)
    squared = np.square(values, dtype=np.float64)
    for index in range(coordinate_max + 1):
        selected = squared[coordinates == index]
        profile[index] = np.sqrt(selected.mean()) if selected.size else np.nan
    return profile


def _profiles_from_magnitude(
    magnitude: np.ndarray,
    *,
    centered_xy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = magnitude.shape
    if centered_xy:
        kx = np.abs(np.arange(nx) - nx // 2)
        ky = np.abs(np.arange(ny) - ny // 2)
    else:
        kx = np.arange(nx)
        ky = np.arange(ny)
    kz = np.arange(nz)

    grid_x, grid_y, grid_z = np.meshgrid(kx, ky, kz, indexing="ij")
    shell = np.floor(
        np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    ).astype(np.int64)
    return (
        _grouped_rms(magnitude, grid_x),
        _grouped_rms(magnitude, grid_y),
        _grouped_rms(magnitude, grid_z),
        _grouped_rms(magnitude, shell),
    )


@torch.no_grad()
def extract_spectral_weight_profiles(
    model: nn.Module,
) -> list[SpectralWeightProfile]:
    """Summarize each FNO/U-FNO spectral layer by absolute mode index."""
    profiles: list[SpectralWeightProfile] = []
    root = _unwrap_model(model)

    for name, module in root.named_modules():
        # NeuralOperator SpectralConv. Dense and factorized tensors both
        # expose to_tensor(), so the analysis does not depend on storage form.
        weight = getattr(module, "weight", None)
        if (
            weight is not None
            and hasattr(weight, "to_tensor")
            and hasattr(module, "n_modes")
        ):
            magnitude = _channel_rms([weight.to_tensor()])
            x, y, z, shell = _profiles_from_magnitude(
                magnitude, centered_xy=True
            )
            profiles.append(SpectralWeightProfile(name, x, y, z, shell))
            continue

        # Wen et al. U-FNO stores four tensors for the +/- X/Y quadrants.
        quadrant_names = ("weights1", "weights2", "weights3", "weights4")
        if all(hasattr(module, attr) for attr in quadrant_names):
            tensors = [getattr(module, attr) for attr in quadrant_names]
            magnitude = _channel_rms(tensors)
            x, y, z, shell = _profiles_from_magnitude(
                magnitude, centered_xy=False
            )
            profiles.append(SpectralWeightProfile(name, x, y, z, shell))

    if not profiles:
        raise ValueError("No FNO or U-FNO spectral weight layers were found")
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
            "epochs": epochs[order],
            "layers": layers,
            **{axis: values[order] for axis, values in history.items()},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(self.path)
