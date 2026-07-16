"""PyTorch Dataset mapping density lightcones to 2-D z_re(x, y) maps.

One sample = one cone. The 3-D density lightcone is interpolated to
``n_z_in`` LOS slices which enter a 2-D FNO as input *channels* (the output
map has no LOS axis, so LOS translation equivariance is not needed). Optional
sampled-parameter channels are broadcast as constant maps, mirroring the 3-D
pipeline's conditioning.

Sample layout (dict, matching the other datasets / neuralop Trainer):

* ``x``    -- ``(n_z_in [+ n_params], Nx, Ny)`` float32
* ``y``    -- ``(1, Nx, Ny)`` float32, z_re normalized to [0, 1] via
  ``(z_re - z_min) / (z_max - z_min)``; NaN target pixels (transition outside
  the cone) are filled with 0, i.e. clamped to the z_min edge
* ``mask`` -- ``(1, Nx, Ny)`` float32, 1 where the target pixel has a real
  transition inside the cone, 0 where it was filled

Targets come from the sidecar cache built by ``dataset/zre_target.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from dataset.dataset_3d import ParameterNormalization
from dataset.lightcone_params import PARAM_NAMES, read_sampled_params
from dataset.loader import LightconeFile
from dataset.zre_target import TARGET_KINDS, load_zre_map


class ZreMapDataset(Dataset):
    """Density lightcone -> z_re map pairs.

    With ``preload=True`` (training default) every cone's density slices are
    read once at construction: ~5 MB per cone at ``n_z_in=64``, ~33 GB for
    the full 6600-cone design -- fits the training node's memory budget and
    removes all epoch-time I/O. Visualization passes ``preload=False`` to
    read density on demand for just the cones it renders; targets and masks
    (~80 KB per cone) are always kept in memory.
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        target_cache: str | Path,
        target_kind: str = "gompertz",
        n_z_in: int = 64,
        z_min: float = 5.0,
        z_max: float = 25.0,
        density_scale: float = 10.0,
        use_params: bool = True,
        parameter_normalization: ParameterNormalization | None = None,
        preload: bool = True,
    ):
        if target_kind not in TARGET_KINDS:
            raise ValueError(
                f"target_kind must be one of {TARGET_KINDS}, got {target_kind!r}"
            )
        self.file_paths = [Path(p) for p in file_paths]
        self.target_cache = Path(target_cache)
        self.target_kind = target_kind
        self.n_z_in = int(n_z_in)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.density_scale = float(density_scale)
        self.use_params = bool(use_params)
        self.parameter_normalization = parameter_normalization
        self.input_z = np.linspace(self.z_min, self.z_max, self.n_z_in,
                                   dtype=np.float64)

        self.params: np.ndarray | None = None
        if self.use_params:
            rows = []
            for path in self.file_paths:
                with h5py.File(path, "r") as h5_file:
                    rows.append(read_sampled_params(h5_file))
            self.params = np.stack(rows).astype(np.float32)

        self.preload = bool(preload)
        self._density: dict[int, torch.Tensor] = {}
        self._target: list[torch.Tensor] = []
        self._mask: list[torch.Tensor] = []
        for idx, path in enumerate(self.file_paths):
            if self.preload:
                self._density[idx] = self._load_density(idx)

            zre = load_zre_map(self.target_cache, path, self.target_kind)
            valid = np.isfinite(zre)
            norm = (zre - self.z_min) / (self.z_max - self.z_min)
            norm = np.where(valid, norm, 0.0).astype(np.float32)
            self._target.append(torch.from_numpy(norm[None]))
            self._mask.append(
                torch.from_numpy(valid.astype(np.float32)[None])
            )

        self.map_shape = tuple(self._target[0].shape[-2:])
        self.n_params = len(PARAM_NAMES) if self.use_params else 0
        self.in_channels = self.n_z_in + self.n_params

    def _load_density(self, idx: int) -> torch.Tensor:
        with LightconeFile(self.file_paths[idx]) as lf:
            dens = lf.read_interpolated("density", self.input_z)
        # (Nx, Ny, n_z_in) -> channels-first (n_z_in, Nx, Ny)
        dens = np.moveaxis(dens, -1, 0) / self.density_scale
        return torch.from_numpy(np.ascontiguousarray(dens))

    @property
    def channel_names(self) -> tuple[str, ...]:
        names = [f"density/{self.density_scale:g}@z={z:.2f}"
                 for z in self.input_z]
        if self.use_params:
            names.extend(PARAM_NAMES)
        return tuple(names)

    def fit_parameter_normalization(
        self, train_indices: Sequence[int]
    ) -> ParameterNormalization | None:
        if not self.use_params:
            return None
        assert self.params is not None
        return ParameterNormalization.fit(self.params, train_indices)

    def set_parameter_normalization(
        self, normalization: ParameterNormalization | None
    ) -> None:
        if self.use_params and normalization is None:
            raise ValueError("parameter-conditioned inputs require normalization")
        self.parameter_normalization = normalization

    def denormalize_zre(self, y: torch.Tensor | np.ndarray):
        """Map a normalized target/prediction back to redshift units."""
        return y * (self.z_max - self.z_min) + self.z_min

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = (self._density[idx] if self.preload
             else self._load_density(idx))
        if self.use_params:
            if self.parameter_normalization is None:
                raise RuntimeError(
                    "call set_parameter_normalization() before sampling a "
                    "parameter-conditioned ZreMapDataset"
                )
            values = self.parameter_normalization.normalize(
                self.params[idx : idx + 1]
            )[0]
            param_maps = torch.from_numpy(values).view(-1, 1, 1).expand(
                -1, *self.map_shape
            )
            x = torch.cat([x, param_maps], dim=0)
        return {"x": x, "y": self._target[idx], "mask": self._mask[idx]}


def split_by_cone(
    dataset: ZreMapDataset,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Seeded train/val/test split; one cone is one sample, so a plain
    shuffled index split already prevents cross-split leakage."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    n_test = max(1, round(test_frac * len(dataset)))
    n_val = max(1, round(val_frac * len(dataset)))
    test_idx = order[:n_test].tolist()
    val_idx = order[n_test : n_test + n_val].tolist()
    train_idx = order[n_test + n_val :].tolist()
    if not train_idx:
        raise ValueError(
            f"split left no training cones (n={len(dataset)}, "
            f"val={n_val}, test={n_test})"
        )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
