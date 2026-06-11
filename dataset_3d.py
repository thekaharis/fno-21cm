"""PyTorch Dataset for full 21cm lightcone cubes.

One sample = one lightcone, downsampled along the line-of-sight to a fixed
``n_z``-cell grid so the whole cube fits in a single FNO forward pass.

Input tensor layout is ``(C, Nx, Ny, Nz)``. The selected channels are recorded
by :class:`InputFeatures`; parameter channels are normalized from training
cones only and use the same order in the raw and cached pipelines.

The ``positional_embedding="grid"`` option on the FNO supplies normalized
(x, y, grid-z) coordinate channels automatically. Since samples are
interpolated to a uniform redshift grid, grid-z is normalized redshift rather
than normalized comoving distance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from lightcone_params import (
    PARAM_NAMES as LIGHTCONE_PARAM_NAMES,
    read_sampled_params,
)
from loader import LightconeFile


@dataclass(frozen=True)
class InputFeatures:
    """Explicit input-channel contract for conditioning ablations."""

    name: str = "density_z_params"

    VALID = ("density", "params", "density_z", "density_z_params")

    def __post_init__(self) -> None:
        if self.name not in self.VALID:
            raise ValueError(
                f"input features must be one of {self.VALID}, got {self.name!r}"
            )

    @property
    def use_density(self) -> bool:
        return self.name in {"density", "density_z", "density_z_params"}

    @property
    def use_redshift(self) -> bool:
        return self.name in {"density_z", "density_z_params"}

    @property
    def use_params(self) -> bool:
        return self.name in {"params", "density_z_params"}

    @property
    def channel_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.use_density:
            names.append("density/10")
        if self.use_redshift:
            names.append("1/(1+z)")
        if self.use_params:
            names.extend(LIGHTCONE_PARAM_NAMES)
        return tuple(names)


@dataclass(frozen=True)
class ParameterNormalization:
    """Train-split statistics used to z-score sampled parameters."""

    names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        params: np.ndarray,
        train_indices: Sequence[int],
        names: Sequence[str] = LIGHTCONE_PARAM_NAMES,
    ) -> "ParameterNormalization":
        selected = np.asarray(params, dtype=np.float64)[list(train_indices)]
        if selected.size == 0:
            raise ValueError("cannot fit parameter normalization on an empty split")
        if not np.isfinite(selected).all():
            bad = np.where(~np.isfinite(selected).all(axis=1))[0]
            raise ValueError(
                "training parameters contain missing/non-finite values "
                f"(local rows {bad[:5].tolist()})"
            )
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(
            names=tuple(names),
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
        )

    @classmethod
    def from_dict(cls, values: dict) -> "ParameterNormalization":
        return cls(
            names=tuple(values["names"]),
            mean=tuple(float(value) for value in values["mean"]),
            std=tuple(float(value) for value in values["std"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def normalize(self, params: np.ndarray) -> np.ndarray:
        if tuple(self.names) != tuple(LIGHTCONE_PARAM_NAMES):
            raise ValueError("parameter normalization schema does not match dataset")
        return (
            (np.asarray(params, dtype=np.float32) - np.asarray(self.mean, dtype=np.float32))
            / np.asarray(self.std, dtype=np.float32)
        ).astype(np.float32)


def _coerce_input_features(value: InputFeatures | str) -> InputFeatures:
    return value if isinstance(value, InputFeatures) else InputFeatures(str(value))


class LightconeCubeDataset(Dataset):
    """One 3-D cube per lightcone, interpolated to a common z-grid.

    Parameters
    ----------
    file_paths : sequence of path-like
        Paths to the lightcone ``.h5`` files.
    n_z : int
        Number of LOS cells after interpolation (default 256).
    z_min, z_max : float
        Redshift range of the common LOS grid (default 5.0 - 25.0).
    input_field : str
        HDF5 dataset name for the density input.
    target_field : str
        HDF5 dataset name for the neutral-fraction target.
    density_scale : float
        Fixed divisor applied to the density input (matches the v2 pipeline).
    preload : bool
        If True, read and cache every cube at construction time. If False
        (default), read on demand from disk.
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        n_z: int = 256,
        z_min: float = 5.0,
        z_max: float = 25.0,
        input_field: str = "density",
        target_field: str = "neutral_fraction",
        density_scale: float = 10.0,
        preload: bool = False,
        input_features: InputFeatures | str = "density_z_params",
        parameter_normalization: ParameterNormalization | None = None,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        # Global cone id == position in the sorted file list (the same
        # ordering build_cubes.py records in the cache's `cone_id` dataset).
        self.cone_ids = np.arange(len(self.file_paths), dtype=np.int64)
        self.n_z = int(n_z)
        self.target_z = np.linspace(float(z_min), float(z_max), self.n_z,
                                    dtype=np.float64)
        self.input_field = input_field
        self.target_field = target_field
        self.density_scale = float(density_scale)
        self.preload = bool(preload)
        self.input_features = _coerce_input_features(input_features)
        self.parameter_normalization = parameter_normalization
        self.params = None
        if self.input_features.use_params:
            rows = []
            for path in self.file_paths:
                with h5py.File(path, "r") as h5_file:
                    rows.append(read_sampled_params(h5_file))
            self.params = np.stack(rows).astype(np.float32)
        self.n_params = len(LIGHTCONE_PARAM_NAMES) if self.input_features.use_params else 0
        self.in_channels = len(self.input_features.channel_names)

        # Pre-compute the per-LOS redshift channel once; broadcast at __getitem__.
        self._z_channel_1d = (1.0 / (1.0 + self.target_z)).astype(np.float32)

        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.preload:
            if self.input_features.use_params and self.parameter_normalization is None:
                raise ValueError(
                    "parameter_normalization must be set before preloading "
                    "parameter-conditioned raw cubes"
                )
            for i in range(len(self.file_paths)):
                self._cache[i] = self._load(i)

    def fit_parameter_normalization(
        self, train_indices: Sequence[int]
    ) -> ParameterNormalization | None:
        if not self.input_features.use_params:
            return None
        assert self.params is not None
        return ParameterNormalization.fit(self.params, train_indices)

    def set_parameter_normalization(
        self, normalization: ParameterNormalization | None
    ) -> None:
        if self.input_features.use_params and normalization is None:
            raise ValueError("parameter-conditioned inputs require normalization")
        self.parameter_normalization = normalization
        self._cache.clear()

    # ----------------------------------------------------------------- core
    def _load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.file_paths[idx]
        with LightconeFile(p) as lf:
            dens = lf.read_interpolated(self.input_field, self.target_z)
            xhi = lf.read_interpolated(self.target_field, self.target_z)

        # Density channel, normalized.
        c_dens = (dens / self.density_scale).astype(np.float32)

        # 1/(1+z) channel, broadcast across the transverse plane.
        nx, ny, nz = c_dens.shape
        c_z = np.broadcast_to(
            self._z_channel_1d.reshape(1, 1, nz), (nx, ny, nz)
        ).astype(np.float32, copy=False)

        channels = []
        if self.input_features.use_density:
            channels.append(c_dens)
        if self.input_features.use_redshift:
            channels.append(c_z)
        if self.input_features.use_params:
            if self.parameter_normalization is None:
                raise RuntimeError(
                    "parameter normalization has not been fitted or loaded"
                )
            assert self.params is not None
            normalized = self.parameter_normalization.normalize(self.params[idx])
            channels.extend(
                np.broadcast_to(value, (nx, ny, nz))
                for value in normalized
            )

        x = np.stack(channels, axis=0)
        y = xhi.astype(np.float32)[None, ...]          # (1, Nx, Ny, Nz)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index in self._cache:
            x, y = self._cache[index]
        else:
            x, y = self._load(index)
            if self.preload:
                self._cache[index] = (x, y)
        return {"x": x, "y": y}


# ===================================================================== cache
# The class below consumes the compact cube cache produced by the one-time
# ``build_cubes.py`` pass.  It replaces the on-the-fly ``LightconeCubeDataset``
# for runs where the raw HDF5 reads become the bottleneck (cluster project FS
# at ~370 MB/cone is the typical case).  Cubes are stored pre-interpolated to
# the same (Nx, Ny, n_z) grid the model trains on, so each per-sample read is
# ~10x smaller.


class LightconeCubeCache(Dataset):
    """In-memory or lazy dataset of pre-extracted 3-D cubes.

    Reads the compact HDF5 cache written by ``build_cubes.py`` (datasets
    ``density``, ``neutral_fraction``, ``cone_id``, ``target_z``, ``params``).

    Each item is one cube ``{"x": (C, Nx, Ny, Nz), "y": (1, Nx, Ny, Nz)}``,
    where ``C = 2`` if ``use_params=False`` (density + 1/(1+z)) or
    ``C = 2 + n_params = 13`` if ``use_params=True`` (the 11 astrophysical
    parameters are z-scored against the training distribution and broadcast
    as constant channels over the entire cube).

    Parameters
    ----------
    cache_path : path-like
        Path to ``cubes_3d.h5`` (or merged shard).
    density_scale : float
        Fixed divisor applied to the density input (default 10.0).
    preload : bool
        If True, load every cube into RAM at construction (~420 GB for a full
        cluster run -- only sensible on fat-mem nodes or small subsets).
    use_params : bool
        If True (default), include the 11 astrophysical parameters as broadcast
        input channels.  Provides the model the conditioning it needs to
        disambiguate reionization histories that produce similar densities --
        empirically the difference between plateauing at val_l2 ~0.20 and
        actually learning bubble morphology.
    """

    # Names of the 11 LHS-sampled parameters in the cache, in column order.
    # Matches build_cubes.PARAMS exactly.
    PARAM_NAMES = LIGHTCONE_PARAM_NAMES

    def __init__(self, cache_path: str | Path,
                 density_scale: float = 10.0,
                 preload: bool = False,
                 use_params: bool | None = None,
                 input_features: InputFeatures | str = "density_z_params",
                 parameter_normalization: ParameterNormalization | None = None):
        self.cache_path = Path(cache_path)
        self.density_scale = float(density_scale)
        self.preload = bool(preload)
        if use_params is not None:
            input_features = "density_z_params" if use_params else "density_z"
        self.input_features = _coerce_input_features(input_features)
        self.use_params = self.input_features.use_params
        self.parameter_normalization = parameter_normalization

        with h5py.File(self.cache_path, "r") as f:
            self.target_z = np.asarray(f["target_z"], dtype=np.float64)
            # Global file index of each cache row.  Caches merged with the
            # cone_id-sorted merge have cone_ids == row index; older caches
            # are shard-interleaved, which is why downstream split / label
            # logic must go through this map instead of row positions.
            self.cone_ids = np.asarray(f["cone_id"], dtype=np.int64)
            self.n_cones, self.n_x, self.n_y, self.n_z = f["density"].shape
            if "params" in f:
                self.params = np.asarray(f["params"], dtype=np.float32)
            else:
                self.params = None
            if preload:
                self._dens = np.asarray(f["density"][...], dtype=np.float32)
                self._xhi = np.asarray(f["neutral_fraction"][...],
                                       dtype=np.float32)
            else:
                self._dens = None
                self._xhi = None

        self._z_channel_1d = (1.0 / (1.0 + self.target_z)).astype(np.float32)

        if self.use_params:
            if self.params is None:
                raise ValueError(
                    f"use_params=True but {self.cache_path} has no `params` "
                    "dataset -- rebuild the cache with build_cubes.py.")
            if np.isnan(self.params).any():
                # NaNs come from build_cubes.read_params() when a lightcone
                # has an unexpected param layout; we can't condition on those.
                bad = np.where(np.isnan(self.params).any(axis=1))[0]
                raise ValueError(
                    f"{len(bad)}/{len(self.params)} cones have NaN params in "
                    f"{self.cache_path} (first bad indices: {bad[:5].tolist()}). "
                    "Either rebuild the cache or pass use_params=False.")
            self.n_params = self.params.shape[1]
        else:
            self.n_params = 0

        self.in_channels = len(self.input_features.channel_names)

        # Lazy per-process h5py handle (one per DataLoader worker after fork).
        self._h5: h5py.File | None = None

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __len__(self) -> int:
        return int(self.n_cones)

    def fit_parameter_normalization(
        self, train_indices: Sequence[int]
    ) -> ParameterNormalization | None:
        if not self.input_features.use_params:
            return None
        assert self.params is not None
        return ParameterNormalization.fit(self.params, train_indices)

    def set_parameter_normalization(
        self, normalization: ParameterNormalization | None
    ) -> None:
        if self.input_features.use_params and normalization is None:
            raise ValueError("parameter-conditioned inputs require normalization")
        self.parameter_normalization = normalization

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._dens is not None:
            dens = self._dens[idx]
            xhi = self._xhi[idx]
        else:
            f = self._ensure_open()
            dens = np.asarray(f["density"][idx], dtype=np.float32)
            xhi = np.asarray(f["neutral_fraction"][idx], dtype=np.float32)

        c_dens = dens / self.density_scale
        c_z = np.broadcast_to(
            self._z_channel_1d.reshape(1, 1, self.n_z),
            (self.n_x, self.n_y, self.n_z),
        ).astype(np.float32, copy=False)
        channels = []
        if self.input_features.use_density:
            channels.append(c_dens)
        if self.input_features.use_redshift:
            channels.append(c_z)

        # Append the 11 z-scored astrophysical params as constant channels.
        # Each is a scalar per-cone, broadcast over the (Nx, Ny, Nz) volume so
        # the lifting layer's 1x1 conv can mix them with the spatial inputs.
        if self.input_features.use_params:
            if self.parameter_normalization is None:
                raise RuntimeError(
                    "parameter normalization has not been fitted or loaded"
                )
            assert self.params is not None
            params_normed = self.parameter_normalization.normalize(self.params[idx])
            for j in range(self.n_params):
                c_p = np.broadcast_to(
                    np.float32(params_normed[j]),
                    (self.n_x, self.n_y, self.n_z),
                )
                channels.append(c_p)

        x = np.stack(channels, axis=0)                    # (C, Nx, Ny, Nz)
        y = xhi[None, ...]                                # (1, Nx, Ny, Nz)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
        }


# --------------------------------------------------------------------- split
def make_file_split(
    n_files: int,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Seeded shuffled split of file indices into train / val / test.

    Mirrors ``dataset.make_file_split`` so the two pipelines pick the same
    held-out cones under identical seeds.  Val and test each get at least one
    file even on small datasets.
    """
    import random

    perm = list(range(n_files))
    random.Random(seed).shuffle(perm)
    n_test = max(1, round(test_frac * n_files))
    n_val = max(1, round(val_frac * n_files))
    test = sorted(perm[:n_test])
    val = sorted(perm[n_test:n_test + n_val])
    train = sorted(perm[n_test + n_val:])
    return train, val, test


def split_cubes(
    dataset: LightconeCubeDataset | LightconeCubeCache,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset, tuple[list[int], list[int], list[int]]]:
    """Split a cube dataset into train / val / test subsets.

    Works with either :class:`LightconeCubeDataset` (raw streaming) or
    :class:`LightconeCubeCache` (pre-computed cache) -- both expose the same
    one-cube-per-index interface, so the split is index-level.

    Returns ``(train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx))``.
    The raw index lists are returned alongside the ``Subset`` views so the
    training script can print and sanity-check them.
    """
    train_idx, val_idx, test_idx = make_file_split(
        len(dataset), seed=seed, val_frac=val_frac, test_frac=test_frac,
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
        (train_idx, val_idx, test_idx),
    )


def rows_for_cone_ids(
    dataset: LightconeCubeDataset | LightconeCubeCache,
    cone_ids: Sequence[int],
) -> list[int]:
    """Map global cone ids to dataset row indices via ``dataset.cone_ids``.

    Cone ids are invariant to cache row ordering (shard-interleaved merges,
    rebuilds with a different shard count), so they are the safe currency for
    persisting a train/val/test split across runs and pipelines.
    """
    row_by_id = {
        int(cid): row for row, cid in enumerate(np.asarray(dataset.cone_ids))
    }
    missing = sorted({int(c) for c in cone_ids} - row_by_id.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} recorded cone ids are absent from this dataset "
            f"(first few: {missing[:5]}); the cache/file set does not cover "
            "the training-time cones"
        )
    return [row_by_id[int(c)] for c in cone_ids]


def resolve_split(
    dataset: LightconeCubeDataset | LightconeCubeCache,
    run_metadata: dict | None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], str]:
    """Resolve train/val/test rows, preferring the recorded training split.

    Preference order:
      1. ``*_cone_ids`` from run metadata -- invariant to cache row order.
      2. ``*_indices`` from run metadata (legacy runs) -- row positions,
         only valid with the exact training-time cache ordering.
      3. Recomputed seeded split -- last resort when no metadata exists;
         matches training only if the dataset length is unchanged.

    Returns ``(train_rows, val_rows, test_rows, source)`` where *source*
    describes which path was taken (for logging).
    """
    split_meta = (run_metadata or {}).get("split") or {}
    if "train_cone_ids" in split_meta:
        train_rows, val_rows, test_rows = (
            rows_for_cone_ids(dataset, split_meta[key])
            for key in ("train_cone_ids", "val_cone_ids", "test_cone_ids")
        )
        return train_rows, val_rows, test_rows, "run metadata (cone ids)"
    if "train_indices" in split_meta:
        train_rows, val_rows, test_rows = (
            [int(i) for i in split_meta[key]]
            for key in ("train_indices", "val_indices", "test_indices")
        )
        highest = max(
            (max(rows) for rows in (train_rows, val_rows, test_rows) if rows),
            default=-1,
        )
        if highest >= len(dataset):
            raise ValueError(
                f"run metadata row index {highest} is out of range for this "
                f"dataset of {len(dataset)} cones; the cache does not match "
                "the training-time data"
            )
        return (
            train_rows, val_rows, test_rows,
            "run metadata (legacy row indices; assumes the training-time "
            "cache ordering)",
        )
    train_rows, val_rows, test_rows = make_file_split(
        len(dataset), seed=seed, val_frac=val_frac, test_frac=test_frac,
    )
    return (
        train_rows, val_rows, test_rows,
        "recomputed from seed (no run metadata)",
    )
