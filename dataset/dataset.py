"""PyTorch Dataset for 21cm lightcone slices (density → neutral fraction)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from dataset.loader import LightconeFile
from dataset.dataset_3d import InputFeatures, ParameterNormalization
from dataset.lightcone_params import PARAM_NAMES


SLICE_CACHE_VERSION = 2


def _interp_field(lf: LightconeFile, field: str,
                  target_z: np.ndarray) -> np.ndarray:
    data = lf.read_full(field)
    src_z = lf.los_redshifts()
    from scipy.interpolate import interp1d
    n_x, n_y, _ = data.shape
    flat = data.reshape(-1, data.shape[2])
    fn = interp1d(src_z, flat, kind="linear", axis=1,
                  bounds_error=False, fill_value=0.0, assume_sorted=True)
    return fn(target_z).reshape(n_x, n_y, -1).astype(np.float32)


class LightconeSliceDataset(Dataset):
    """Per-redshift 2-D slices from interpolated lightcone cubes.

    Each file is interpolated to a common redshift grid of *n_z* points.
    A single ``__getitem__`` returns one 2-D slice ``(1, 140, 140)``
    as a dict ``{"x": density, "y": neutral_fraction}``.

    Parameters
    ----------
    file_paths : list of path-like
        Paths to ``.h5`` lightcone files.
    n_z : int
        Number of redshift grid points (default 256).
    z_min, z_max : float
        Redshift range for the common grid (default 5.0–25.0).
    input_field : str
        HDF5 dataset name for the input (default ``"density"``).
    target_field : str
        HDF5 dataset name for the target (default ``"neutral_fraction"``).
    preload : bool
        If True, load and interpolate all files at construction time.
        If False, load each file on-the-fly (slower, lower memory).
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        n_z: int = 256,
        z_min: float = 5.0,
        z_max: float = 25.0,
        input_field: str = "density",
        target_field: str = "neutral_fraction",
        preload: bool = True,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.n_z = n_z
        self.target_z = np.linspace(z_min, z_max, n_z, dtype=np.float64)
        self.input_field = input_field
        self.target_field = target_field
        self.preload = preload
        self.n_files = len(self.file_paths)

        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if preload:
            for i, p in enumerate(self.file_paths):
                self._cache[i] = self._load_file(i)

    def _load_file(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.file_paths[idx]
        with LightconeFile(p) as lf:
            x = _interp_field(lf, self.input_field, self.target_z)
            y = _interp_field(lf, self.target_field, self.target_z)
        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(y.copy()).float(),
        )

    def __len__(self) -> int:
        return self.n_files * self.n_z

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx = index // self.n_z
        z_idx = index % self.n_z
        try:
            x_all, y_all = self._cache[file_idx]
        except KeyError:
            self._cache[file_idx] = self._load_file(file_idx)
            x_all, y_all = self._cache[file_idx]
        return {
            "x": x_all[:, :, z_idx].unsqueeze(0) / 10.0,
            "y": y_all[:, :, z_idx].unsqueeze(0),
        }

    def file_ids(self) -> list[int]:
        """Return list of file indices for each sample, in order."""
        return [i // self.n_z for i in range(len(self))]


def split_by_file(
    dataset: LightconeSliceDataset,
    train_files: Sequence[int],
    val_files: Sequence[int] | None = None,
    test_files: Sequence[int] | None = None,
) -> tuple[LightconeSliceDataset, ...]:
    """Split an existing dataset by file indices.

    Returns subsets that are themselves ``LightconeSliceDataset``-compatible
    via ``Subset``; each contains only the slices belonging to the given file
    indices.

    Parameters
    ----------
    dataset : LightconeSliceDataset
    train_files : sequence of int
        File indices for the training split.
    val_files : sequence of int or None
        File indices for the validation split.
    test_files : sequence of int or None
        File indices for the test split.

    Returns
    -------
    train, val, test : Tuple[Dataset, ...]
        Always returns a 3-tuple; val/test are ``None`` if not provided.
    """
    idx_map: dict[int, list[int]] = {}
    for global_idx, fid in enumerate(dataset.file_ids()):
        idx_map.setdefault(fid, []).append(global_idx)

    def _subset(file_ids):
        if file_ids is None:
            return None
        indices = []
        for fid in file_ids:
            indices.extend(idx_map[fid])
        return Subset(dataset, indices)

    return _subset(train_files), _subset(val_files), _subset(test_files)


def make_file_split(
    n_files: int,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic shuffled split of file *indices* into train/val/test.

    The lightcone files come from a space-filling parameter design, so a
    seeded random partition gives val/test sets that are representative of
    the full parameter space.  Using a fixed *seed* makes the split identical
    across training and evaluation runs.

    Returns sorted index lists ``(train, val, test)``.  Val and test each get
    at least one file.
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


# ===================================================================== cache
# The classes below consume the compact ``trainset.h5`` produced by the
# one-time ``dataset/build_trainset.py`` pass (few slices per cone, many cones). They
# replace the on-the-fly ``LightconeSliceDataset`` for large datasets that
# cannot be preloaded from raw lightcones.


class SliceCache(Dataset):
    """In-memory dataset of pre-extracted 2-D slices.

    Reads the compact HDF5 cache written by ``dataset/build_trainset.py`` (datasets
    ``x``, ``y``, ``z``, ``xHI_mean``, ``cone_id``, ``params``) fully into RAM.
    Each item is one slice ``{"x": density / scale, "y": x_HI}`` of shape
    ``(1, 140, 140)`` -- the same interface ``LightconeSliceDataset`` exposed,
    so the model / trainer code is unchanged.

    Parameters
    ----------
    cache_path : path-like
        Path to ``trainset.h5``.
    density_scale : float
        Fixed divisor applied to the density input (default 10.0), matching
        the normalization used everywhere else.
    """

    def __init__(
        self,
        cache_path: str | Path,
        density_scale: float = 10.0,
        input_features: InputFeatures | str = "density",
        parameter_normalization: ParameterNormalization | None = None,
    ):
        self.cache_path = Path(cache_path)
        self.density_scale = float(density_scale)
        self.input_features = (
            input_features
            if isinstance(input_features, InputFeatures)
            else InputFeatures(str(input_features))
        )
        self.parameter_normalization = parameter_normalization
        self._normalized_params: np.ndarray | None = None
        with h5py.File(self.cache_path, "r") as f:
            version = int(f.attrs.get("slice_cache_version", 0))
            if version != SLICE_CACHE_VERSION:
                raise ValueError(
                    f"slice cache version {version} is unsupported; expected "
                    f"{SLICE_CACHE_VERSION}. Rebuild the cache and every shard."
                )
            self.x = f["x"][:].astype(np.float32)          # (N, 140, 140)
            self.y = f["y"][:].astype(np.float32)
            self.cone_id = f["cone_id"][:].astype(np.int64)
            self.z = f["z"][:].astype(np.float32)
            self.xHI_mean = f["xHI_mean"][:].astype(np.float32)
            self.params = (f["params"][:].astype(np.float32)
                           if "params" in f else None)
            raw_names = f.attrs.get("param_names", ())
            self.param_names = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in raw_names
            )
            self.selection = {
                "k_per_cone": int(f.attrs.get("k_per_cone", 0)),
                "xHI_window": [
                    float(value) for value in f.attrs.get("xHI_window", ())
                ],
            }

        n = len(self.x)
        for name, values in (
            ("y", self.y), ("cone_id", self.cone_id), ("z", self.z),
            ("xHI_mean", self.xHI_mean),
        ):
            if len(values) != n:
                raise ValueError(f"slice-cache {name} length does not match x")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("slice cache contains non-finite fields")
        if not np.isfinite(self.z).all():
            raise ValueError("slice cache contains non-finite redshifts")
        if not np.isfinite(self.xHI_mean).all():
            raise ValueError("slice cache contains non-finite mean neutral fractions")
        if self.input_features.use_params:
            if self.params is None:
                raise ValueError("parameter-conditioned inputs require cache params")
            if (self.params.ndim != 2 or len(self.params) != n
                    or self.params.shape[1] != len(PARAM_NAMES)):
                raise ValueError("slice-cache parameter array has the wrong shape")
            if not np.isfinite(self.params).all():
                bad = np.unique(self.cone_id[~np.isfinite(self.params).all(axis=1)])
                raise ValueError(
                    "slice-cache parameters contain non-finite values for cone IDs "
                    f"{bad[:10].tolist()}"
                )
            if self.param_names != tuple(PARAM_NAMES):
                raise ValueError("slice-cache parameter names do not match schema")
        if self.parameter_normalization is not None:
            self.set_parameter_normalization(self.parameter_normalization)

        self.in_channels = len(self.input_features.channel_names)
        self.channel_names = self.input_features.channel_names

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        shape = self.x[i].shape
        channels: list[np.ndarray] = []
        if self.input_features.use_density:
            channels.append(self.x[i] / self.density_scale)
        if self.input_features.use_redshift:
            channels.append(np.full(
                shape, 1.0 / (1.0 + float(self.z[i])), dtype=np.float32
            ))
        if self.input_features.use_params:
            if self._normalized_params is None:
                raise RuntimeError(
                    "fit or set parameter normalization before reading samples"
                )
            channels.extend(
                np.full(shape, value, dtype=np.float32)
                for value in self._normalized_params[i]
            )
        return {
            "x": torch.from_numpy(np.stack(channels)),
            "y": torch.from_numpy(self.y[i][None]),
            "z": torch.tensor(float(self.z[i]), dtype=torch.float32),
            "xhi_mean": torch.tensor(
                float(self.xHI_mean[i]), dtype=torch.float32
            ),
            "cone_id": torch.tensor(int(self.cone_id[i]), dtype=torch.int64),
        }

    def fit_parameter_normalization(
        self, train_indices: Sequence[int]
    ) -> ParameterNormalization:
        """Fit once per training cone, not once per correlated slice."""
        if self.params is None:
            raise ValueError("slice cache has no parameters")
        indices = np.asarray(list(train_indices), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("cannot fit normalization on an empty split")
        _, first = np.unique(self.cone_id[indices], return_index=True)
        cone_rows = indices[first]
        normalization = ParameterNormalization.fit(
            self.params, cone_rows, names=PARAM_NAMES
        )
        self.set_parameter_normalization(normalization)
        return normalization

    def set_parameter_normalization(
        self, normalization: ParameterNormalization
    ) -> None:
        if self.params is None:
            raise ValueError("slice cache has no parameters")
        self.parameter_normalization = normalization
        self._normalized_params = normalization.normalize(self.params)


def split_by_cone(
    cache: SliceCache,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split a :class:`SliceCache` into train/val/test by **cone**.

    Splitting on ``cone_id`` (not on individual slices) guarantees that no cone
    contributes slices to more than one split -- otherwise correlated slices
    from the same lightcone leak between train and val, inflating the score.

    Returns ``(train, val, test)`` as ``Subset`` views over *cache*.
    """
    cones = np.unique(cache.cone_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(cones)
    n = len(cones)
    n_test = max(1, round(test_frac * n))
    n_val = max(1, round(val_frac * n))
    test_c = set(cones[:n_test].tolist())
    val_c = set(cones[n_test:n_test + n_val].tolist())
    train_c = set(cones[n_test + n_val:].tolist())

    def _subset(cone_set: set[int]) -> Subset:
        idx = np.where(np.isin(cache.cone_id, list(cone_set)))[0]
        return Subset(cache, idx.tolist())

    return _subset(train_c), _subset(val_c), _subset(test_c)


def build_dataloaders(
    dataset: LightconeSliceDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Convenience: split by file and wrap in DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds, val_ds, test_ds = split_by_file(dataset, train_idx, val_idx, test_idx)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds is not None else None
    return train_loader, val_loader, test_loader
