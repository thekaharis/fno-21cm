"""PyTorch Dataset for 21cm lightcone slices (density → neutral fraction)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from loader import LightconeFile


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
