"""PyTorch Dataset for full 21cm lightcone cubes (density -> neutral fraction).

One sample = one lightcone, downsampled along the line-of-sight to a fixed
``n_z``-cell grid so the whole cube fits in a single FNO forward pass.

Input tensor layout: ``(C, Nx, Ny, Nz)`` with two channels per sample:

    channel 0 : density / density_scale
    channel 1 : 1 / (1 + z), broadcast across transverse axes

The ``positional_embedding="grid"`` option on the FNO supplies normalized
(x, y, z) coordinate channels automatically -- the z-coordinate doubles as
the normalized comoving distance because the native LOS cells are uniform in
comoving distance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from loader import LightconeFile


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
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.n_z = int(n_z)
        self.target_z = np.linspace(float(z_min), float(z_max), self.n_z,
                                    dtype=np.float64)
        self.input_field = input_field
        self.target_field = target_field
        self.density_scale = float(density_scale)
        self.preload = bool(preload)

        # Pre-compute the per-LOS redshift channel once; broadcast at __getitem__.
        self._z_channel_1d = (1.0 / (1.0 + self.target_z)).astype(np.float32)

        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.preload:
            for i in range(len(self.file_paths)):
                self._cache[i] = self._load(i)

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

        x = np.stack([c_dens, c_z], axis=0)            # (2, Nx, Ny, Nz)
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
    dataset: LightconeCubeDataset,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset, tuple[list[int], list[int], list[int]]]:
    """Split a :class:`LightconeCubeDataset` into train / val / test subsets.

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
