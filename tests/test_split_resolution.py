from __future__ import annotations

import h5py
import numpy as np
import pytest

from dataset_3d import (
    LightconeCubeCache,
    make_file_split,
    resolve_split,
    rows_for_cone_ids,
)
from lightcone_params import PARAM_NAMES


def _write_cache(path, cone_ids):
    cone_ids = np.asarray(cone_ids, dtype=np.int64)
    n = len(cone_ids)
    shape = (n, 2, 2, 3)
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=np.zeros(shape, dtype=np.float32))
        f.create_dataset("neutral_fraction",
                         data=np.zeros(shape, dtype=np.float32))
        f.create_dataset("cone_id", data=cone_ids)
        f.create_dataset(
            "params", data=np.zeros((n, len(PARAM_NAMES)), dtype=np.float32)
        )
        f.create_dataset("target_z", data=np.linspace(5.0, 7.0, 3))


class _FakeDataset:
    """Stand-in exposing just the interface resolve_split needs."""

    def __init__(self, n):
        self.cone_ids = np.arange(n, dtype=np.int64)
        self._n = n

    def __len__(self):
        return self._n


def test_rows_for_cone_ids_handles_unsorted_cache(tmp_path):
    path = tmp_path / "cache.h5"
    _write_cache(path, [0, 3, 1, 4, 2])      # shard-interleaved row order
    dataset = LightconeCubeCache(path)
    assert rows_for_cone_ids(dataset, [1, 4, 0]) == [2, 3, 0]
    with pytest.raises(ValueError, match="absent"):
        rows_for_cone_ids(dataset, [99])


def test_resolve_split_prefers_recorded_cone_ids(tmp_path):
    path = tmp_path / "cache.h5"
    _write_cache(path, [0, 3, 1, 4, 2])
    dataset = LightconeCubeCache(path)
    metadata = {"split": {
        # Row indices are decoys here: cone ids must win.
        "train_indices": [0, 1, 2],
        "val_indices": [3],
        "test_indices": [4],
        "train_cone_ids": [0, 1, 2],
        "val_cone_ids": [3],
        "test_cone_ids": [4],
    }}
    train, val, test, source = resolve_split(dataset, metadata)
    assert (train, val, test) == ([0, 2, 4], [1], [3])
    assert "cone ids" in source


def test_resolve_split_falls_back_to_legacy_indices():
    metadata = {"split": {
        "train_indices": [0, 1, 2],
        "val_indices": [3],
        "test_indices": [4],
    }}
    train, val, test, source = resolve_split(_FakeDataset(5), metadata)
    assert (train, val, test) == ([0, 1, 2], [3], [4])
    assert "legacy" in source


def test_resolve_split_rejects_out_of_range_legacy_indices():
    metadata = {"split": {
        "train_indices": [0, 1],
        "val_indices": [2],
        "test_indices": [7],
    }}
    with pytest.raises(ValueError, match="out of range"):
        resolve_split(_FakeDataset(5), metadata)


def test_resolve_split_recomputes_without_metadata():
    expected = make_file_split(10, seed=42, val_frac=0.1, test_frac=0.1)
    train, val, test, source = resolve_split(_FakeDataset(10), None)
    assert (train, val, test) == expected
    assert "recomputed" in source
