from __future__ import annotations

import h5py
import numpy as np

from dataset_3d import (
    LightconeCubeCache,
    LightconeCubeDataset,
    split_cubes,
)
from lightcone_params import PARAM_NAMES


def _write_raw(path, density, xhi, target_z, params):
    with h5py.File(path, "w") as handle:
        lightcone = handle.create_group("lightcone")
        lightcone.create_dataset("density", data=density)
        lightcone.create_dataset("neutral_fraction", data=xhi)
        lightcone.create_dataset("lightcone_redshifts", data=target_z)
        group = handle.create_group("params")
        group.create_dataset("names", data=np.asarray(PARAM_NAMES, dtype="S"))
        group.create_dataset("values", data=params)


def _write_cache(path, density_stack, xhi_stack, target_z, cone_ids, params):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=density_stack)
        handle.create_dataset("neutral_fraction", data=xhi_stack)
        handle.create_dataset("target_z", data=target_z)
        handle.create_dataset("cone_id", data=np.asarray(cone_ids, dtype=np.int32))
        handle.create_dataset("params", data=params)
        handle.attrs["z_min"] = float(target_z[0])
        handle.attrs["z_max"] = float(target_z[-1])


def _make_cube(nx, ny, n_z, value):
    density = np.full((nx, ny, n_z), float(value), dtype=np.float32)
    xhi = np.full((nx, ny, n_z), float(value) / 10.0, dtype=np.float32)
    return density, xhi


def test_split_cubes_uses_cone_ids_not_row_positions(tmp_path):
    """A shard-interleaved cache must still split on physical cone ids."""
    n_cones = 10
    nx = ny = 2
    n_z = 4
    target_z = np.linspace(5.0, 8.0, n_z, dtype=np.float32)

    # Row order is intentionally not cone_id order.
    row_cone_ids = [3, 0, 4, 1, 5, 2, 9, 6, 7, 8]
    density_stack = []
    xhi_stack = []
    params_stack = []
    for cid in row_cone_ids:
        dens, xhi = _make_cube(nx, ny, n_z, cid)
        density_stack.append(dens)
        xhi_stack.append(xhi)
        params_stack.append(np.zeros(len(PARAM_NAMES), dtype=np.float32))

    cache_path = tmp_path / "cache.h5"
    _write_cache(
        cache_path,
        np.stack(density_stack),
        np.stack(xhi_stack),
        target_z,
        row_cone_ids,
        np.stack(params_stack),
    )
    dataset = LightconeCubeCache(cache_path, input_features="density_z")

    train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx) = split_cubes(
        dataset, val_frac=0.2, test_frac=0.2, seed=123,
    )

    # Reconstruct physical cone ids selected for each split.
    train_cids = [int(dataset.cone_ids[i]) for i in train_idx]
    val_cids = [int(dataset.cone_ids[i]) for i in val_idx]
    test_cids = [int(dataset.cone_ids[i]) for i in test_idx]

    # Splits should be disjoint and cover the full cone id range.
    assert sorted(train_cids + val_cids + test_cids) == list(range(n_cones))
    assert len(set(train_cids) & set(val_cids)) == 0
    assert len(set(train_cids) & set(test_cids)) == 0
    assert len(set(val_cids) & set(test_cids)) == 0

    # Spot-check that a held-out cone id really lands in the expected subset.
    # make_file_split returns sorted cone ids within each split.
    assert max(val_cids) < n_cones


def test_raw_and_cache_splits_match(tmp_path):
    """Raw files and a permuted cache of the same cones select identical splits."""
    n_cones = 20
    nx = ny = 2
    n_z = 4
    target_z = np.linspace(5.0, 8.0, n_z, dtype=np.float32)
    params = np.zeros(len(PARAM_NAMES), dtype=np.float32)

    raw_paths = []
    density_stack = []
    xhi_stack = []
    params_stack = []
    for cid in range(n_cones):
        dens, xhi = _make_cube(nx, ny, n_z, cid)
        path = tmp_path / f"21cmfast_11d_sample_{cid:04d}.h5"
        _write_raw(path, dens, xhi, target_z, params)
        raw_paths.append(path)
        density_stack.append(dens)
        xhi_stack.append(xhi)
        params_stack.append(params)

    raw_dataset = LightconeCubeDataset(
        raw_paths, n_z=n_z, z_min=5.0, z_max=8.0, input_features="density_z",
    )

    # Cache rows are a random permutation of the canonical cone id order.
    perm = np.random.default_rng(7).permutation(n_cones)
    cache_path = tmp_path / "cache.h5"
    _write_cache(
        cache_path,
        np.stack(density_stack)[perm],
        np.stack(xhi_stack)[perm],
        target_z,
        perm.tolist(),
        np.stack(params_stack)[perm],
    )
    cache_dataset = LightconeCubeCache(cache_path, input_features="density_z")

    raw_train, raw_val, raw_test, _ = split_cubes(
        raw_dataset, val_frac=0.2, test_frac=0.2, seed=42,
    )
    cache_train, cache_val, cache_test, _ = split_cubes(
        cache_dataset, val_frac=0.2, test_frac=0.2, seed=42,
    )

    raw_train_cids = {int(raw_dataset.cone_ids[i]) for i in raw_train.indices}
    raw_val_cids = {int(raw_dataset.cone_ids[i]) for i in raw_val.indices}
    raw_test_cids = {int(raw_dataset.cone_ids[i]) for i in raw_test.indices}

    cache_train_cids = {int(cache_dataset.cone_ids[i]) for i in cache_train.indices}
    cache_val_cids = {int(cache_dataset.cone_ids[i]) for i in cache_val.indices}
    cache_test_cids = {int(cache_dataset.cone_ids[i]) for i in cache_test.indices}

    assert raw_train_cids == cache_train_cids
    assert raw_val_cids == cache_val_cids
    assert raw_test_cids == cache_test_cids
