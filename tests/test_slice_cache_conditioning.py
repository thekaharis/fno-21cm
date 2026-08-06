from __future__ import annotations

import h5py
import numpy as np
import pytest

from legacy.xhi2d.dataset import SliceCache, split_by_cone
from dataset.lightcone_params import PARAM_NAMES


def _write_cache(path) -> None:
    cones = np.repeat(np.arange(5), 2)
    with h5py.File(path, "w") as handle:
        handle["x"] = np.ones((10, 4, 4), dtype=np.float32) * 10
        handle["y"] = np.zeros((10, 4, 4), dtype=np.float32)
        handle["z"] = np.linspace(5, 10, 10, dtype=np.float32)
        handle["xHI_mean"] = np.zeros(10, dtype=np.float32)
        handle["cone_id"] = cones
        handle["params"] = np.repeat(
            np.arange(5, dtype=np.float32)[:, None], len(PARAM_NAMES), axis=1
        ).repeat(2, axis=0)
        handle.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype="S")
        handle.attrs["slice_cache_version"] = 2


def test_conditioned_slice_cache_and_cone_split(tmp_path) -> None:
    path = tmp_path / "slices.h5"
    _write_cache(path)
    cache = SliceCache(path, input_features="density_z_params")
    train, val, test = split_by_cone(cache, val_frac=0.2, test_frac=0.2)
    normalization = cache.fit_parameter_normalization(train.indices)

    sample = cache[train.indices[0]]

    assert sample["x"].shape == (13, 4, 4)
    assert np.allclose(sample["x"][0], 1.0)
    assert np.allclose(
        sample["x"][1], 1.0 / (1.0 + float(sample["z"]))
    )
    assert normalization.names == tuple(PARAM_NAMES)
    train_cones = set(cache.cone_id[train.indices])
    val_cones = set(cache.cone_id[val.indices])
    test_cones = set(cache.cone_id[test.indices])
    assert not train_cones & val_cones
    assert not train_cones & test_cones
    assert not val_cones & test_cones


def test_conditioned_slice_cache_requires_parameter_schema(tmp_path) -> None:
    path = tmp_path / "slices.h5"
    _write_cache(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["param_names"]

    with pytest.raises(ValueError, match="parameter names do not match schema"):
        SliceCache(path, input_features="density_z_params")


def test_slice_cache_rejects_nonfinite_mean_neutral_fraction(tmp_path) -> None:
    path = tmp_path / "slices.h5"
    _write_cache(path)
    with h5py.File(path, "r+") as handle:
        handle["xHI_mean"][0] = np.nan

    with pytest.raises(ValueError, match="non-finite mean neutral fractions"):
        SliceCache(path)
