from __future__ import annotations

import h5py
import numpy as np

from dataset_3d import (
    InputFeatures,
    LightconeCubeCache,
    LightconeCubeDataset,
    ParameterNormalization,
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


def _write_cache(path, density, xhi, target_z, params):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=density[None])
        handle.create_dataset("neutral_fraction", data=xhi[None])
        handle.create_dataset("target_z", data=target_z)
        handle.create_dataset("cone_id", data=np.array([0]))
        handle.create_dataset("params", data=params[None])


def test_parameter_normalization_uses_only_train_indices():
    params = np.vstack([
        np.zeros(len(PARAM_NAMES)),
        np.full(len(PARAM_NAMES), 2.0),
        np.full(len(PARAM_NAMES), 1000.0),
    ])
    normalization = ParameterNormalization.fit(params, [0, 1])
    assert np.allclose(normalization.mean, 1.0)
    assert np.allclose(normalization.std, 1.0)
    assert np.allclose(normalization.normalize(params[0]), -1.0)


def test_raw_and_cache_channel_contracts_are_equivalent(tmp_path):
    target_z = np.linspace(5.0, 8.0, 4, dtype=np.float32)
    density = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    xhi = np.linspace(0, 1, density.size, dtype=np.float32).reshape(density.shape)
    params = np.arange(len(PARAM_NAMES), dtype=np.float32)
    raw_path = tmp_path / "raw.h5"
    cache_path = tmp_path / "cache.h5"
    _write_raw(raw_path, density, xhi, target_z, params)
    _write_cache(cache_path, density, xhi, target_z, params)

    features = InputFeatures("density_z_params")
    normalization = ParameterNormalization(
        names=PARAM_NAMES,
        mean=tuple(float(value - 1) for value in params),
        std=tuple(2.0 for _ in params),
    )
    raw = LightconeCubeDataset(
        [raw_path],
        n_z=4,
        z_min=5.0,
        z_max=8.0,
        input_features=features,
        parameter_normalization=normalization,
    )
    cache = LightconeCubeCache(
        cache_path,
        input_features=features,
        parameter_normalization=normalization,
    )

    assert raw.input_features.channel_names == cache.input_features.channel_names
    assert raw.in_channels == cache.in_channels == 13
    assert np.allclose(raw[0]["x"].numpy(), cache[0]["x"].numpy())
    assert np.allclose(raw[0]["y"].numpy(), cache[0]["y"].numpy())


def test_all_conditioning_ablations_have_declared_channel_counts():
    expected = {
        "density": 1,
        "params": len(PARAM_NAMES),
        "density_z": 2,
        "density_z_params": 2 + len(PARAM_NAMES),
    }
    for name, count in expected.items():
        assert len(InputFeatures(name).channel_names) == count
