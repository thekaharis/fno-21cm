from __future__ import annotations

import h5py
import numpy as np

from dataset.build_slices import xHI_profile


def test_global_profile_is_interpolated_to_los_grid(tmp_path) -> None:
    path = tmp_path / "cone.h5"
    with h5py.File(path, "w") as handle:
        handle["lightcone/lightcone_redshifts"] = np.array([5.0, 6.0, 7.0])
        handle["lightcone/node_redshifts"] = np.array([7.0, 5.0])
        handle["lightcone/global_quantities/neutral_fraction"] = np.array(
            [0.9, 0.1], dtype=np.float32
        )
    with h5py.File(path, "r") as handle:
        profile = xHI_profile(handle)

    np.testing.assert_allclose(profile, [0.1, 0.5, 0.9])
