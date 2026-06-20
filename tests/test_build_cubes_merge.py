from __future__ import annotations

import h5py
import numpy as np
import pytest

from dataset.build_cubes import merge
from dataset.lightcone_params import PARAM_NAMES


def _write_shard(path, cone_ids, nx=2, ny=2, n_z=4):
    """Minimal shard file matching the layout produced by build()."""
    cone_ids = np.asarray(cone_ids, dtype=np.int32)
    n = len(cone_ids)
    density = np.empty((n, nx, ny, n_z), dtype=np.float32)
    xhi = np.empty_like(density)
    for row, cid in enumerate(cone_ids):
        density[row] = float(cid)
        xhi[row] = float(cid) / 10.0
    params = np.tile(
        cone_ids.astype(np.float32)[:, None], (1, len(PARAM_NAMES))
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=density, chunks=(1, nx, ny, n_z))
        f.create_dataset("neutral_fraction", data=xhi,
                         chunks=(1, nx, ny, n_z))
        f.create_dataset("cone_id", data=cone_ids)
        f.create_dataset("params", data=params)
        f.create_dataset("target_z",
                         data=np.linspace(5.0, 8.0, n_z, dtype=np.float32))
        f.attrs["z_min"] = 5.0
        f.attrs["z_max"] = 8.0


def test_merge_sorts_rows_by_cone_id(tmp_path):
    out = tmp_path / "cubes_3d.h5"
    # Interleaved shard layout produced by ``build --num-shards 2``
    # (shard s holds files [s, s + 2, s + 4, ...]).
    _write_shard(tmp_path / "cubes_3d.shard000.h5", [0, 2, 4])
    _write_shard(tmp_path / "cubes_3d.shard001.h5", [1, 3])

    merge(out, num_shards=2, compress=False)

    with h5py.File(out, "r") as f:
        cone_ids = f["cone_id"][:]
        density = f["density"][:]
        xhi = f["neutral_fraction"][:]
        params = f["params"][:]

    assert cone_ids.tolist() == [0, 1, 2, 3, 4]
    for row, cid in enumerate(cone_ids):
        assert np.all(density[row] == float(cid))
        assert np.allclose(xhi[row], float(cid) / 10.0)
        assert np.all(params[row] == float(cid))


def test_merge_rejects_duplicate_cone_ids(tmp_path):
    out = tmp_path / "cubes_3d.h5"
    _write_shard(tmp_path / "cubes_3d.shard000.h5", [0, 1])
    _write_shard(tmp_path / "cubes_3d.shard001.h5", [1, 2])

    with pytest.raises(ValueError, match="not a complete permutation"):
        merge(out, num_shards=2, compress=False)


def test_merge_rejects_missing_cone_ids(tmp_path):
    out = tmp_path / "cubes_3d.h5"
    _write_shard(tmp_path / "cubes_3d.shard000.h5", [0, 2])
    _write_shard(tmp_path / "cubes_3d.shard001.h5", [3, 4])

    with pytest.raises(ValueError, match="not a complete permutation"):
        merge(out, num_shards=2, compress=False)


def test_merge_row_to_cone_mapping_invariant_to_shard_count(tmp_path):
    """A 1-shard build and a multi-shard merge yield identical row-to-cone maps."""
    n_cones = 7
    nx = ny = 2
    n_z = 4

    # Simulate the output of ``build --num-shards 1`` writing directly to out.
    one_shard_out = tmp_path / "merged_1.h5"
    _write_shard(one_shard_out, list(range(n_cones)), nx=nx, ny=ny, n_z=n_z)

    # Multi-shard merge: shard s holds files [s, s + num_shards, ...].
    multi_shard_out = tmp_path / "merged_3.h5"
    num_shards = 3
    for s in range(num_shards):
        cone_ids = list(range(s, n_cones, num_shards))
        _write_shard(
            tmp_path / f"merged_3.shard{s:03d}.h5",
            cone_ids, nx=nx, ny=ny, n_z=n_z,
        )
    merge(multi_shard_out, num_shards=num_shards, compress=False)

    with h5py.File(one_shard_out, "r") as f:
        ids_1 = f["cone_id"][:].tolist()
        dens_1 = f["density"][:]
    with h5py.File(multi_shard_out, "r") as f:
        ids_3 = f["cone_id"][:].tolist()
        dens_3 = f["density"][:]

    assert ids_1 == ids_3 == list(range(n_cones))
    assert np.array_equal(dens_1, dens_3)
