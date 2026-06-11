#!/usr/bin/env python3
"""One-time pass: pre-interpolate every lightcone cube to a fixed z-grid.

Why this exists
---------------
The 3-D training pipeline streams raw lightcones at training time, but each
sample requires reading two ~183 MB float32 arrays from HDF5 plus a scipy
linear interpolation along the LOS axis -- roughly 0.5 s/sample even with 8
workers on a cluster project filesystem.  At 5280 cones/epoch that is ~55 min
per epoch, with the GPU sitting nearly idle.

This script reads every lightcone once, interpolates ``density`` and
``neutral_fraction`` from native ``n_los~2340`` cells down to ``n_z`` (default
256), and writes the cubes into a single compact HDF5 cache.  Each cube is
~40 MB (vs 370 MB raw), so per-sample I/O drops ~10x.  Combined with the 8
DataLoader workers already in place, epochs should drop from ~55 min to ~5-10
min on the same A30.

Run it ONCE.  Parallelize across files with a SLURM array, then merge:

    # serial (small)
    python build_cubes.py --data /path/to/lightcones --out cubes_3d.h5

    # parallel: array of N tasks each writing a shard, then one merge
    python build_cubes.py --data /path/to/lightcones --out cubes_3d.h5 \
        --shard "$SLURM_ARRAY_TASK_ID" --num-shards 33
    python build_cubes.py --out cubes_3d.h5 --merge --num-shards 33

Layout produced::

    cubes_3d.h5
      density           (N, Nx, Ny, n_z)  float32   per-cone chunks
      neutral_fraction  (N, Nx, Ny, n_z)  float32   per-cone chunks
      cone_id           (N,)              int32     global file index
      target_z          (n_z,)            float32
      params            (N, 11)           float32   gzip
      attrs: n_z, z_min, z_max, param_names

The cone_id matches the global ``sorted(...glob(...))`` ordering.  ``merge``
writes rows sorted by cone_id, so row index == cone_id whenever no source
file was skipped and positional splits agree between the raw-streaming and
cached pipelines.  Caches merged before this ordering existed have
shard-interleaved rows; downstream tools should map through the ``cone_id``
dataset rather than trust row positions (see dataset_3d.resolve_split).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from lightcone_params import PARAM_NAMES, read_sampled_params
from loader import LightconeFile

PARAMS = list(PARAM_NAMES)


# --------------------------------------------------------------- per-cone IO
def interp_one(path: Path, target_z: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read density + x_HI from one lightcone, interpolate to target_z grid."""
    with LightconeFile(path) as lf:
        dens = lf.read_interpolated("density", target_z)
        xhi = lf.read_interpolated("neutral_fraction", target_z)
    with h5py.File(path, "r") as f:
        params = read_sampled_params(f)
    return dens, xhi, params


# ------------------------------------------------------------------ writer
def _shard_path(out: Path, shard: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return out
    return out.with_suffix(f".shard{shard:03d}.h5")


def _create_cube_dataset(o: h5py.File, name: str,
                         shape: tuple[int, int, int, int],
                         compress: bool) -> h5py.Dataset:
    """Create a (N, Nx, Ny, Nz) float32 dataset chunked one cone per chunk.

    Compression on neutral_fraction is a big win (~5x: large 0/1 regions) and
    a small win on density (~1.2x). Compressed reads add ~150 ms per cube vs
    ~50 ms uncompressed -- with 8 DataLoader workers that's still GPU-bound,
    so default to compression on both.
    """
    kwargs = dict(chunks=(1, shape[1], shape[2], shape[3]), dtype="float32")
    if compress:
        kwargs.update(compression="gzip", compression_opts=4)
    return o.create_dataset(name, shape=shape, **kwargs)


# ------------------------------------------------------------------- build
def build(data_dir: Path, out: Path, n_z: int, z_min: float, z_max: float,
          shard: int, num_shards: int, compress: bool) -> None:
    files = sorted(Path(data_dir).glob("21cmfast_11d_sample*.h5"))
    if not files:
        sys.exit(f"No lightcone files found in {data_dir}")

    # cone_id = global file index, preserved across shards
    todo = list(enumerate(files))
    if num_shards > 1:
        todo = todo[shard::num_shards]
    print(f"[build] {len(todo)}/{len(files)} cones (shard {shard}/{num_shards})"
          f", n_z={n_z}, z in [{z_min}, {z_max}], compress={compress}",
          flush=True)

    target_z = np.linspace(float(z_min), float(z_max), int(n_z),
                           dtype=np.float64)

    # Determine cube shape from the first file
    with LightconeFile(files[0]) as lf:
        nx = ny = lf.meta.transverse

    n_local = len(todo)
    out_path = _shard_path(out, shard, num_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {out_path}  shape=({n_local}, {nx}, {ny}, {n_z}) per field",
          flush=True)

    shape = (n_local, nx, ny, n_z)
    with h5py.File(out_path, "w") as o:
        dset_d = _create_cube_dataset(o, "density", shape, compress)
        dset_x = _create_cube_dataset(o, "neutral_fraction", shape, compress)
        cone_ids = np.empty(n_local, dtype=np.int32)
        params_arr = np.empty((n_local, len(PARAMS)), dtype=np.float32)

        n_skipped = 0
        for j, (cone_id, path) in enumerate(todo):
            try:
                dens, xhi, params = interp_one(path, target_z)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [skip] {path.name}: {exc}", file=sys.stderr,
                      flush=True)
                dens = np.full((nx, ny, n_z), np.nan, dtype=np.float32)
                xhi = np.full((nx, ny, n_z), np.nan, dtype=np.float32)
                params = np.full(len(PARAMS), np.nan, dtype=np.float32)
                n_skipped += 1
            dset_d[j] = dens
            dset_x[j] = xhi
            cone_ids[j] = cone_id
            params_arr[j] = params
            if (j + 1) % 50 == 0:
                print(f"  ... {j + 1}/{n_local} cones", flush=True)

        o.create_dataset("cone_id", data=cone_ids)
        o.create_dataset("target_z", data=target_z.astype(np.float32))
        o.create_dataset("params", data=params_arr,
                         compression="gzip", compression_opts=4)
        o.attrs["n_z"] = int(n_z)
        o.attrs["z_min"] = float(z_min)
        o.attrs["z_max"] = float(z_max)
        o.attrs["param_names"] = np.array(PARAMS, dtype="S")

    print(f"[done] {out_path}  ({n_local} cones, {n_skipped} skipped)",
          flush=True)


# --------------------------------------------------------- direct chunk copy
def _copy_chunks_direct(src_dset: h5py.Dataset, dst_dset: h5py.Dataset,
                        dst_rows: np.ndarray) -> int:
    """Copy per-cone chunks from ``src_dset`` to ``dst_dset`` byte-for-byte.

    The chunks are *(1, Nx, Ny, Nz)*, so chunk *j* of the source maps to chunk
    *dst_rows[j]* of the destination.  We use h5py's low-level direct chunk
    API (``read_direct_chunk`` / ``write_direct_chunk``), which moves the
    compressed bytes through unchanged -- no gzip decode + re-encode, no numpy
    round-trip.  On a typical ~40 MB gzip-4 cube this is ~5-10x faster than the
    classic ``dst[row] = src[j]`` pattern and is the difference between
    fitting the full 33-shard merge in walltime or not.

    Falls back to the classic copy for any chunk that the direct API rejects
    (e.g. different chunk layouts or unsupported filters).

    Returns the number of chunks copied via the fast path; the remainder used
    the fallback.
    """
    n_fast = 0
    src_id = src_dset.id
    dst_id = dst_dset.id
    for j, dst_row in enumerate(dst_rows):
        try:
            filter_mask, chunk_bytes = src_id.read_direct_chunk((j, 0, 0, 0))
            dst_id.write_direct_chunk(
                (int(dst_row), 0, 0, 0),
                chunk_bytes,
                filter_mask=filter_mask,
            )
            n_fast += 1
        except Exception:                                 # noqa: BLE001
            # Fall back to decompress + recompress for this chunk.
            dst_dset[int(dst_row)] = src_dset[j]
    return n_fast


# -------------------------------------------------------------------- merge
def merge(out: Path, num_shards: int, compress: bool) -> None:
    shards = [_shard_path(out, i, num_shards) for i in range(num_shards)]
    missing = [s for s in shards if not s.exists()]
    if missing:
        sys.exit(f"Missing shards: {[s.name for s in missing]}")

    # First pass: gather per-shard counts, cube shape, and cone ids.  The
    # shards were built as ``files[shard::num_shards]``, so concatenating
    # them naively produces shard-interleaved rows ([0, 33, 66, ..., 1, 34,
    # ...]).  Sort destination rows by cone_id instead so row index ==
    # cone_id (when no file was skipped) and positional splits agree with
    # the raw-streaming pipeline.
    counts: list[int] = []
    nx = ny = n_z = None
    shard_cone_ids: list[np.ndarray] = []
    for s in shards:
        with h5py.File(s, "r") as f:
            shp = f["density"].shape
            counts.append(int(shp[0]))
            if nx is None:
                _, nx, ny, n_z = shp
            shard_cone_ids.append(np.asarray(f["cone_id"][:], dtype=np.int64))
    all_ids = np.concatenate(shard_cone_ids)
    n_total = int(all_ids.size)
    if np.unique(all_ids).size != n_total:
        print("[merge] WARNING: duplicate cone ids across shards; rows are "
              "still cone_id-sorted but the cache contains repeated cones",
              flush=True)
    # Destination row of each global source position = rank in cone_id order.
    order = np.argsort(all_ids, kind="stable")
    dst_of_src = np.empty(n_total, dtype=np.int64)
    dst_of_src[order] = np.arange(n_total)
    print(f"[merge] {num_shards} shards -> {n_total} cones "
          f"of shape ({nx}, {ny}, {n_z}), rows sorted by cone_id", flush=True)

    shape = (n_total, nx, ny, n_z)
    with h5py.File(out, "w") as o:
        dset_d = _create_cube_dataset(o, "density", shape, compress)
        dset_x = _create_cube_dataset(o, "neutral_fraction", shape, compress)
        cone_ids = np.empty(n_total, dtype=np.int32)
        params_arr = np.empty((n_total, len(PARAMS)), dtype=np.float32)
        target_z = None
        z_min = z_max = None

        # Fast path requires identical chunking + filters on source and
        # destination -- which is the case here since both go through
        # _create_cube_dataset with the same `compress` setting.
        n_fast_total = 0
        n_total_chunks = 0
        offset = 0
        for shard_path, count in zip(shards, counts):
            dst_rows = dst_of_src[offset:offset + count]
            with h5py.File(shard_path, "r") as f:
                n_fast_total += _copy_chunks_direct(
                    f["density"], dset_d, dst_rows)
                n_fast_total += _copy_chunks_direct(
                    f["neutral_fraction"], dset_x, dst_rows)
                n_total_chunks += 2 * count
                cone_ids[dst_rows] = f["cone_id"][:]
                params_arr[dst_rows] = f["params"][:]
                if target_z is None:
                    target_z = f["target_z"][:]
                    z_min = float(f.attrs["z_min"])
                    z_max = float(f.attrs["z_max"])
            print(f"  ... merged {shard_path.name} ({count} cones)",
                  flush=True)
            offset += count

        print(f"[merge] direct-chunk copies: {n_fast_total}/{n_total_chunks} "
              f"({100*n_fast_total/max(n_total_chunks,1):.1f}%)", flush=True)

        o.create_dataset("cone_id", data=cone_ids)
        o.create_dataset("target_z", data=target_z)
        o.create_dataset("params", data=params_arr,
                         compression="gzip", compression_opts=4)
        o.attrs["n_z"] = int(n_z)
        o.attrs["z_min"] = z_min
        o.attrs["z_max"] = z_max
        o.attrs["param_names"] = np.array(PARAMS, dtype="S")

    print(f"[done] merged -> {out}", flush=True)


# ----------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path,
                    help="directory of lightcone .h5 files")
    ap.add_argument("--out", type=Path, default=Path("cubes_3d.h5"))
    ap.add_argument("--n-z", type=int, default=256,
                    help="LOS resolution after interpolation (default 256)")
    ap.add_argument("--z-min", type=float, default=5.0)
    ap.add_argument("--z-max", type=float, default=25.0)
    ap.add_argument("--no-compress", action="store_true",
                    help="skip gzip on cube datasets (2-3x larger on disk, "
                         "but faster reads at training time)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="combine shard files into the final cache")
    args = ap.parse_args()

    compress = not args.no_compress
    if args.merge:
        merge(args.out, args.num_shards, compress)
        return
    if args.data is None:
        ap.error("--data is required unless --merge is given")
    build(args.data, args.out, args.n_z, args.z_min, args.z_max,
          args.shard, args.num_shards, compress)


if __name__ == "__main__":
    main()
