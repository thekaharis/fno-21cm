"""Rewrite raw lightcones into a window-friendly native mirror.

Training and validation on native LOS windows are I/O-bound, not compute-bound:
the raw fields are gzip-compressed in 9x9x307 chunks, so reading one 256-cell
window decompresses hundreds of small chunks spread over the whole transverse
plane and costs about as much as reading the entire field.

This writes the same HDF5 layout (so NativeLightconeDataset reads it with no
code change) keeping only the fields a mapping needs, chunked contiguously
along the line of sight and uncompressed. Measured on one cone: a two-field
256-cell window read drops from 270 ms to 13.8 ms.

Storage is the price: ~0.73 GiB per cone for four fields, versus a compressed
raw file holding every field. float32 throughout -- los_velocity is ~1e-16 in
Mpc/s and would underflow float16.

  python tools_build_native_mirror.py --shard 0 --num-shards 16 --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

RAW = Path("/pfs/10/work/hd_id260-fno_training/data/data")
FIELDS = ("density", "los_velocity", "neutral_fraction", "brightness_temp")
AXES = ("lightcone_redshifts", "lightcone_distances")
CHUNK_LOS = 64


def build_one(sample_id, out_dir, fields, overwrite=False):
    name = f"21cmfast_11d_sample{sample_id:06d}.h5"
    src, dst = RAW / name, out_dir / name
    if dst.exists() and not overwrite:
        return "exists", 0
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(src, "r") as f, h5py.File(tmp, "w") as g:
        group = f["lightcone"]
        out = g.create_group("lightcone")
        for field in fields:
            value = np.asarray(group[field], dtype=np.float32)
            if not np.isfinite(value).all():
                tmp.unlink(missing_ok=True)
                return "nonfinite", 0
            nz = value.shape[2]
            out.create_dataset(field, data=value,
                               chunks=(value.shape[0], value.shape[1], min(CHUNK_LOS, nz)))
        for axis in AXES:
            out.create_dataset(axis, data=np.asarray(group[axis]))
        f.copy("params", g)                      # sampled parameters, verbatim
        for key, value in f.attrs.items():       # provenance (sample_id, box, seeds)
            g.attrs[key] = value
        g.attrs["mirror_fields"] = list(fields)
        g.attrs["mirror_chunk_los"] = CHUNK_LOS
        g.attrs["mirror_source"] = str(src)
    tmp.replace(dst)
    return "written", dst.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cones", type=Path,
                    help="JSON list of sample ids; default 0..1999 minus the exclusions")
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.cones:
        ids = json.loads(args.cones.read_text())
    else:
        exclude = {72, 168, 385, 428, 457, 1150, 1589, 1794}   # non-finite brightness_temp
        ids = [i for i in range(2000) if i not in exclude]
    mine = ids[args.shard::args.num_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    tally, total = {}, 0
    for sample_id in mine:
        status, size = build_one(sample_id, args.out, tuple(args.fields.split(",")),
                                 args.overwrite)
        tally[status] = tally.get(status, 0) + 1
        total += size
        if status == "nonfinite":
            print(f"  skipped {sample_id}: non-finite values", flush=True)
    print(f"shard {args.shard}: {len(mine)} cones -> {tally}, {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
