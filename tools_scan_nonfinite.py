"""Scan raw lightcones for non-finite voxels, one shard per array task.

The multi-field reader refuses any cube containing non-finite values, so the
affected simulations must be identified before the cache build rather than
discovered 10 minutes into it. Reads full resolution -- a subsampled scan cannot
see a 9-voxel defect -- in LOS slabs to bound memory.

  python tools_scan_nonfinite.py --shard K --num-shards N --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np

FIELDS = ("density", "neutral_fraction", "brightness_temp", "los_velocity")
RAW = "/pfs/10/work/hd_id260-fno_training/data/data"


def scan(path, slab=256):
    out = {}
    with h5py.File(path, "r") as h:
        for f in FIELDS:
            d = h[f"lightcone/{f}"]
            n = 0
            for s in range(0, d.shape[2], slab):
                a = np.asarray(d[:, :, s:s + slab], dtype=np.float32)
                n += int((~np.isfinite(a)).sum())
            out[f] = n
        out["n_z"] = int(d.shape[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{RAW}/21cmfast_11d_sample*.h5"))[a.shard::a.num_shards]
    rows = []
    for p in files:
        rec = {"sample_id": int(p[-9:-3])}
        rec.update(scan(p))
        rows.append(rec)
        if rec["brightness_temp"] or rec["density"] or rec["neutral_fraction"] or rec["los_velocity"]:
            print(f"  BAD {rec['sample_id']}: "
                  + " ".join(f"{f}={rec[f]}" for f in FIELDS if rec[f]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"shard{a.shard:03d}.json").write_text(json.dumps(rows))
    print(f"shard {a.shard}: scanned {len(rows)} cones")


if __name__ == "__main__":
    main()
