"""Survey ionization-front sharpness across the raw 21cmFAST lightcones.

Measures, over every LOS position of every lightcone, how much of the field is
partially ionized and how sharp the ionization fronts are, accumulated against
the slice's own mean neutral fraction.

Two things dictate the implementation.

*Read the raw lightcones, not cubes_3d.h5.* The cache interpolates 2340 LOS
steps down to 256 -- a 9x resample along the line of sight -- so any sharpness
measured on it is the cache's, not the simulation's.

*Read chunk-aligned z-slabs.* The HDF5 chunking is (9, 9, 293), so a single
transverse plane touches ~24 MB of chunks to return 78 KB and costs ~430 ms.
A whole 293-plane slab costs the same read and yields every plane in it, so
this walks slabs and vectorises the statistics over the slab's z axis. Every
LOS position is therefore surveyed, not a subsample.

Results accumulate into x_HI bins rather than per-slice rows: 6600 cones x 2340
planes is 15 million slices, and only the binned aggregates are wanted.

    python -m legacy.xhi2d.survey_truth_sharpness --shard 0 --num-shards 33
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from dataset import paths

# x_HI bins the statistics accumulate against (the slice's own mean).
N_XHI_BINS = 50
# Value histogram, for "what fraction of voxels is partially ionized".
N_VALUE_BINS = 200
# Partial-ionization definitions, from strict to the loose one used elsewhere
# in this project (blur_frac).
PARTIAL_BANDS = ((1e-3, 1 - 1e-3), (0.01, 0.99), (0.1, 0.9))

FIELD = "lightcone/neutral_fraction"        # matches build_cubes.py


def accumulators() -> dict:
    return {
        "n_planes": np.zeros(N_XHI_BINS, dtype=np.int64),
        "sum_partial": np.zeros((len(PARTIAL_BANDS), N_XHI_BINS)),
        "sum_tv_xy": np.zeros(N_XHI_BINS),      # transverse total variation
        "sum_tv_z": np.zeros(N_XHI_BINS),       # line-of-sight
        "sum_band": np.zeros(N_XHI_BINS),       # pixels in 0.1..0.9
        "sum_width": np.zeros(N_XHI_BINS),      # per-plane band/TV, averaged
        "sum_peak": np.zeros(N_XHI_BINS),       # p99.9 of |grad_xy|
        "value_hist": np.zeros(N_VALUE_BINS, dtype=np.int64),
        "n_cones": 0, "n_planes_total": 0,
    }


def survey_cone(path: Path, acc: dict) -> None:
    with h5py.File(path, "r") as f:
        xh = f[FIELD]
        nx, ny, nz = xh.shape
        czs = xh.chunks[2] if xh.chunks else 256
        for z0 in range(0, nz, czs):
            z1 = min(z0 + czs, nz)
            blk = np.asarray(xh[:, :, z0:z1], dtype=np.float32)   # (nx,ny,nzs)
            _accumulate(blk, acc)
    acc["n_cones"] += 1


def _accumulate(blk: np.ndarray, acc: dict) -> None:
    """Per-plane statistics for one z-slab, vectorised over its z axis."""
    npl = blk.shape[2]
    area = blk.shape[0] * blk.shape[1]
    mean_xhi = blk.mean(axis=(0, 1))
    idx = np.clip((mean_xhi * N_XHI_BINS).astype(np.int64), 0, N_XHI_BINS - 1)

    # transverse gradient, periodic like the rest of the project's metrics
    gx = np.roll(blk, -1, axis=0) - blk
    gy = np.roll(blk, -1, axis=1) - blk
    gmag = np.sqrt(gx * gx + gy * gy)
    tv_xy = gmag.sum(axis=(0, 1))
    peak = np.percentile(gmag.reshape(-1, npl), 99.9, axis=0)
    # line-of-sight gradient: interior only, so slab edges are not counted
    gz = np.abs(np.diff(blk, axis=2))
    tv_z = np.zeros(npl)
    tv_z[:-1] = gz.sum(axis=(0, 1))

    band = ((blk > 0.1) & (blk < 0.9)).sum(axis=(0, 1)).astype(np.float64)
    width = band / np.maximum(tv_xy, 1e-9)

    np.add.at(acc["n_planes"], idx, 1)
    np.add.at(acc["sum_tv_xy"], idx, tv_xy / area)
    np.add.at(acc["sum_tv_z"], idx, tv_z / area)
    np.add.at(acc["sum_band"], idx, band / area)
    np.add.at(acc["sum_width"], idx, width)
    np.add.at(acc["sum_peak"], idx, peak)
    for b, (lo, hi) in enumerate(PARTIAL_BANDS):
        frac = ((blk > lo) & (blk < hi)).sum(axis=(0, 1)) / area
        np.add.at(acc["sum_partial"][b], idx, frac)

    acc["value_hist"] += np.histogram(blk, bins=N_VALUE_BINS, range=(0.0, 1.0))[0]
    acc["n_planes_total"] += npl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.LIGHTCONES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=33)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures/shared/diagnostics/truth_survey"))
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("21cmfast_11d_sample*.h5"))
    if not files:
        sys.exit(f"no lightcones in {args.data}")
    mine = files[args.shard::args.num_shards]
    if args.max_cones:
        mine = mine[: args.max_cones]
    print(f"[survey] shard {args.shard}/{args.num_shards}: {len(mine)} cones "
          f"of {len(files)}", flush=True)

    acc = accumulators()
    for i, p in enumerate(mine):
        try:
            survey_cone(p, acc)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [skip] {p.name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(mine)} cones, "
                  f"{acc['n_planes_total']:,} planes", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"shard{args.shard:03d}.npz"
    np.savez_compressed(dest, n_xhi_bins=N_XHI_BINS, n_value_bins=N_VALUE_BINS,
                        partial_bands=np.array(PARTIAL_BANDS), **acc)
    print(f"[survey] wrote {dest}  ({acc['n_cones']} cones, "
          f"{acc['n_planes_total']:,} planes)")


if __name__ == "__main__":
    main()
