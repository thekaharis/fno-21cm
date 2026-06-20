#!/usr/bin/env python3
"""One-time pass: extract a few slices per lightcone into a compact cache.

Why this exists
---------------
The full dataset is ~6600 lightcones x ~1.3 GB = ~8.6 TB, far too large to
preload or to re-read every epoch.  Overfitting in earlier runs came from too
few *independent* cones (each contributing 256 highly-correlated z-slices), so
the strategy here is the opposite: **few slices per cone, many cones**.

This script reads every lightcone once, selects ``K_PER_CONE`` 2-D slices per
cone (biased toward the partially-ionized reionization window so we don't drown
in trivial all-neutral / all-ionized fields), and writes them - together with
redshift, mean x_HI, a global cone id, and the 11 sampled parameters - into a
single compact HDF5 cache that fits in RAM for training.

Run it ONCE.  Parallelize across files with a SLURM array, then merge:

    # serial
    python build_trainset.py --data /path/to/lightcones --out trainset.h5

    # parallel: array of N tasks each writing a shard, then one merge
    python build_trainset.py --data /path/to/lightcones --out trainset.h5 \
        --shard "$SLURM_ARRAY_TASK_ID" --num-shards 33
    python build_trainset.py --out trainset.h5 --merge --num-shards 33

The selection RNG is seeded per-cone, so the output is identical regardless of
how the work is sharded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from lightcone_params import PARAM_NAMES, read_sampled_params

PARAMS = list(PARAM_NAMES)


# ----------------------------------------------------------------- selection
def xHI_profile(f: h5py.File) -> np.ndarray:
    """Per-LOS mean neutral fraction, as cheaply as possible.

    Prefers a stored global-quantity profile (no cube read); otherwise falls
    back to a strided transverse mean of the neutral_fraction cube (~1/16 IO).
    """
    g = f.get("lightcone/global_quantities")
    if g is not None and "neutral_fraction" in g:
        return np.asarray(g["neutral_fraction"], dtype=np.float32)
    xH = f["lightcone/neutral_fraction"]                 # (140, 140, n_los)
    return xH[::4, ::4, :].mean(axis=(0, 1)).astype(np.float32)


def select_indices(prof: np.ndarray, k: int, rng: np.random.Generator,
                   lo: float, hi: float, floor: float = 0.02) -> np.ndarray:
    """Pick *k* LOS indices for one cone, weighted toward ``lo < x_HI < hi``.

    Slices inside the reionization window get weight 1; trivial fully-neutral /
    fully-ionized slices keep a small ``floor`` weight so the model still sees a
    few of them.  Cones that never reionize fall back to uniform sampling.
    """
    n = prof.shape[0]
    w = np.where((prof > lo) & (prof < hi), 1.0, floor)
    s = float(w.sum())
    if s <= 0:
        w = np.ones(n, dtype=np.float64)
        s = float(n)
    w = w / s
    k = min(k, n)
    return np.sort(rng.choice(n, size=k, replace=False, p=w))


def extract_one(path: Path, cone_id: int, k: int,
                rng: np.random.Generator, lo: float, hi: float):
    """Return (x, y, z, xHI_mean, cone_id[], params[]) for one cone."""
    with h5py.File(path, "r") as f:
        prof = xHI_profile(f)
        idx = select_indices(prof, k, rng, lo, hi)
        dens = f["lightcone/density"]
        xH = f["lightcone/neutral_fraction"]
        z_all = np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float32)
        # read ONLY the selected 2-D slices (each ~78 KB)
        x = np.stack([np.asarray(dens[:, :, i], dtype=np.float32) for i in idx])
        y = np.stack([np.asarray(xH[:, :, i], dtype=np.float32) for i in idx])
        params = read_sampled_params(f)
    n = len(idx)
    return (x, y, z_all[idx], prof[idx],
            np.full(n, cone_id, dtype=np.int32),
            np.tile(params, (n, 1)))


# -------------------------------------------------------------------- writer
def _shard_path(out: Path, shard: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return out
    return out.with_suffix(f".shard{shard:03d}.h5")


def write_cache(path: Path, data: dict, k: int, lo: float, hi: float) -> None:
    with h5py.File(path, "w") as o:
        for name, arr in data.items():
            o.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        o.attrs["k_per_cone"] = k
        o.attrs["xHI_window"] = (lo, hi)
        o.attrs["param_names"] = np.array(PARAMS, dtype="S")
    print(f"[write] {path}  ({len(data['x'])} slices)")


# --------------------------------------------------------------------- build
def build(data_dir: Path, out: Path, k: int, lo: float, hi: float,
          seed: int, shard: int, num_shards: int) -> None:
    files = sorted(Path(data_dir).glob("21cmfast_11d_sample*.h5"))
    if not files:
        sys.exit(f"No lightcone files found in {data_dir}")

    todo = list(enumerate(files))                         # (global cone id, path)
    if num_shards > 1:
        todo = todo[shard::num_shards]
    print(f"[build] {len(todo)}/{len(files)} cones "
          f"(shard {shard}/{num_shards}), k={k}, window=({lo},{hi})")

    buf: dict[str, list] = {n: [] for n in
                            ("x", "y", "z", "xHI_mean", "cone_id", "params")}
    for j, (cone_id, path) in enumerate(todo):
        rng = np.random.default_rng([seed, cone_id])      # per-cone => shard-invariant
        try:
            x, y, z, m, cone, par = extract_one(path, cone_id, k, rng, lo, hi)
        except Exception as exc:                          # noqa: BLE001
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
            continue
        buf["x"].append(x); buf["y"].append(y); buf["z"].append(z)
        buf["xHI_mean"].append(m); buf["cone_id"].append(cone); buf["params"].append(par)
        if (j + 1) % 100 == 0:
            print(f"  ... {j + 1}/{len(todo)} cones", flush=True)

    data = {n: np.concatenate(v) for n, v in buf.items()}
    write_cache(_shard_path(out, shard, num_shards), data, k, lo, hi)


# --------------------------------------------------------------------- merge
def merge(out: Path, num_shards: int) -> None:
    shards = [out.with_suffix(f".shard{i:03d}.h5") for i in range(num_shards)]
    missing = [s for s in shards if not s.exists()]
    if missing:
        sys.exit(f"Missing shards: {[s.name for s in missing]}")

    buf: dict[str, list] = {}
    k = lo = hi = None
    for s in shards:
        with h5py.File(s, "r") as f:
            for name in ("x", "y", "z", "xHI_mean", "cone_id", "params"):
                buf.setdefault(name, []).append(f[name][:])
            k = int(f.attrs["k_per_cone"]); lo, hi = map(float, f.attrs["xHI_window"])
    data = {n: np.concatenate(v) for n, v in buf.items()}
    # sort by cone id for tidy, reproducible ordering
    order = np.argsort(data["cone_id"], kind="stable")
    data = {n: arr[order] for n, arr in data.items()}
    write_cache(out, data, k, lo, hi)
    print(f"[merge] combined {num_shards} shards -> {out}")


# ----------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="directory of lightcone .h5 files")
    ap.add_argument("--out", type=Path, default=Path("trainset.h5"))
    ap.add_argument("--k", type=int, default=6, help="slices per cone")
    ap.add_argument("--lo", type=float, default=0.05, help="reionization window low")
    ap.add_argument("--hi", type=float, default=0.95, help="reionization window high")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="combine shard files")
    args = ap.parse_args()

    if args.merge:
        merge(args.out, args.num_shards)
        return
    if args.data is None:
        ap.error("--data is required unless --merge is given")
    build(args.data, args.out, args.k, args.lo, args.hi,
          args.seed, args.shard, args.num_shards)


if __name__ == "__main__":
    main()
