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
    python -m legacy.xhi2d.build_trainset --data /path/to/lightcones --out trainset.h5

    # parallel: array of N tasks each writing a shard, then one merge
    python -m legacy.xhi2d.build_trainset --data /path/to/lightcones --out trainset.h5 \
        --shard "$SLURM_ARRAY_TASK_ID" --num-shards 33
    python -m legacy.xhi2d.build_trainset --out trainset.h5 --merge --num-shards 33

The selection RNG is seeded per-cone, so the output is identical regardless of
how the work is sharded.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from dataset.lightcone_params import PARAM_NAMES, read_sampled_params
from dataset import paths

PARAMS = list(PARAM_NAMES)
SLICE_CACHE_VERSION = 2


# ----------------------------------------------------------------- selection
def xHI_profile(f: h5py.File) -> np.ndarray:
    """Per-LOS mean neutral fraction, as cheaply as possible.

    Prefers a stored global-quantity profile (no cube read); otherwise falls
    back to a strided transverse mean of the neutral_fraction cube (~1/16 IO).
    """
    g = f.get("lightcone/global_quantities")
    if g is not None and "neutral_fraction" in g:
        profile = np.asarray(g["neutral_fraction"], dtype=np.float32)
        los_z = np.asarray(
            f["lightcone/lightcone_redshifts"], dtype=np.float64
        )
        if profile.shape == los_z.shape:
            return profile
        if "lightcone/node_redshifts" not in f:
            raise ValueError(
                "global neutral-fraction profile does not match the LOS grid "
                "and node_redshifts is absent"
            )
        node_z = np.asarray(f["lightcone/node_redshifts"], dtype=np.float64)
        if profile.shape != node_z.shape:
            raise ValueError(
                "global neutral-fraction profile does not match node_redshifts"
            )
        order = np.argsort(node_z)
        return np.interp(
            los_z,
            node_z[order],
            profile[order],
            left=float(profile[order][0]),
            right=float(profile[order][-1]),
        ).astype(np.float32)
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
        if not np.isfinite(params).all():
            raise ValueError("sampled parameters contain missing/non-finite values")
    n = len(idx)
    return (x, y, z_all[idx], y.mean(axis=(1, 2), dtype=np.float64).astype(np.float32),
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
        o.attrs["slice_cache_version"] = SLICE_CACHE_VERSION
    print(f"[write] {path}  ({len(data['x'])} slices)")


# --------------------------------------------------------------------- build
def build(data_dir: Path, out: Path, k: int, lo: float, hi: float,
          seed: int, shard: int, num_shards: int,
          max_cones: int | None = None) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= lo < hi <= 1:
        raise ValueError("xHI window must satisfy 0 <= lo < hi <= 1")
    if num_shards <= 0 or not 0 <= shard < num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    if max_cones is not None and max_cones <= 0:
        raise ValueError("max_cones must be positive")

    files = sorted(Path(data_dir).glob("21cmfast_11d_sample*.h5"))
    if max_cones is not None:
        files = files[:max_cones]
    if not files:
        sys.exit(f"No lightcone files found in {data_dir}")

    todo = list(enumerate(files))                         # (global cone id, path)
    if num_shards > 1:
        todo = todo[shard::num_shards]
    if not todo:
        raise ValueError(
            f"shard {shard} has no input cones; use at most {len(files)} shards"
        )
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

    if not buf["x"]:
        raise RuntimeError(f"no valid cones were extracted for shard {shard}")
    data = {n: np.concatenate(v) for n, v in buf.items()}
    write_cache(_shard_path(out, shard, num_shards), data, k, lo, hi)


# --------------------------------------------------------------------- merge
def merge(out: Path, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = [out.with_suffix(f".shard{i:03d}.h5") for i in range(num_shards)]
    missing = [s for s in shards if not s.exists()]
    if missing:
        sys.exit(f"Missing shards: {[s.name for s in missing]}")

    names = ("x", "y", "z", "xHI_mean", "cone_id", "params")
    total = 0
    k = lo = hi = None
    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for shard_index, s in enumerate(shards):
        with h5py.File(s, "r") as f:
            version = int(f.attrs.get("slice_cache_version", 0))
            if version != SLICE_CACHE_VERSION:
                raise ValueError(
                    f"{s} has slice-cache version {version}; "
                    f"expected {SLICE_CACHE_VERSION}. Rebuild all shards."
                )
            shard_k = int(f.attrs["k_per_cone"])
            shard_lo, shard_hi = map(float, f.attrs["xHI_window"])
            if shard_index == 0:
                k, lo, hi = shard_k, shard_lo, shard_hi
                specs = {
                    name: (f[name].shape[1:], f[name].dtype) for name in names
                }
            elif (shard_k, shard_lo, shard_hi) != (k, lo, hi):
                raise ValueError(f"{s} has incompatible extraction settings")
            n = len(f["cone_id"])
            if any(len(f[name]) != n for name in names):
                raise ValueError(f"{s} contains arrays with different lengths")
            total += n

    if total == 0:
        raise ValueError("cannot merge slice-cache shards with no samples")

    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with ExitStack() as stack:
        inputs = [stack.enter_context(h5py.File(s, "r")) for s in shards]
        output = stack.enter_context(h5py.File(tmp, "w"))
        output_datasets = {
            name: output.create_dataset(
                name,
                shape=(total, *specs[name][0]),
                dtype=specs[name][1],
                # Writes occur one cone at a time. Matching the chunk's first
                # dimension avoids repeatedly recompressing partially-filled
                # auto-selected chunks during the streaming merge.
                chunks=(min(k, total), *specs[name][0]),
                compression="gzip",
                compression_opts=4,
            )
            for name in names
        }
        output.attrs["k_per_cone"] = k
        output.attrs["xHI_window"] = (lo, hi)
        output.attrs["param_names"] = np.array(PARAMS, dtype="S")
        output.attrs["slice_cache_version"] = SLICE_CACHE_VERSION

        cone_ids = [f["cone_id"][:] for f in inputs]
        positions = [0] * len(inputs)
        output_position = 0
        previous_cone = -1
        while output_position < total:
            active = [i for i, pos in enumerate(positions) if pos < len(cone_ids[i])]
            shard_index = min(active, key=lambda i: int(cone_ids[i][positions[i]]))
            ids = cone_ids[shard_index]
            start = positions[shard_index]
            cone_id = int(ids[start])
            end = start + 1
            while end < len(ids) and int(ids[end]) == cone_id:
                end += 1
            if cone_id <= previous_cone:
                raise ValueError(f"duplicate or unsorted cone ID {cone_id} in shards")
            count = end - start
            out_slice = slice(output_position, output_position + count)
            for name in names:
                output_datasets[name][out_slice] = inputs[shard_index][name][start:end]
            positions[shard_index] = end
            output_position += count
            previous_cone = cone_id

    tmp.replace(out)
    print(f"[merge] combined {num_shards} shards -> {out}")


# ----------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="directory of lightcone .h5 files")
    ap.add_argument("--out", type=Path, default=paths.TRAINSET)
    ap.add_argument("--k", type=int, default=6, help="slices per cone")
    ap.add_argument("--lo", type=float, default=0.05, help="reionization window low")
    ap.add_argument("--hi", type=float, default=0.95, help="reionization window high")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-cones", type=int, default=None,
                    help="limit input cones (useful for smoke tests)")
    ap.add_argument("--merge", action="store_true", help="combine shard files")
    args = ap.parse_args()

    if args.merge:
        merge(args.out, args.num_shards)
        return
    if args.data is None:
        ap.error("--data is required unless --merge is given")
    build(args.data, args.out, args.k, args.lo, args.hi,
          args.seed, args.shard, args.num_shards, args.max_cones)


if __name__ == "__main__":
    main()
