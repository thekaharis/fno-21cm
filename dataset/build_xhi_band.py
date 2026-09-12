"""Build a slice cache targeted at a chosen x_HI band.

``build_trainset.py`` samples slices with a *weight* -- ones outside its
reionization window keep only ``floor=0.02`` -- so the tails of x_HI are ~50x
under-represented.  That is right for training and wrong for analysing a
specific regime: the whole late-reionization range x_HI < 0.36 holds just 379
of the 3960 test slices, and the lowest bins end up too thin to fit anything.

This builds a cache in the same ``SliceCache`` format, drawing only from LOS
positions whose mean neutral fraction lands in ``[--lo, --hi]``.

Cone ids match ``build_trainset.py`` exactly (index into the sorted lightcone
glob), so ``--cones`` can restrict the build to a run's val/test cones and the
result stays clear of anything the model trained on.

    python -m dataset.build_xhi_band --lo 0.0 --hi 0.36 --k 12 \
        --split-from checkpoints/2d_xhi/fno_whno/xhi2d_whno_glob_lr3e4 --splits val,test \
        --out xhi_band_000_036.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from dataset import paths
from dataset.build_slices import xHI_profile
from dataset.lightcone_params import PARAM_NAMES, read_sampled_params


def band_indices(prof: np.ndarray, k: int, rng: np.random.Generator,
                 lo: float, hi: float, stratify: bool = True) -> np.ndarray:
    """Up to *k* LOS indices whose mean x_HI lies in [lo, hi].

    Unlike the training sampler this is a hard filter, not a weight -- the
    point is to saturate one band, not to keep a representative mixture.

    Stratified by default, and that matters: once a cone finishes reionizing
    every later LOS position sits at x_HI ~ 0, so sampling eligible indices
    uniformly buries the band under fully-ionized slices (a 25-cone trial gave
    104 of 156 slices in 0.00-0.05).  Those are the degenerate maps where a
    contrast map "wins" by thresholding away noise on an empty field.  Taking
    at most one index per sub-bin spreads the draw evenly across the band.
    """
    eligible = np.flatnonzero((prof >= lo) & (prof <= hi))
    if eligible.size == 0 or not stratify:
        if eligible.size <= k:
            return np.sort(eligible)
        return np.sort(rng.choice(eligible, size=k, replace=False))
    edges = np.linspace(lo, hi, k + 1)
    which = np.clip(np.digitize(prof[eligible], edges) - 1, 0, k - 1)
    picked = [rng.choice(eligible[which == b]) for b in range(k)
              if np.any(which == b)]
    return np.sort(np.asarray(picked, dtype=np.int64))


def extract_one(path: Path, cone_id: int, k: int, rng, lo: float, hi: float,
                stratify: bool = True):
    with h5py.File(path, "r") as f:
        idx = band_indices(xHI_profile(f), k, rng, lo, hi, stratify)
        if idx.size == 0:
            return None
        dens, xH = f["lightcone/density"], f["lightcone/neutral_fraction"]
        z_all = np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float32)
        x = np.stack([np.asarray(dens[:, :, i], dtype=np.float32) for i in idx])
        y = np.stack([np.asarray(xH[:, :, i], dtype=np.float32) for i in idx])
        params = read_sampled_params(f)
        if not np.isfinite(params).all():
            raise ValueError("sampled parameters contain missing/non-finite values")
    n = len(idx)
    return (x, y, z_all[idx],
            y.mean(axis=(1, 2), dtype=np.float64).astype(np.float32),
            np.full(n, cone_id, dtype=np.int32), np.tile(params, (n, 1)))


def cones_from_split(run_dir: Path, splits: list[str]) -> set[int]:
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    key = {"train": "train_cone_ids", "val": "val_cone_ids",
           "test": "test_cone_ids"}
    out: set[int] = set()
    for s in splits:
        out |= set(int(c) for c in meta["split"][key[s]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.LIGHTCONES)
    ap.add_argument("--out", type=Path, default=Path("xhi_band.h5"),
                    help="filename; written into data/compressed unless absolute")
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=0.36)
    ap.add_argument("--k", type=int, default=12, help="max slices per cone")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-from", type=Path, default=None,
                    help="run dir whose cone split restricts the build")
    ap.add_argument("--splits", default="val,test",
                    help="which splits to draw from (default val,test)")
    ap.add_argument("--no-stratify", action="store_true",
                    help="sample eligible LOS uniformly instead of one per sub-bin")
    ap.add_argument("--max-cones", type=int, default=None)
    args = ap.parse_args()

    if not 0.0 <= args.lo < args.hi <= 1.0:
        sys.exit("require 0 <= lo < hi <= 1")
    out = args.out if args.out.is_absolute() else paths.compressed(args.out.name)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data).glob("21cmfast_11d_sample*.h5"))
    if not files:
        sys.exit(f"no lightcone files in {args.data}")
    todo = list(enumerate(files))
    if args.split_from is not None:
        keep = cones_from_split(args.split_from, args.splits.split(","))
        todo = [(c, p) for c, p in todo if c in keep]
        print(f"[band] restricted to {len(todo)} cones from "
              f"{args.split_from.name} splits={args.splits}")
    if args.max_cones:
        todo = todo[: args.max_cones]

    print(f"[band] {len(todo)} cones, x_HI in [{args.lo}, {args.hi}], k<={args.k}")
    buf: dict[str, list] = {n: [] for n in
                            ("x", "y", "z", "xHI_mean", "cone_id", "params")}
    empty = 0
    for j, (cone_id, path) in enumerate(todo):
        rng = np.random.default_rng([args.seed, cone_id])
        try:
            got = extract_one(path, cone_id, args.k, rng, args.lo, args.hi,
                              stratify=not args.no_stratify)
        except Exception as exc:                             # noqa: BLE001
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
            continue
        if got is None:
            empty += 1
            continue
        for name, arr in zip(("x", "y", "z", "xHI_mean", "cone_id", "params"), got):
            buf[name].append(arr)
        if (j + 1) % 200 == 0:
            n = sum(len(a) for a in buf["x"])
            print(f"  ... {j + 1}/{len(todo)} cones, {n} slices", flush=True)

    if not buf["x"]:
        sys.exit("no slices matched the band")
    data = {n: np.concatenate(v) for n, v in buf.items()}
    with h5py.File(out, "w") as o:
        for name, arr in data.items():
            o.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        o.attrs["k_per_cone"] = args.k
        o.attrs["xHI_window"] = (args.lo, args.hi)
        o.attrs["xHI_band_hard_filter"] = True
        o.attrs["xHI_band_stratified"] = not args.no_stratify
        o.attrs["slice_cache_version"] = 2
        o.attrs["param_names"] = np.array(PARAM_NAMES, dtype="S16")
        o.attrs["source_splits"] = args.splits
    m = data["xHI_mean"]
    print(f"\n[band] wrote {out}  ({len(m)} slices, "
          f"{len(np.unique(data['cone_id']))} cones, {empty} cones had none)")
    print(f"[band] x_HI {m.min():.4f}-{m.max():.4f}, median {np.median(m):.4f}")
    for a, b in ((0, .05), (.05, .1), (.1, .15), (.15, .2), (.2, .25),
                 (.25, .3), (.3, .36)):
        print(f"        {a:.2f}-{b:.2f}: {int(((m >= a) & (m < b)).sum())}")


if __name__ == "__main__":
    main()
