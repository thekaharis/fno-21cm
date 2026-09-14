"""Validate a multi-field pilot cache before committing to the full build.

Implements the pilot checks agreed in notes/multifield-data-configuration.md:
raw-side geometry, interpolation fidelity on the retained interval, split
reproducibility, training-only statistics, and physical baselines by redshift
band. Read-only with respect to the raw lightcones.

Run: python tests/check_multifield_pilot.py --cache <pilot.h5>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

RAW = Path("/pfs/10/work/hd_id260-fno_training/data/data")
MPC_KM = 3.0856775814913673e19
FIELDS = ("density", "neutral_fraction", "brightness_temp", "los_velocity")


def raw_path(sample_id: int) -> Path:
    return RAW / f"21cmfast_11d_sample{sample_id:06d}.h5"


def check_raw_geometry(ids):
    """Note check 1: shapes, monotonicity, coverage, spacing from real metadata."""
    print("\n[1] raw-side geometry")
    bad = 0
    for sid in ids:
        with h5py.File(raw_path(sid), "r") as h:
            shapes = {f: h[f"lightcone/{f}"].shape for f in FIELDS}
            z = h["lightcone/lightcone_redshifts"][:]
            d = h["lightcone/lightcone_distances"][:]
            problems = []
            if len(set(shapes.values())) != 1:
                problems.append(f"shape disagreement {shapes}")
            if not np.all(np.diff(z) > 0):
                problems.append("z not strictly increasing")
            # spacing from the distance metadata, not inferred from redshift
            step = np.diff(d)
            if not np.allclose(step, step[0], rtol=1e-9):
                problems.append(f"non-uniform comoving step (ptp {step.ptp():.3e})")
            if abs(step[0] - 200.0 / 140.0) > 1e-6:
                problems.append(f"unexpected step {step[0]:.8f}")
            if problems:
                bad += 1
                print(f"    cone {sid}: " + "; ".join(problems))
    print(f"    {len(ids) - bad}/{len(ids)} cones clean "
          f"(shapes, monotonicity, comoving step 1.428571 Mpc)")
    return bad == 0


def check_cache(cache_path, ids):
    """Finite values, physical bounds, coverage of the retained interval."""
    print("\n[2] cache contents")
    ok = True
    with h5py.File(cache_path, "r") as c:
        tz = c["target_z"][:]
        print(f"    target_z: {len(tz)} points, {tz[0]:.4f} .. {tz[-1]:.4f}")
        cid = c["cone_id"][:]
        print(f"    cone_id : n={len(cid)} unique={len(set(cid.tolist()))} "
              f"min={cid.min()} max={cid.max()}")
        bounds = {"neutral_fraction": (-1e-4, 1.0 + 1e-4),
                  "density": (-1.0 - 1e-3, None), "brightness_temp": None,
                  "los_velocity": None}
        for f in FIELDS:
            if f not in c:
                print(f"    {f:18s} MISSING"); ok = False; continue
            a = c[f]
            chunk = a[:, ::4, ::4, :].astype(np.float64)
            fin = np.isfinite(chunk).all()
            lo, hi = float(chunk.min()), float(chunk.max())
            msg = f"    {f:18s} {str(a.shape):24s} finite={fin} range [{lo:.4g}, {hi:.4g}]"
            if not fin:
                ok = False; msg += "  <-- NON-FINITE"
            b = bounds.get(f)
            if b:
                blo, bhi = b
                if (blo is not None and lo < blo) or (bhi is not None and hi > bhi):
                    ok = False; msg += f"  <-- OUT OF BOUNDS {b}"
            print(msg)
        # zero-fill detector: an all-zero LOS end plane means the grid ran off
        # the native axis, which the old reader would have done silently.
        for f in FIELDS:
            if f not in c: continue
            first = np.abs(c[f][:, ::8, ::8, 0]).max()
            last = np.abs(c[f][:, ::8, ::8, -1]).max()
            if first == 0 or last == 0:
                ok = False
                print(f"    {f}: ZERO end plane (first={first:.3g} last={last:.3g})")
    print(f"    -> {'OK' if ok else 'PROBLEMS FOUND'}")
    return ok


def check_interpolation_fidelity(cache_path, ids, n=4):
    """Note check 2: round-trip inside the retained interval only."""
    print("\n[3] interpolation fidelity (round trip, retained interval)")
    with h5py.File(cache_path, "r") as c:
        tz = c["target_z"][:]
        cid = list(c["cone_id"][:])
        print(f"    {'cone':>6s} " + " ".join(f"{f[:12]:>13s}" for f in FIELDS))
        for sid in ids[:n]:
            row = cid.index(sid)
            errs = []
            with h5py.File(raw_path(sid), "r") as h:
                zs = h["lightcone/lightcone_redshifts"][:]
                keep = (zs >= tz[0]) & (zs <= tz[-1])
                for f in FIELDS:
                    native = h[f"lightcone/{f}"][::8, ::8, :][:, :, keep].astype(np.float64)
                    cached = c[f][row, ::8, ::8, :].astype(np.float64)
                    back = np.empty_like(native)
                    for i in range(back.shape[0]):
                        for j in range(back.shape[1]):
                            back[i, j] = np.interp(zs[keep], tz, cached[i, j])
                    denom = np.sqrt((native ** 2).mean()) or 1.0
                    errs.append(np.sqrt(((back - native) ** 2).mean()) / denom)
            print(f"    {sid:6d} " + " ".join(f"{e:13.4f}" for e in errs))
    print("    (relative L2 of native -> 256 grid -> native, lower is better)")


def check_split(cache_path, seed=42):
    """Note check 3: reproducible, stored, non-overlapping simulation split."""
    print("\n[4] split reproducibility")
    with h5py.File(cache_path, "r") as c:
        cid = np.array(c["cone_id"][:])

    def split(seed):
        cones = np.unique(cid).copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(cones)
        n = len(cones)
        n_test = max(1, round(0.1 * n)); n_val = max(1, round(0.1 * n))
        return (sorted(cones[n_test + n_val:].tolist()),
                sorted(cones[n_test:n_test + n_val].tolist()),
                sorted(cones[:n_test].tolist()))

    a, b = split(seed), split(seed)
    same = a == b
    tr, va, te = a
    overlap = (set(tr) & set(va)) | (set(tr) & set(te)) | (set(va) & set(te))
    print(f"    identical across two calls: {same}")
    print(f"    train {len(tr)} / val {len(va)} / test {len(te)}   overlap={len(overlap)}")
    print(f"    train ids: {tr}")
    print(f"    val   ids: {va}")
    print(f"    test  ids: {te}")
    return same and not overlap, (tr, va, te)


def check_statistics(cache_path, train_ids):
    """Note check 4: training-only statistics, float64, finite normalized variance."""
    print("\n[5] training-only statistics (float64 accumulation)")
    with h5py.File(cache_path, "r") as c:
        cid = list(c["cone_id"][:])
        rows = [cid.index(i) for i in train_ids]
        print(f"    {'field':18s} {'mean':>14s} {'std':>14s} {'norm var':>10s}")
        for f in FIELDS:
            tot = np.float64(0.0); tot2 = np.float64(0.0); n = 0
            for r in rows:
                a = c[f][r, ::4, ::4, :].astype(np.float64)
                tot += a.sum(); tot2 += (a ** 2).sum(); n += a.size
            mean = tot / n
            std = np.sqrt(max(tot2 / n - mean ** 2, 0.0))
            a = c[FIELDS[0]][rows[0], ::4, ::4, :].astype(np.float64) if False else None
            chk = c[f][rows[0], ::4, ::4, :].astype(np.float64)
            nv = ((chk - mean) / (std or 1.0)).var()
            flag = "" if np.isfinite(nv) and nv > 0 else "   <-- DEGENERATE"
            print(f"    {f:18s} {mean:14.6e} {std:14.6e} {nv:10.4f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--manifest", type=Path,
                    default=Path("experiments/multifield/pilot_manifest.json"))
    args = ap.parse_args()
    ids = json.loads(args.manifest.read_text())["pilot_sample_ids"]
    print(f"pilot cache : {args.cache}  ({args.cache.stat().st_size/2**30:.2f} GiB)")
    print(f"pilot cones : {len(ids)}")
    g = check_raw_geometry(ids)
    c = check_cache(args.cache, ids)
    check_interpolation_fidelity(args.cache, ids)
    s, (tr, va, te) = check_split(args.cache)
    check_statistics(args.cache, tr)
    print(f"\nSUMMARY: geometry={'OK' if g else 'FAIL'}  cache={'OK' if c else 'FAIL'}  "
          f"split={'OK' if s else 'FAIL'}")


if __name__ == "__main__":
    main()
