"""Rebuild brightness_temp with the velocity-gradient term clipped.

The raw lightcones were produced with ``include_dvdr_in_tau21=True`` and, as
stored, the optical-depth velocity factor 1/(1 + (dv/dr)/H) is not limited: at
velocity caustics, where (dv/dr)/H -> -1, brightness_temp diverges (up to
~1e6 mK; in one validation cone 0.03% of voxels carry 99.8% of its T_b
variance). 21cmFAST normally bounds this term; here it has to be done
afterwards.

The gas is optically thin everywhere (tau_21 ~ 1e-4, also at the spikes), so
T_b is proportional to the velocity factor and the dv/dr-free signal is
T_b * (1 + r), r = (dv/dr)/H. The rebuilt field is

    T_b_clean = T_b * (1 + r) / (1 + clip(r, -MAX_DVDR, +MAX_DVDR))

with r from central finite differences of los_velocity along the comoving LOS
distance (this reproduces 21cmFAST's gradient far better than a spectral
derivative: spikes sit at r = -1.00 to three decimals). Voxels with
|r| <= MAX_DVDR are copied bit for bit.

Near the singularity the finite-difference r cannot match 21cmFAST's exactly,
so a few voxels per cone (~1e-5) remain unphysical after the rebuild; they are
clipped to +-RESIDUAL_CLIP mK and counted in the file attributes.

Everything else (density, los_velocity, neutral_fraction, axes, params,
provenance attributes, chunking) is copied from the source mirror unchanged.

  python tools_clean_brightness_temp.py --shard 0 --num-shards 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

SRC = Path("/pfs/10/work/hd_id260-fno_training/data/native_mirror_4f")
OUT = Path("/pfs/10/work/hd_id260-fno_training/data/native_mirror_4f_tbclean")
MAX_DVDR = 0.2            # 21cmFAST's conventional bound on |dv/dr|/H
RESIDUAL_CLIP = 1000.0    # mK; physical |T_b| stays well below this
HUBBLE_H = 0.6766         # 21cmFAST (Planck18) default; OMm is a sampled parameter
H0_PER_S = 100.0*HUBBLE_H/3.0856775814913673e19


def hubble(z, omm):
    return H0_PER_S*np.sqrt(omm*(1+z)**3 + 1 - omm)


def clean(tb, velocity, distances, redshifts, omm):
    grad = np.gradient(velocity.astype(np.float64), distances, axis=2)
    r = grad/hubble(redshifts, omm)[None, None, :]
    changed = np.abs(r) > MAX_DVDR
    out = tb.astype(np.float64)
    out[changed] = out[changed]*(1 + r[changed])/(1 + np.clip(r[changed], -MAX_DVDR, MAX_DVDR))
    residual = np.abs(out) > RESIDUAL_CLIP
    out = np.clip(out, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    result = tb.copy()
    result[changed | residual] = out[changed | residual].astype(tb.dtype)
    return result, {"changed": int(changed.sum()), "residual_clipped": int(residual.sum()),
                    "voxels": int(tb.size), "raw_max_abs": float(np.abs(tb).max()),
                    "clean_max_abs": float(np.abs(result).max())}


def build_one(path, out_dir, overwrite=False):
    dst = out_dir/path.name
    if dst.exists() and not overwrite:
        return "exists", None
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(path, "r") as f, h5py.File(tmp, "w") as g:
        src = f["lightcone"]
        out = g.create_group("lightcone")
        for key in src:
            if key != "brightness_temp":
                f.copy(src[key], out, name=key)
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["params/names"][:]]
        omm = float(dict(zip(names, f["params/values"][:]))["OMm"])
        tb = src["brightness_temp"][:]
        cleaned, stats = clean(tb, src["los_velocity"][:], src["lightcone_distances"][:],
                               src["lightcone_redshifts"][:], omm)
        ds = out.create_dataset("brightness_temp", data=cleaned, chunks=src["brightness_temp"].chunks)
        ds.attrs["dvdr_clip"] = MAX_DVDR
        f.copy("params", g)
        for key, value in f.attrs.items():
            g.attrs[key] = value
        g.attrs["tb_clean_max_dvdr"] = MAX_DVDR
        g.attrs["tb_clean_residual_clip_mK"] = RESIDUAL_CLIP
        g.attrs["tb_clean_hubble_h"] = HUBBLE_H
        for key, value in stats.items():
            g.attrs[f"tb_clean_{key}"] = value
        g.attrs["tb_clean_source"] = str(path)
    tmp.replace(dst)
    return "written", stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--samples", type=int, nargs="*", help="only these sample ids (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    paths = sorted(args.src.glob("21cmfast_11d_sample*.h5"))
    if args.samples:
        paths = [p for p in paths if int(p.stem[-6:]) in set(args.samples)]
    mine = paths[args.shard::args.num_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    changed = residual = voxels = 0
    for p in mine:
        status, stats = build_one(p, args.out, args.overwrite)
        if stats:
            changed += stats["changed"]; residual += stats["residual_clipped"]; voxels += stats["voxels"]
            if stats["raw_max_abs"] > RESIDUAL_CLIP:
                print(f"  {p.stem}: max |T_b| {stats['raw_max_abs']:.0f} -> {stats['clean_max_abs']:.0f} mK, "
                      f"residual-clipped {stats['residual_clipped']}", flush=True)
    print(f"shard {args.shard}: {len(mine)} cones; changed {changed/max(voxels,1):.2%} of voxels, "
          f"residual-clipped {residual} ({residual/max(voxels,1):.1e})", flush=True)


if __name__ == "__main__":
    main()
