"""Separate cache-imposed error from model error for a multi-field prediction.

Three quantities per field, all relative L2 against the field's own RMS:

  cache      native -> 256-point grid -> native.  What the LOS grid discards,
             independent of any model. A floor on end-to-end accuracy.
  model      prediction vs the *cached* truth, both on the 256 grid. NOT floored
             by the cache: a perfect model scores zero here.
  end_to_end prediction lifted back to the native grid vs the native truth.
             What the pipeline actually delivers; bounded below by `cache`.

If end_to_end ~= cache, the grid dominates and a better model cannot help much.
If end_to_end >> cache, the model is the limiting factor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

RAW = Path("/pfs/10/work/hd_id260-fno_training/data/data")


def rel_l2(a, b):
    """||a - b|| / ||b||, both arrays on the same grid."""
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


def analyse(pred_path, sub=2):
    with h5py.File(pred_path, "r") as f:
        cone = int(f.attrs["cone_id"])
        tz = f["target_z"][:]
        fields = list(f["target"])
        cached = {k: f[f"target/{k}"][::sub, ::sub, :].astype(np.float64) for k in fields}
        predic = {k: f[f"prediction/{k}"][::sub, ::sub, :].astype(np.float64) for k in fields}

    out = []
    with h5py.File(RAW / f"21cmfast_11d_sample{cone:06d}.h5", "r") as h:
        zs = h["lightcone/lightcone_redshifts"][:]
        keep = (zs >= tz[0]) & (zs <= tz[-1])
        zk = zs[keep]
        for k in fields:
            nat = h[f"lightcone/{k}"][::sub, ::sub, :][:, :, keep].astype(np.float64)
            nx, ny, _ = nat.shape
            rt = np.empty_like(nat)      # native -> grid -> native
            up = np.empty_like(nat)      # prediction -> native
            for i in range(nx):
                for j in range(ny):
                    coarse = np.interp(tz, zk, nat[i, j])
                    rt[i, j] = np.interp(zk, tz, coarse)
                    up[i, j] = np.interp(zk, tz, predic[k][i, j])
            out.append(dict(cone=cone, field=k,
                            cache=rel_l2(rt, nat),
                            model=rel_l2(predic[k], cached[k]),
                            end_to_end=rel_l2(up, nat)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--sub", type=int, default=2, help="transverse stride")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for p in args.prediction:
        rows.extend(analyse(p, args.sub))
    print(f"{'cone':>5s} {'field':>17s} {'cache':>8s} {'model':>8s} {'end_to_end':>11s} "
          f"{'grid share':>11s}")
    for r in sorted(rows, key=lambda r: (r["field"], r["cone"])):
        # fraction of the end-to-end error variance attributable to the grid
        share = (r["cache"] / r["end_to_end"]) ** 2 if r["end_to_end"] > 0 else float("nan")
        print(f"{r['cone']:5d} {r['field']:>17s} {r['cache']:8.4f} {r['model']:8.4f} "
              f"{r['end_to_end']:11.4f} {min(share,1.0)*100:10.1f}%")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
