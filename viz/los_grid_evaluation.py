#!/usr/bin/env python3
"""Round-trip evaluation of candidate LOS sampling grids for the cube cache.

Motivation
----------
The cube cache downsamples the native lightcone LOS axis (2340 slices,
1.43 Mpc uniform comoving) to 256 slices *uniform in redshift* -- which is
coarsest (~37 Mpc) at low z where reionization fronts live and finest
(~5 Mpc) at high z where the field is saturated neutral. Any information
destroyed here is unlearnable for every model trained on the cache, so the
grid choice bounds model quality.

This script measures that bound directly, with no training: for each
candidate grid it interpolates the native truth onto the grid and back
(native -> grid -> native, both linear, matching build_cubes.py), then
quantifies the damage where it matters:

* transition RMSE   -- error over voxels with native x_HI in (0.02, 0.98)
* sharpness         -- per-ray max |d x_HI / dchi| after / before (fronts only)
* fronts missed     -- rays whose native 0.5-crossing disappears entirely
* global RMSE       -- whole-volume (dominated by saturated regions; context)

Candidate grids (same slice budget unless noted):

* uniform_z    -- current cache scheme (baseline)
* uniform_chi  -- uniform comoving distance
* warped       -- density ~ ensemble-mean |d<x_HI>/dchi| + uniform floor,
                  CDF-inverted (the "sample where ionization changes" grid)
* crop15_chi   -- z in [5, 15] uniform-chi; z > 15 reconstructed as constant

Typical use (login node, CPU only)::

    python -m viz.los_grid_evaluation --n-cones 12 --out grid_eval_out/

Self-test (synthetic, no data needed)::

    python -m viz.los_grid_evaluation --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("/pfs/10/work/hd_id260-fno_training/data/data")
Z_REPORT = (5.5, 7.0, 10.0, 15.0, 20.0)   # cell sizes reported at these z


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #
def warped_grid(chi: np.ndarray, weight: np.ndarray, n: int,
                floor_frac: float = 0.2) -> np.ndarray:
    """CDF-invert a sampling density ~ weight + uniform floor onto n points."""
    w = np.clip(np.asarray(weight, float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = (1.0 - floor_frac) * w / np.trapezoid(w, chi) + \
        floor_frac / (chi[-1] - chi[0])
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(chi))])
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, chi)


def build_grids(z: np.ndarray, chi: np.ndarray,
                weights: dict[str, np.ndarray],
                budgets: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Candidate grids as ``target_z`` arrays (the cube pipeline's coordinate).

    ``chi`` is the ENSEMBLE-MEAN chi(z): per-cone comoving distance varies
    with the cosmology draw, so grids are defined once in z via the mean
    mapping and each cone is interpolated in z (matching build_cubes.py).
    ``weights`` maps a name to a sampling density on (z, chi); each becomes
    a CDF-inverted grid (e.g. "warped" = ensemble-mean |dx/dchi|,
    "envelope90" = 90th percentile across cones, which covers early/late
    reionizers instead of only the average one).
    """
    chi_to_z = lambda c: np.interp(c, chi, z)  # noqa: E731
    grids: dict[str, np.ndarray] = {}
    for n in budgets:
        grids[f"uniform_z_{n}"] = np.linspace(z[0], z[-1], n)
        grids[f"uniform_chi_{n}"] = chi_to_z(np.linspace(chi[0], chi[-1], n))
        hi = np.interp(15.0, z, chi)
        grids[f"crop15_chi_{n}"] = chi_to_z(np.linspace(chi[0], hi, n))
        for wname, w in weights.items():
            grids[f"{wname}_{n}"] = chi_to_z(warped_grid(chi, w, n))
    return grids


def classify_cone(history: np.ndarray, z: np.ndarray) -> str:
    """Reionization-timing class from a cone's global x_HI(z) history.

    z50 = redshift where the global neutral fraction crosses 0.5.
    """
    h = np.clip(history, 0.0, 1.0)
    if h[0] >= 0.5:          # midpoint below the z range: reionizes late
        return "late"
    z50 = float(np.interp(0.5, h, z))
    if z50 > 9.5:
        return "early"
    return "mid" if z50 > 6.5 else "late"


# --------------------------------------------------------------------------- #
# Fast linear interpolation along the last axis (shared index tables)
# --------------------------------------------------------------------------- #
class LinInterp:
    """Precomputed linear interpolation src_chi -> dst_chi along axis -1."""

    def __init__(self, src_chi: np.ndarray, dst_chi: np.ndarray):
        src = np.asarray(src_chi, float)
        dst = np.clip(np.asarray(dst_chi, float), src[0], src[-1])
        i1 = np.clip(np.searchsorted(src, dst), 1, len(src) - 1)
        i0 = i1 - 1
        denom = src[i1] - src[i0]
        w = np.where(denom > 0, (dst - src[i0]) / np.where(denom > 0, denom, 1.0), 0.0)
        self.i0, self.i1, self.w = i0, i1, w.astype(np.float32)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        return (cube[..., self.i0] * (1.0 - self.w)
                + cube[..., self.i1] * self.w)


# --------------------------------------------------------------------------- #
# Per-cone metrics
# --------------------------------------------------------------------------- #
def evaluate_roundtrip(native: np.ndarray, recon: np.ndarray,
                       chi: np.ndarray) -> dict[str, float]:
    err = recon - native
    trans = (native > 0.02) & (native < 0.98)
    out = {
        "global_rmse": float(np.sqrt(np.mean(err ** 2))),
        "transition_rmse": (float(np.sqrt(np.mean(err[trans] ** 2)))
                            if trans.any() else np.nan),
        "transition_frac": float(trans.mean()),
    }
    # LOS sharpness + front survival, per transverse ray
    g_nat = np.abs(np.gradient(native, chi, axis=-1)).max(axis=-1)
    g_rec = np.abs(np.gradient(recon, chi, axis=-1)).max(axis=-1)
    sign_nat = np.diff(np.signbit(native - 0.5), axis=-1).any(axis=-1)
    sign_rec = np.diff(np.signbit(recon - 0.5), axis=-1).any(axis=-1)
    fronts = sign_nat & (g_nat > 0.01)   # rays with a real LOS front
    if fronts.any():
        out["sharpness_ratio"] = float((g_rec[fronts] / g_nat[fronts]).mean())
        out["fronts_missed_pct"] = float(100.0 * (~sign_rec[fronts]).mean())
        out["n_front_rays"] = int(fronts.sum())
    else:
        out["sharpness_ratio"] = np.nan
        out["fronts_missed_pct"] = np.nan
        out["n_front_rays"] = 0
    return out


def cell_sizes_at(grid_z: np.ndarray, z: np.ndarray, chi: np.ndarray,
                  z_report=Z_REPORT) -> dict[str, float]:
    gchi = np.interp(grid_z, z, chi)
    dchi = np.abs(np.diff(gchi))
    out = {}
    for zq in z_report:
        if grid_z[-1] < zq or grid_z[0] > zq:
            out[f"cell@z{zq:g}"] = np.nan
            continue
        i = int(np.clip(np.searchsorted(grid_z, zq) - 1, 0, len(dchi) - 1))
        out[f"cell@z{zq:g}"] = float(dchi[i])
    return out


# --------------------------------------------------------------------------- #
# Data driver
# --------------------------------------------------------------------------- #
def run_on_data(data_dir: Path, n_cones: int, budgets: tuple[int, ...],
                field: str, out_dir: Path,
                envelope_pct: float = 90.0) -> dict[str, dict]:
    import h5py

    files = sorted(data_dir.glob("21cmfast_11d_sample*.h5"))
    if not files:
        raise SystemExit(f"no lightcone files in {data_dir}")
    picks = files[:: max(1, len(files) // n_cones)][:n_cones]
    print(f"[grids] evaluating {len(picks)} cones from {len(files)} files")

    # Common coordinate is REDSHIFT: native LOS length and chi(z) vary per
    # cone with the cosmology draw (e.g. 2340 vs 2178 slices), so grids are
    # defined on the ensemble-mean chi(z) and applied per cone in z.
    z_common = np.linspace(5.0, 25.0, 2340)
    hist, chi_maps = [], []
    for p in picks:
        with h5py.File(p, "r") as f:
            nz = f["lightcone/node_redshifts"][:]
            g = f["lightcone/global_quantities/neutral_fraction"][:]
            lz = f["lightcone/lightcone_redshifts"][:]
            lchi = f["lightcone/lightcone_distances"][:]
        o = np.argsort(nz)
        hist.append(np.interp(z_common, nz[o], g[o]))
        chi_maps.append(np.interp(z_common, lz, lchi))
    xbar = np.mean(hist, axis=0)
    chi_mean = np.mean(chi_maps, axis=0)

    # sampling densities: ensemble mean vs high-percentile envelope of the
    # per-cone |d x_HI / d chi| -- the envelope covers every timing class
    per_cone_dx = np.abs(np.gradient(np.asarray(hist), chi_mean, axis=1))
    weights = {
        "warped": per_cone_dx.mean(axis=0),
        f"envelope{envelope_pct:g}": np.percentile(per_cone_dx,
                                                   envelope_pct, axis=0),
    }
    grids = build_grids(z_common, chi_mean, weights, budgets)

    classes = [classify_cone(h, z_common) for h in hist]
    from collections import Counter
    print(f"[grids] cone timing classes: {dict(Counter(classes))}")

    per_grid: dict[str, list[dict]] = {name: [] for name in grids}
    for ci, p in enumerate(picks):
        with h5py.File(p, "r") as f:
            native = f[f"lightcone/{field}"][:].astype(np.float32)
            lz = f["lightcone/lightcone_redshifts"][:]
            lchi = f["lightcone/lightcone_distances"][:]
        for name, grid_z in grids.items():
            fwd, bwd = LinInterp(lz, grid_z), LinInterp(grid_z, lz)
            recon = bwd(fwd(native))
            row = evaluate_roundtrip(native, recon, lchi)
            row["cone_class"] = classes[ci]
            per_grid[name].append(row)
        print(f"[grids] cone {ci + 1}/{len(picks)} done "
              f"({p.name}, {len(lz)} native slices, {classes[ci]})")

    results = {}
    for name, rows in per_grid.items():
        num = [k for k in rows[0] if k not in ("n_front_rays", "cone_class")]
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in num}
        agg["n_front_rays"] = int(sum(r["n_front_rays"] for r in rows))
        agg["n_slices"] = len(grids[name])
        agg.update(cell_sizes_at(grids[name], z_common, chi_mean))
        for cls in ("early", "mid", "late"):
            sub = [r for r in rows if r["cone_class"] == cls]
            if sub:
                for k in ("transition_rmse", "sharpness_ratio",
                          "fronts_missed_pct"):
                    agg[f"{cls}_{k}"] = float(np.nanmean([r[k] for r in sub]))
        results[name] = agg

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "grids.npz",
                        **{n: g for n, g in grids.items()},
                        z_common=z_common, chi_mean=chi_mean, xbar=xbar)
    with open(out_dir / "grid_eval.csv", "w", newline="") as f:
        keys = list(next(iter(results.values())).keys())
        w = csv.writer(f)
        w.writerow(["grid"] + keys)
        for name, agg in results.items():
            w.writerow([name] + [agg[k] for k in keys])
    (out_dir / "grid_eval.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def print_table(results: dict[str, dict]) -> None:
    cols = ("n_slices", "transition_rmse", "sharpness_ratio",
            "fronts_missed_pct", "global_rmse")
    header = f"{'grid':<18}" + "".join(f"{c:>18}" for c in cols) \
        + "".join(f"{f'cell@z{zq:g}':>12}" for zq in Z_REPORT)
    print("\n" + header)
    print("-" * len(header))
    for name, agg in sorted(results.items(),
                            key=lambda kv: kv[1].get("transition_rmse", 9)):
        row = f"{name:<18}"
        for c in cols:
            v = agg.get(c, np.nan)
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{v:>18}"
        for zq in Z_REPORT:
            v = agg.get(f"cell@z{zq:g}", np.nan)
            row += f"{v:>12.1f}"
        print(row)
    print("\n(sharpness_ratio: 1.0 = fronts fully preserved; "
          "cell@z in Mpc; sorted best transition_rmse first)")

    # per-timing-class breakdown: the ensemble-mean warp hides how badly a
    # grid treats the rare early reionizers, so show them separately
    cls_cols = [f"{cls}_{m}" for cls in ("early", "mid", "late")
                for m in ("transition_rmse", "sharpness_ratio")]
    if any(c in agg for agg in results.values() for c in cls_cols):
        header = f"{'grid':<18}" + "".join(
            f"{c.replace('transition_rmse', 'rmse').replace('sharpness_ratio', 'sharp'):>16}"
            for c in cls_cols)
        print("\n" + header)
        print("-" * len(header))
        for name, agg in sorted(results.items(),
                                key=lambda kv: kv[1].get("transition_rmse", 9)):
            row = f"{name:<18}"
            for c in cls_cols:
                v = agg.get(c, np.nan)
                row += f"{v:>16.4f}"
            print(row)


# --------------------------------------------------------------------------- #
# Self-test (synthetic)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    rng = np.random.default_rng(0)
    # synthetic geometry: chi uniform, z convex in chi so uniform_z is coarse
    # at LOW chi -- mirroring the real lightcone
    n_nat = 2000
    chi = np.linspace(0.0, 3300.0, n_nat)
    z = 5.0 + 20.0 * (chi - chi[0]) ** 2 / (chi[-1] - chi[0]) ** 2
    # fronts at low chi (transition zone), tanh width ~3 Mpc
    rays = []
    for _ in range(300):
        pos = rng.uniform(200.0, 900.0)
        rays.append(0.5 * (1 + np.tanh((chi - pos) / 3.0)))
    native = np.asarray(rays, dtype=np.float32)[None, ...]  # (1, 300, n_nat)
    xbar = native[0].mean(axis=0)

    dxbar = np.abs(np.gradient(xbar, chi))
    grids = build_grids(z, chi, {"warped": dxbar}, budgets=(256,))
    res = {}
    for name, grid_z in grids.items():
        fwd, bwd = LinInterp(z, grid_z), LinInterp(grid_z, z)
        res[name] = evaluate_roundtrip(native, bwd(fwd(native)), chi)
    # classifier sanity: synthetic fronts at low chi = low z = late reionizers
    assert classify_cone(xbar, z) in ("late", "mid")

    uz, wp = res["uniform_z_256"], res["warped_256"]
    print(f"[selftest] uniform_z transition_rmse={uz['transition_rmse']:.4f} "
          f"sharpness={uz['sharpness_ratio']:.3f}")
    print(f"[selftest] warped    transition_rmse={wp['transition_rmse']:.4f} "
          f"sharpness={wp['sharpness_ratio']:.3f}")
    assert wp["transition_rmse"] < 0.5 * uz["transition_rmse"], \
        "warped grid should beat uniform_z on fronts by a wide margin"
    assert wp["sharpness_ratio"] > uz["sharpness_ratio"], \
        "warped grid should preserve gradients better"
    assert res["uniform_chi_256"]["transition_rmse"] < uz["transition_rmse"], \
        "uniform_chi should beat uniform_z when fronts sit at low chi"
    # identity check: native-resolution grid must round-trip losslessly
    fwd, bwd = LinInterp(z, z), LinInterp(z, z)
    assert np.allclose(bwd(fwd(native)), native, atol=1e-6)
    print("[selftest] OK")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--n-cones", type=int, default=12,
                    help="lightcones to evaluate (spread over the design set)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512],
                    help="slice budgets to build each grid family at")
    ap.add_argument("--field", default="neutral_fraction",
                    help="lightcone field to round-trip (try also: density)")
    ap.add_argument("--envelope-pct", type=float, default=90.0,
                    help="percentile across cones for the envelope warp "
                         "(covers early/late reionizers, not just the mean)")
    ap.add_argument("--out", type=Path, default=Path("figures/shared/eval/grid_eval_out"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    results = run_on_data(args.data_dir, args.n_cones, tuple(args.budgets),
                          args.field, args.out, args.envelope_pct)
    print_table(results)
    print(f"\nGrid evaluation complete: {args.out}")


if __name__ == "__main__":
    main()
