"""Per-pixel z_re(x, y) target maps computed from lightcone neutral fraction.

Two target definitions, both fit per sky pixel by least squares against the
full x_HI(z) sightline:

* ``gompertz`` -- two-parameter front ``x_HI = exp(-exp(-(z - z0)/dz))`` over
  a (z0, dz) grid; the stored z_re is the midpoint where the fit crosses
  x_HI = 0.5 (``z_half = z0 - ln(ln 2) * dz``). This is the smoothest map and
  the recommended training target.
* ``step`` -- one-parameter step ``x_HI = Theta(z - z_re)`` with the exact
  least-squares threshold.

Pixels whose fitted transition lies outside the cone (midpoint below the
low-z edge, or no transition at all) are stored as NaN; the dataset decides
how to present them to the model (see ``dataset/dataset_zre.py``).

Fitting costs a few seconds per cone, so maps are cached in a small sidecar
HDF5 (one dataset per cone, keyed by file stem). Build it once with::

    python -m dataset.zre_target --data <lightcone dir> --out zre_targets.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

TARGET_KINDS = ("gompertz", "step")

# Fit grid; matches the analysis notebooks. z0 spans below the cone edge so
# partially-started fronts can place their midpoint outside the volume.
Z0_GRID = np.arange(2.5, 18.001, 0.1)
DZ_GRID = np.geomspace(0.02, 4.0, 18)
_LNLN2 = float(np.log(np.log(2.0)))


def _read_xhi(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        xhi = f["lightcone/neutral_fraction"][:]
        z = np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float64)
    if z[0] > z[-1]:
        z, xhi = z[::-1], xhi[:, :, ::-1]
    return xhi, z


def fit_step_zre(xhi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Exact least-squares step threshold per pixel; NaN when degenerate."""
    x2 = xhi.astype(np.float32) ** 2
    om2 = (1.0 - xhi.astype(np.float32)) ** 2
    zeros = np.zeros_like(x2[:, :, :1])
    left = np.concatenate([zeros, np.cumsum(x2, axis=2)], axis=2)
    right = om2.sum(axis=2, keepdims=True) - np.concatenate(
        [zeros, np.cumsum(om2, axis=2)], axis=2
    )
    t = np.argmin(left + right, axis=2)
    zmid = 0.5 * (z[:-1] + z[1:])
    zre = np.full(xhi.shape[:2], np.nan)
    inner = (t > 0) & (t < xhi.shape[2])
    zre[inner] = zmid[t[inner] - 1]
    return zre


def fit_gompertz_zre(xhi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Gompertz-front midpoint per pixel; NaN when the midpoint leaves the cone."""
    g_z0, g_dz = (g.ravel() for g in np.meshgrid(Z0_GRID, DZ_GRID, indexing="ij"))
    t = (z[None, :] - g_z0[:, None]) / g_dz[:, None]
    profiles = np.exp(-np.exp(-np.clip(t, -60, 60))).astype(np.float32)

    n_los = z.size
    flat = xhi.reshape(-1, n_los).T.astype(np.float32)      # (n_los, npix)
    cost = (
        (profiles**2).sum(axis=1)[:, None]
        - 2.0 * (profiles @ flat)
        + (flat**2).sum(axis=0)[None, :]
    )
    best = cost.argmin(axis=0)
    zre = (g_z0[best] - _LNLN2 * g_dz[best]).reshape(xhi.shape[:2])
    zre[zre < z.min()] = np.nan
    zre[zre > z.max()] = np.nan
    return zre


def compute_zre_map(path: str | Path, kind: str = "gompertz") -> np.ndarray:
    if kind not in TARGET_KINDS:
        raise ValueError(f"kind must be one of {TARGET_KINDS}, got {kind!r}")
    xhi, z = _read_xhi(path)
    fit = fit_gompertz_zre if kind == "gompertz" else fit_step_zre
    return fit(xhi, z).astype(np.float32)


def build_target_cache(
    file_paths: Sequence[str | Path],
    cache_path: str | Path,
    kind: str = "gompertz",
    verbose: bool = True,
) -> Path:
    """Compute-and-cache z_re maps for every cone; idempotent per file stem."""
    cache_path = Path(cache_path)
    # Fast read-only path: a complete cache needs no writable open, so
    # concurrent jobs don't fight over HDF5's exclusive write lock.
    if cache_path.exists():
        try:
            with h5py.File(cache_path, "r") as cache:
                group = cache.get(kind)
                if group is not None and all(
                    Path(p).stem in group for p in file_paths
                ):
                    return cache_path
        except OSError:
            pass  # unreadable/locked: fall through to the writable open
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "a") as cache:
        group = cache.require_group(kind)
        for path in file_paths:
            stem = Path(path).stem
            if stem in group:
                continue
            if verbose:
                print(f"[zre_target] fitting {kind} z_re for {stem} ...")
            zre = compute_zre_map(path, kind)
            ds = group.create_dataset(stem, data=zre, compression="gzip")
            ds.attrs["source_file"] = str(path)
            ds.attrs["nan_fraction"] = float(np.isnan(zre).mean())
    return cache_path


def load_zre_map(
    cache_path: str | Path, file_path: str | Path, kind: str = "gompertz"
) -> np.ndarray:
    with h5py.File(cache_path, "r") as cache:
        return np.asarray(cache[kind][Path(file_path).stem], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True,
                        help="directory containing 21cmfast_11d_sample*.h5")
    parser.add_argument("--out", type=Path, default=Path("zre_targets.h5"))
    parser.add_argument("--kind", choices=TARGET_KINDS, default="gompertz")
    args = parser.parse_args()

    files = sorted(args.data.glob("21cmfast_11d_sample*.h5"))
    if not files:
        raise SystemExit(f"no lightcone files found in {args.data}")
    build_target_cache(files, args.out, kind=args.kind)
    print(f"[zre_target] cache complete: {args.out} ({len(files)} cones)")


if __name__ == "__main__":
    main()
