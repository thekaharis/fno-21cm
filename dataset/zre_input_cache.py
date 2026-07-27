"""Sidecar cache of z_re-task *inputs*: interpolated density slices + params.

The z_re pipeline's targets are cached in ``zre_targets.h5``, but the inputs
were re-read from all 6600 raw ~1.3 GB lightcone files at every training
start (~1.5-2 h of Lustre I/O).  This module builds a single HDF5 sidecar
holding, per cone, the ``(n_z_in, Nx, Ny)`` float32 density block already
interpolated to the training z-grid (UNSCALED -- ``ZreMapDataset`` applies
``density_scale`` at load time) plus the 11 sampled parameters, so a run
opens one file instead of 6600.

Layout::

    /density/<stem>   (n_z_in, Nx, Ny) float32
    /params/<stem>    (n_params,) float32
    attrs: n_z_in, z_min, z_max   (validated by the loader)

Build (idempotent per cone; a killed build resumes where it stopped)::

    python -m dataset.zre_input_cache --data /path/to/lightcones \
        --out zre_inputs.h5 --n-z-in 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from dataset.lightcone_params import read_sampled_params
from dataset import paths
from dataset.loader import LightconeFile


def build_input_cache(
    file_paths: Sequence[str | Path],
    cache_path: str | Path,
    n_z_in: int = 64,
    z_min: float = 5.0,
    z_max: float = 25.0,
    verbose: bool = True,
) -> Path:
    """Interpolate-and-cache density slices + params for every cone."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    input_z = np.linspace(z_min, z_max, n_z_in, dtype=np.float64)
    with h5py.File(cache_path, "a") as cache:
        for key, value in (("n_z_in", n_z_in), ("z_min", z_min),
                           ("z_max", z_max)):
            existing = cache.attrs.get(key)
            if existing is None:
                cache.attrs[key] = value
            elif float(existing) != float(value):
                raise ValueError(
                    f"{cache_path} was built with {key}={existing}, "
                    f"requested {value}; delete the cache to rebuild."
                )
        density = cache.require_group("density")
        params = cache.require_group("params")
        for path in file_paths:
            stem = Path(path).stem
            if stem in density and stem in params:
                continue
            if verbose:
                print(f"[zre_input_cache] caching {stem} ...", flush=True)
            with LightconeFile(path) as lf:
                dens = lf.read_interpolated("density", input_z)
            dens = np.ascontiguousarray(
                np.moveaxis(dens, -1, 0).astype(np.float32)
            )
            with h5py.File(path, "r") as h5_file:
                row = np.asarray(read_sampled_params(h5_file),
                                 dtype=np.float32)
            if stem in density:
                del density[stem]
            if stem in params:
                del params[stem]
            density.create_dataset(stem, data=dens)
            params.create_dataset(stem, data=row)
    return cache_path


def validate_cache_attrs(cache: h5py.File, n_z_in: int,
                         z_min: float, z_max: float) -> None:
    """Raise if the cache was built on a different z-grid than requested."""
    got = (int(cache.attrs.get("n_z_in", -1)),
           float(cache.attrs.get("z_min", float("nan"))),
           float(cache.attrs.get("z_max", float("nan"))))
    want = (int(n_z_in), float(z_min), float(z_max))
    if got != want:
        raise ValueError(
            f"input cache grid mismatch: cache has (n_z_in, z_min, z_max)="
            f"{got}, dataset wants {want}; rebuild the cache."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True,
                        help="directory containing 21cmfast_11d_sample*.h5")
    parser.add_argument("--out", type=Path, default=paths.ZRE_INPUTS)
    parser.add_argument("--n-z-in", type=int, default=64)
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=25.0)
    args = parser.parse_args()

    files = sorted(args.data.glob("21cmfast_11d_sample*.h5"))
    if not files:
        raise SystemExit(f"no lightcone files found in {args.data}")
    build_input_cache(files, args.out, n_z_in=args.n_z_in,
                      z_min=args.z_min, z_max=args.z_max)
    print(f"[zre_input_cache] cache complete: {args.out} ({len(files)} cones)")


if __name__ == "__main__":
    main()
