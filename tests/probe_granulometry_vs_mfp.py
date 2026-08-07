"""Does the differentiable granulometry track the real MFP bubble-size distribution?

``losses.GranulometrySpectrum`` is a stand-in for the mean-free-path estimator in
``viz/bubble_size_evaluation.py``, which cannot be differentiated (it thresholds,
takes a first-crossing per ray, and histograms). The stand-in agrees with it at
Pearson r = 0.99 on synthetic discs -- but discs are not reionization
morphology, which is a percolating network of irregular, nested regions.

This measures the agreement on real cubes, per reionization stage, before any
GPU time is spent training against it. Run it on the cluster, where the cube
cache lives:

    python tests/probe_granulometry_vs_mfp.py --cones 8

A weak correlation here does not mean the loss is broken -- it means the
surrogate is optimising something other than what the evaluator reports, which
is the thing worth knowing first.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataset import paths
from dataset.dataset_3d import LightconeCubeCache
from losses import GranulometrySpectrum
from util.metrics_21cm import mean_free_path_samples_2d

# x_HI bands to resolve separately: bubble morphology is not one regime.
STAGES = ((0.05, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.95))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=None, help="cube cache (.h5)")
    parser.add_argument("--cones", type=int, default=8)
    parser.add_argument("--slices-per-stage", type=int, default=6)
    parser.add_argument("--rays", type=int, default=4000)
    parser.add_argument("--box-mpc", type=float, default=200.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--radii", default="1,2,4,8")
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args(argv)

    radii = tuple(int(v) for v in args.radii.split(","))
    gran = GranulometrySpectrum(radii=radii, downsample=args.downsample,
                                max_slices=None)
    scales = torch.tensor([float(r) for r in radii[:-1]])

    cache = LightconeCubeCache(args.cache or paths.CUBES)
    rng = np.random.default_rng(0)

    rows = {stage: ([], []) for stage in STAGES}
    for index in range(min(args.cones, len(cache))):
        truth = cache[index]["y"].numpy()[0]              # (X, Y, Z)
        xhi = truth.mean(axis=(0, 1))
        for stage in STAGES:
            lo, hi = stage
            candidates = np.where((xhi >= lo) & (xhi < hi))[0]
            if candidates.size == 0:
                continue
            chosen = rng.choice(
                candidates, size=min(args.slices_per_stage, candidates.size),
                replace=False,
            )
            for los in chosen:
                plane = truth[:, :, los]
                cell = (args.box_mpc / plane.shape[0],
                        args.box_mpc / plane.shape[1])
                sample = mean_free_path_samples_2d(
                    plane < args.threshold, args.rays, cell,
                    args.box_mpc / 2.0, seed=int(los),
                )
                if sample["distances_mpc"].size < 10:
                    continue
                mfp = float(np.mean(sample["distances_mpc"]))
                # The granulometry measures the *ionized* phase, so feed it
                # the same field the MFP mask selects: 1 where ionized.
                ionized = torch.tensor(1.0 - plane)[None, None]
                spectrum = gran.spectrum(ionized)[0]
                scale = float((spectrum * scales).sum())
                rows[stage][0].append(mfp)
                rows[stage][1].append(scale)

    print(f"\n{'x_HI stage':>14} {'n':>5} {'MFP mean (Mpc)':>15} "
          f"{'granulometry':>13} {'Pearson r':>10} {'Spearman':>9}")
    all_m, all_g = [], []
    for stage, (mfp, gr) in rows.items():
        if len(mfp) < 3:
            print(f"{str(stage):>14} {len(mfp):>5}  too few slices")
            continue
        m, g = np.array(mfp), np.array(gr)
        all_m.extend(m); all_g.extend(g)
        pearson = float(np.corrcoef(m, g)[0, 1])
        spearman = float(np.corrcoef(
            np.argsort(np.argsort(m)), np.argsort(np.argsort(g))
        )[0, 1])
        print(f"{str(stage):>14} {len(m):>5} {m.mean():>15.2f} "
              f"{g.mean():>13.2f} {pearson:>10.3f} {spearman:>9.3f}")
    if len(all_m) >= 3:
        pooled = float(np.corrcoef(all_m, all_g)[0, 1])
        print(f"\npooled Pearson r = {pooled:.3f} over {len(all_m)} slices")
        print("On synthetic discs this was 0.99. A pooled r below ~0.8 means "
              "the surrogate\nand the reported metric disagree on real "
              "morphology -- reconsider before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
