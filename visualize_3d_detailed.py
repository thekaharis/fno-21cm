#!/usr/bin/env python3
"""Detailed multi-cone viz -- 16 cones x active-z slices per cone.

Variant of ``visualize_3d.py`` with two deliberate differences:

  1. **N_CONES_PER_SPLIT = 16** (vs. 4 in the standard viz).  The 4-cone
     stratified view is enough to see the qualitative behavior across the
     reionization-rate axis, but it's not enough to see whether the model
     handles the *mid-range* cones (neither extremely reionized nor
     extremely neutral) consistently.  16 cones at the same percentile
     spacing gives a much fuller picture.

  2. **Per-cone active-z slice picker** instead of linspace over the full
     LOS range.  In a typical cone the truth-x_HI is fully neutral (yellow)
     above z ~= 9 and fully ionized (purple) below the reionization
     completion epoch -- both regions carry essentially no information
     about the model's bubble-morphology accuracy.  This script ranks the
     LOS slices by transverse spatial variance and picks N slices
     spread evenly through the "active" window where ``std(x_HI) > frac *
     max(std)`` -- so the rendered z-slices land where the model's
     predictions actually have something to be right or wrong about.

Everything else (plot functions, model loader, prediction code, cone
selection by reionization behavior) is reused unchanged from
``visualize_3d``.  Output goes to ``figures/detailed/`` so the standard
4-cone summary plots are not overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse the standard viz machinery.  We override a couple of constants
# (N_CONES_PER_SPLIT) and add the active-z picker; everything else --
# load_model, predict_cube, plot_z_slices, plot_lightcone_strip,
# plot_scatter, plot_lightcone_summary_grid, pick_cones_by_reion_behavior,
# make_run_folder -- comes from visualize_3d unchanged.
import visualize_3d
from visualize_3d import (
    DEVICE, CUBES_CACHE, DATA_DIR, FILE_GLOB,
    N_Z, Z_MIN, Z_MAX, STRATIFY_Z,
    SPLIT_SEED, VAL_FRACTION, TEST_FRACTION,
    INPUT_FEATURES, PARAMETER_NORMALIZATION, RUN_METADATA,
    FIGURES_BASE, VIZ_TAG, make_run_folder,
    load_model, predict_cube,
    plot_z_slices, plot_lightcone_strip, plot_scatter,
    plot_lightcone_summary_grid,
    pick_cones_by_reion_behavior,
)
from dataset_3d import LightconeCubeDataset, LightconeCubeCache, split_cubes

# Pull the CHECKPOINT path lazily so MODEL_KIND env-var changes are still
# honored (visualize_3d evaluates CHECKPOINT at module import time, which
# means MODEL_KIND must be set before this script is imported).
from visualize_3d import CHECKPOINT, MODEL_KIND

# ------------------------------------------------------------------ config
N_CONES_PER_SPLIT = 16             # was 4 in visualize_3d
N_SLICES_PER_CONE = 6              # was 4; with active-z each slice is more
                                   # information-dense, so a few more is fine
ACTIVE_VAR_FRAC = 0.05             # active window = transverse std > 5% of max


# ---------------------------------------------------- active-z slice picker
def pick_active_z_slices(truth: np.ndarray, n_slices: int,
                         frac: float = ACTIVE_VAR_FRAC) -> list[int]:
    """Pick *n_slices* z-indices where the truth has the most spatial structure.

    Computes the transverse standard deviation ``std(truth, axis=(0, 1))``
    for each z-slice, identifies the contiguous "active" window where the
    std exceeds ``frac * max(std)``, then samples *n_slices* z-indices
    evenly through that window.  Slices where the cube is essentially
    constant (fully neutral / fully ionized) are filtered out.

    Edge cases handled:
      * Cone is fully neutral / fully ionized everywhere -> falls back to
        a linspace over the full LOS so we still get some panels.
      * Active window is too narrow for *n_slices* -> takes the top
        *n_slices* z-indices by transverse std.

    Parameters
    ----------
    truth : (Nx, Ny, Nz) ndarray
        Ground-truth x_HI cube for a single cone.
    n_slices : int
        Number of z-slice indices to return.
    frac : float
        Threshold fraction of the per-cone max std used to define "active".

    Returns
    -------
    list of int
        Sorted z-indices, length <= n_slices (usually exactly n_slices).
    """
    per_z_std = truth.std(axis=(0, 1))        # (Nz,)
    max_std = float(per_z_std.max())

    if max_std < 1e-6:
        # Cone is essentially constant.  Just span the LOS so we still get
        # some panels; the reader will see uniform yellow everywhere.
        return np.linspace(0, len(per_z_std) - 1, n_slices,
                           dtype=int).tolist()

    threshold = frac * max_std
    active = np.where(per_z_std > threshold)[0]

    if len(active) < n_slices:
        # Active window is too narrow for n_slices distinct indices --
        # take the top-N by std (still skips the boring slices).
        return sorted(np.argsort(-per_z_std)[:n_slices].tolist())

    # Linspace through [first_active, last_active] -- captures the cone's
    # full reionization history without clustering on the peak-variance
    # epoch alone.
    picks = np.linspace(active[0], active[-1], n_slices, dtype=int)
    return sorted(dict.fromkeys(picks.tolist()))   # dedupe, preserve order


# ------------------------------------------------------------------ main
def main():
    print(f"[visualize_3d_detailed]  MODEL_KIND={MODEL_KIND}  "
          f"N_CONES_PER_SPLIT={N_CONES_PER_SPLIT}  "
          f"N_SLICES_PER_CONE={N_SLICES_PER_CONE}")
    print(f"Device: {DEVICE}")
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found: {CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    if CUBES_CACHE.exists():
        print(f"Using pre-computed cube cache: {CUBES_CACHE}")
        dataset = LightconeCubeCache(
            CUBES_CACHE,
            input_features=INPUT_FEATURES,
        )
    else:
        print(f"No cube cache at {CUBES_CACHE}; streaming raw lightcones "
              f"from {DATA_DIR}")
        files = sorted(DATA_DIR.glob(FILE_GLOB))
        if not files:
            print(f"No lightcone files found under {DATA_DIR}/{FILE_GLOB}",
                  file=sys.stderr)
            sys.exit(1)
        dataset = LightconeCubeDataset(
            file_paths=files, n_z=N_Z, z_min=Z_MIN, z_max=Z_MAX,
            preload=False,
            input_features=INPUT_FEATURES,
        )
    train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx) = split_cubes(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION,
        seed=SPLIT_SEED,
    )
    del train_ds
    normalization = PARAMETER_NORMALIZATION
    if INPUT_FEATURES.use_params and normalization is None:
        if RUN_METADATA is not None:
            raise RuntimeError(
                "run metadata is missing parameter normalization statistics"
            )
        print(
            "WARNING: legacy checkpoint has no run metadata; fitting "
            "train-split parameter statistics for visualization"
        )
        normalization = dataset.fit_parameter_normalization(train_idx)
    dataset.set_parameter_normalization(normalization)

    model = load_model(in_channels=dataset.in_channels)
    print("Model loaded.")

    target_z = dataset.target_z
    # Unique per-run output folder.  Tag is "<VIZ_TAG>-detailed" so the
    # detailed variant is visibly different from the standard 4-cone run
    # in the figures/ listing.  VIZ_TAG defaults to MODEL_KIND for v1/v2
    # ("fno", "ufno"); v3 sbatches set it to e.g. "ufno-v3-anisoz" so the
    # render lands in ``figures/ufno-v3-anisoz-detailed_<timestamp>_job.../``.
    figures_dir = make_run_folder(FIGURES_BASE, tag=f"{VIZ_TAG}-detailed")
    print(f"Writing figures to: {figures_dir}")

    for split_ds, split_idx, split_name in [
        (val_ds, val_idx, "validation"),
        (test_ds, test_idx, "test"),
    ]:
        if len(split_ds) == 0:
            print(f"No cones in {split_name} split; skipping")
            continue

        print(f"--- {split_name}: picking {N_CONES_PER_SPLIT} cones "
              f"by reionization behavior at z={STRATIFY_Z} ---")
        picks = pick_cones_by_reion_behavior(
            split_ds, split_idx, target_z,
            n_cones=N_CONES_PER_SPLIT, stratify_z=STRATIFY_Z,
        )
        for idx_in_split, cone_id, summ in picks:
            print(f"  cone {cone_id:4d}  <x_HI>(z={STRATIFY_Z}) = {summ:.3f}")

        # Per-cone figures + accumulator for the summary grid.
        per_cone_for_grid: list = []
        for idx_in_split, cone_id, summ in picks:
            print(f"--- {split_name} cone {cone_id} "
                  f"(idx_in_split={idx_in_split}, <x_HI>={summ:.3f}) ---")
            sample = split_ds[idx_in_split]
            dens, truth, pred = predict_cube(model, sample)
            per_cone_for_grid.append((cone_id, summ, dens, truth, pred))

            # Active-z slice picker (the headline change vs visualize_3d).
            active_idxs = pick_active_z_slices(truth, N_SLICES_PER_CONE)
            print(f"  active-z slices (z indices): {active_idxs}")
            print(f"  -> z values: "
                  f"{[float(f'{target_z[i]:.2f}') for i in active_idxs]}")

            fig = plot_z_slices(dens, truth, pred, target_z, active_idxs,
                                cone_id, split_name)
            out = figures_dir / f"comparison_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

            # The xz lightcone strip is global by construction (it shows the
            # full LOS), so no active-z analog needed.  Same plot function
            # as visualize_3d.
            fig = plot_lightcone_strip(dens, truth, pred, target_z, cone_id,
                                       split_name)
            out = figures_dir / f"lightcone_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

            # Scatter is a voxel histogram over the whole cube -- volume-
            # weighted by the easy regions; not very different per-cone
            # at 16 cones.  Keep for completeness.
            fig = plot_scatter(truth, pred, cone_id, split_name)
            out = figures_dir / f"scatter_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

        # Summary grid: 16-row lightcone-strip stack.  Tall but readable;
        # the canonical "compare across 16 reionization regimes" figure.
        if per_cone_for_grid:
            fig = plot_lightcone_summary_grid(per_cone_for_grid, target_z,
                                              split_name)
            out = figures_dir / f"lightcone_grid_3d_{split_name}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")

    print("Done.")


if __name__ == "__main__":
    main()
