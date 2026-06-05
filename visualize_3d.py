#!/usr/bin/env python3
"""Visualize 3-D FNO predictions on full 21cm lightcone cubes.

For each held-out cone:
  1. Comparison panel of N evenly-spaced z-slices through the predicted cube.
  2. Edge-on xz lightcone strip at y = Ny // 2 (full LOS extent).
  3. Hexbin scatter of true vs predicted x_HI across all voxels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- Prefer a vendored neuralop checkout if one is available ---------------
# Search order:
#   ./neuraloperator/         (checkout vendored inside the repo)
#   ../neuraloperator/        (checkout sibling to the repo: project/{data,
#                              neuraloperator, fno-21cm} layout)
#   ./                        (neuralop dropped straight into the repo)
# If none has a valid __init__.py, fall back to an installed `neuralop`.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "neuraloperator",
              _HERE.parent / "neuraloperator",
              _HERE):
    if (_cand / "neuralop" / "__init__.py").is_file():
        sys.path.insert(0, str(_cand))
        break
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from neuralop.models import FNO

import neuralop as _neuralop
print(f"[visualize_3d] using neuralop from {_neuralop.__file__}")

from dataset_3d import LightconeCubeDataset, LightconeCubeCache, split_cubes

# ------------------------------------------------------------------ config
CHECKPOINT = "checkpoints_3d/model_state_dict.pt"
FIGURES_DIR = Path("figures")
# Data source: prefer the pre-built cube cache if it exists, otherwise stream
# from raw lightcones.  Must match what training used so the deterministic
# split (driven by len(dataset) + SPLIT_SEED) lines up.
CUBES_CACHE = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"

N_Z = 256
Z_MIN, Z_MAX = 5.0, 25.0
N_SLICES_PER_CONE = 4              # z-slices to render in the comparison panel

# Must match fno_21cm_3d.py.
SPLIT_SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


# ------------------------------------------------------------------ wrapper
class SilentFNO(nn.Module):
    def __init__(self, fno):
        super().__init__()
        self.fno = fno

    def forward(self, x, **kwargs):
        return self.fno(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        return getattr(self._modules["fno"], name)


# ------------------------------------------------------------------ helpers
def load_model(in_channels: int = 2) -> nn.Module:
    """Reconstruct the FNO architecture and load the latest checkpoint.

    ``in_channels`` must match the cache used at training time -- pass
    ``dataset.in_channels`` from the caller so parameter-conditioned runs
    (where in_channels=13) load the correct lifting layer.
    """
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = {f"fno.{k}": v for k, v in sd.items() if k != "_metadata"}
    # Must match fno_21cm_3d.py architecture exactly.
    fno = FNO(n_modes=(16, 16, 16), hidden_channels=32, in_channels=in_channels,
              out_channels=1, n_layers=4, projection_channel_ratio=2,
              positional_embedding="grid")
    model = SilentFNO(fno)
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()


def predict_cube(model, sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on one cube.  Returns (input_density, true_xhi, pred_xhi)."""
    x = sample["x"].unsqueeze(0).to(DEVICE)        # (1, 2, Nx, Ny, Nz)
    with torch.no_grad():
        pred = model(x=x).cpu().numpy()[0, 0]      # (Nx, Ny, Nz)
    dens = sample["x"][0].numpy()                  # density / 10 channel
    truth = sample["y"][0].numpy()                 # (Nx, Ny, Nz)
    return dens, truth, pred


# --------------------------------------------------------- z-slice panel
def plot_z_slices(dens, truth, pred, target_z, idxs, cone_id, split):
    n = len(idxs)
    fig = plt.figure(figsize=(14, 4 * n))
    gs = GridSpec(n, 5, figure=fig,
                  width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.3, wspace=0.3)

    for row, k in enumerate(idxs):
        d = dens[:, :, k]
        t = truth[:, :, k]
        p = pred[:, :, k]
        err = p - t
        mse = float(np.mean(err ** 2))
        z = float(target_z[k])

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(d, cmap="plasma", origin="lower")
        ax0.set_title(f"Density / 10\nz={z:.2f}  cone={cone_id}")
        ax0.set_xticks([]); ax0.set_yticks([])

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(t, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax1.set_title("True x_HI")
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(p, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax2.set_title("Predicted x_HI")
        ax2.set_xticks([]); ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[row, 3])
        vmax_err = max(abs(err.min()), abs(err.max()), 0.01)
        im3 = ax3.imshow(err, cmap="RdBu_r", origin="lower",
                         vmin=-vmax_err, vmax=vmax_err)
        ax3.set_title(f"Pred - True\nMSE = {mse:.4f}")
        ax3.set_xticks([]); ax3.set_yticks([])

        cax = fig.add_subplot(gs[row, 4])
        plt.colorbar(im3, cax=cax, label="dx_HI")

    fig.suptitle(f"FNO 3-D predictions ({split}, cone {cone_id})",
                 fontsize=13, y=0.995)
    return fig


# ----------------------------------------------------- xz lightcone strip
def plot_lightcone_strip(dens, truth, pred, target_z, cone_id, split):
    """Edge-on xz panel at y = Ny // 2 spanning the full LOS extent."""
    ny = dens.shape[1]
    j = ny // 2
    d_strip = dens[:, j, :]
    t_strip = truth[:, j, :]
    p_strip = pred[:, j, :]
    e_strip = p_strip - t_strip

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    extent = [float(target_z[0]), float(target_z[-1]), 0, d_strip.shape[0]]

    axes[0].imshow(d_strip, cmap="plasma", aspect="auto",
                   origin="lower", extent=extent)
    axes[0].set_title(f"Density / 10  (y = Ny/2)   cone {cone_id}, {split}")
    axes[0].set_ylabel("x cell")

    axes[1].imshow(t_strip, cmap="viridis", aspect="auto",
                   origin="lower", extent=extent, vmin=0, vmax=1)
    axes[1].set_title("True x_HI")
    axes[1].set_ylabel("x cell")

    axes[2].imshow(p_strip, cmap="viridis", aspect="auto",
                   origin="lower", extent=extent, vmin=0, vmax=1)
    axes[2].set_title("Predicted x_HI")
    axes[2].set_ylabel("x cell")

    vmax_err = max(abs(e_strip.min()), abs(e_strip.max()), 0.01)
    axes[3].imshow(e_strip, cmap="RdBu_r", aspect="auto",
                   origin="lower", extent=extent,
                   vmin=-vmax_err, vmax=vmax_err)
    axes[3].set_title("Pred - True")
    axes[3].set_xlabel("redshift z")
    axes[3].set_ylabel("x cell")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------- scatter
def plot_scatter(truth, pred, cone_id, split):
    fig, ax = plt.subplots(figsize=(5, 5))
    t_flat = truth.ravel()
    p_flat = pred.ravel()

    max_pts = 50_000
    if len(t_flat) > max_pts:
        idx = np.random.default_rng(42).choice(len(t_flat), max_pts,
                                               replace=False)
        t_flat = t_flat[idx]
        p_flat = p_flat[idx]

    ax.hexbin(t_flat, p_flat, gridsize=100, cmap="Blues",
              mincnt=1, bins="log")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="perfect")
    ax.set_xlabel("True x_HI")
    ax.set_ylabel("Predicted x_HI")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Scatter - {split} (cone {cone_id})")
    ax.legend()

    r2 = float(np.corrcoef(t_flat, p_flat)[0, 1] ** 2)
    rmse = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))
    ax.text(0.05, 0.95, f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    return fig


# ------------------------------------------------------------------ main
def main():
    print(f"Device: {DEVICE}")
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found: {CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    if CUBES_CACHE.exists():
        print(f"Using pre-computed cube cache: {CUBES_CACHE}")
        dataset = LightconeCubeCache(CUBES_CACHE)
    else:
        print(f"No cube cache at {CUBES_CACHE}; streaming raw lightcones "
              f"from {DATA_DIR}")
        files = sorted(DATA_DIR.glob(FILE_GLOB))
        if not files:
            print(f"No lightcone files found under {DATA_DIR}/{FILE_GLOB}",
                  file=sys.stderr)
            sys.exit(1)
        dataset = LightconeCubeDataset(
            file_paths=files,
            n_z=N_Z, z_min=Z_MIN, z_max=Z_MAX,
            preload=False,
        )
    _, val_ds, test_ds, (_, val_idx, test_idx) = split_cubes(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED,
    )
    print(f"Val cones: {val_idx}")
    print(f"Test cones: {test_idx}")

    model = load_model(in_channels=getattr(dataset, "in_channels", 2))
    print("Model loaded.")

    target_z = dataset.target_z
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for split_ds, split_idx, split_name in [
        (val_ds, val_idx, "validation"),
        (test_ds, test_idx, "test"),
    ]:
        if len(split_ds) == 0:
            print(f"No cones in {split_name} split; skipping")
            continue

        # Use the first cone in each split for the per-cone visuals.
        sample = split_ds[0]
        cone_id = split_idx[0]
        dens, truth, pred = predict_cube(model, sample)

        # z-slice grid
        idxs = np.linspace(0, N_Z - 1, N_SLICES_PER_CONE, dtype=int).tolist()
        fig = plot_z_slices(dens, truth, pred, target_z, idxs, cone_id,
                            split_name)
        out = FIGURES_DIR / f"comparison_3d_{split_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        # xz lightcone strip
        fig = plot_lightcone_strip(dens, truth, pred, target_z, cone_id,
                                   split_name)
        out = FIGURES_DIR / f"lightcone_3d_{split_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        # scatter over all voxels in the cube
        fig = plot_scatter(truth, pred, cone_id, split_name)
        out = FIGURES_DIR / f"scatter_3d_{split_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    print("Done.")


if __name__ == "__main__":
    main()
