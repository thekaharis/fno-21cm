#!/usr/bin/env python3
"""Visualize FNO predictions: true x_HI vs predicted x_HI.

Loads the latest checkpoint and plots side-by-side comparisons for
train, val, and test splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- Prefer a vendored neuralop checkout if one is available ---------------
# Accept any of these layouts and skip locations whose package is incomplete:
#   ./neuraloperator/neuralop/    (checkout vendored inside the repo)
#   ../neuraloperator/neuralop/   (checkout sibling to the repo: project/{data,
#                                  neuraloperator, fno-21cm} layout)
#   ./neuralop/                   (the package dropped straight into the repo)
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

# Log which neuralop was actually used.
import neuralop as _neuralop
print(f"[visualize] using neuralop from {_neuralop.__file__}")

from dataset import SliceCache, split_by_cone

# ------------------------------------------------------------------ config
CHECKPOINT = "checkpoints/model_state_dict.pt"
CACHE_FILE = Path("trainset.h5")
FIGURES_DIR = Path("figures")
N_SLICES_PER_SPLIT = 4

# Must match fno_21cm.py so val/test here are the same held-out cones.
SPLIT_SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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
def load_model() -> nn.Module:
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = {f"fno.{k}": v for k, v in sd.items() if k != "_metadata"}
    # NOTE: must match the architecture in fno_21cm.py exactly, or the
    # checkpoint will not load correctly.
    fno = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=1,
               out_channels=1, n_layers=4, projection_channel_ratio=2,
               positional_embedding="grid")
    model = SilentFNO(fno)
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()


def predict_batch(model, samples):
    x = torch.stack([s["x"] for s in samples]).to(DEVICE)
    with torch.no_grad():
        preds = model(x=x).cpu().numpy()  # (N, 1, H, W)
    preds = preds.squeeze(1)
    truths = np.stack([s["y"].squeeze(0).numpy() for s in samples])
    inputs = np.stack([s["x"].squeeze(0).numpy() for s in samples])
    return truths, preds, inputs


# ------------------------------------------------------------------ plot
def plot_comparison(truths, preds, inputs, meta):
    """*meta* is a list of dicts with keys 'z', 'file_id' per slice."""
    n = len(meta)
    fig = plt.figure(figsize=(14, 4 * n))
    gs = GridSpec(n, 5, figure=fig,
                  width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.3, wspace=0.3)

    for row in range(n):
        t = truths[row]
        p = preds[row]
        x_in = inputs[row]
        err = p - t
        mse = np.mean(err ** 2)
        info = meta[row]

        # Density input
        ax0 = fig.add_subplot(gs[row, 0])
        im0 = ax0.imshow(x_in, cmap="plasma", origin="lower")
        ax0.set_title(f"Density / 10\nz={info['z']:.2f}  file={info['file_id']}")
        ax0.set_xticks([]); ax0.set_yticks([])

        # True x_HI
        ax1 = fig.add_subplot(gs[row, 1])
        im1 = ax1.imshow(t, cmap="viridis", origin="lower",
                         vmin=0, vmax=1)
        ax1.set_title("True x_HI")
        ax1.set_xticks([]); ax1.set_yticks([])

        # Predicted x_HI
        ax2 = fig.add_subplot(gs[row, 2])
        im2 = ax2.imshow(p, cmap="viridis", origin="lower",
                         vmin=0, vmax=1)
        ax2.set_title("Predicted x_HI")
        ax2.set_xticks([]); ax2.set_yticks([])

        # Error
        ax3 = fig.add_subplot(gs[row, 3])
        vmax_err = max(abs(err.min()), abs(err.max()), 0.01)
        im3 = ax3.imshow(err, cmap="RdBu_r", origin="lower",
                         vmin=-vmax_err, vmax=vmax_err)
        ax3.set_title(f"Pred − True\nMSE = {mse:.4f}")
        ax3.set_xticks([]); ax3.set_yticks([])

        # Colorbar for error
        cax = fig.add_subplot(gs[row, 4])
        plt.colorbar(im3, cax=cax, label="Δ x_HI")

    fig.suptitle(f"FNO Predictions — {info.get('split','')} split  (epoch 90)",
                 fontsize=13, y=0.995)
    return fig


def plot_scatter(truths, preds, split_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    t_flat = truths.ravel()
    p_flat = preds.ravel()

    # Subsample for performance
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
    ax.set_title(f"Scatter — {split_name}")
    ax.legend()

    r2 = np.corrcoef(t_flat, p_flat)[0, 1] ** 2
    rmse = np.sqrt(np.mean((p_flat - t_flat) ** 2))
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    return fig


# ------------------------------------------------------------------ main
def main():
    print(f"Device: {DEVICE}")
    model = load_model()
    print("Model loaded.")

    # Gather data from the same slice cache + cone split used for training.
    if not CACHE_FILE.exists():
        print(f"Slice cache {CACHE_FILE} not found. Run build_trainset.py first.",
              file=sys.stderr)
        sys.exit(1)
    cache = SliceCache(CACHE_FILE)
    _, val_ds, test_ds = split_by_cone(
        cache, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED,
    )
    print(f"Cache: {len(cache)} slices, {len(np.unique(cache.cone_id))} cones")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for split_ds, name in [(val_ds, "Validation"), (test_ds, "Test")]:
        n = len(split_ds)
        if n <= N_SLICES_PER_SPLIT:
            idxs = list(range(n))
        else:
            idxs = np.linspace(0, n - 1, N_SLICES_PER_SPLIT, dtype=int).tolist()

        samples = [split_ds[i] for i in idxs]
        truths, preds, inputs = predict_batch(model, samples)

        # cache.z / cache.cone_id carry the redshift and source cone per slice.
        global_idxs = [split_ds.indices[i] for i in idxs]
        meta = [{"z": float(cache.z[gi]),
                 "file_id": int(cache.cone_id[gi]),
                 "split": name}
                for gi in global_idxs]

        fig = plot_comparison(truths, preds, inputs, meta)
        out = FIGURES_DIR / f"comparison_{name.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        # Scatter plot across many slices
        n_scatter = min(32, n)
        scatter_idxs = np.linspace(0, n - 1, n_scatter, dtype=int).tolist()
        scatter_samples = [split_ds[i] for i in scatter_idxs]
        truths_all, preds_all, _ = predict_batch(model, scatter_samples)
        fig_s = plot_scatter(truths_all, preds_all, name)
        out = FIGURES_DIR / f"scatter_{name.lower()}.png"
        fig_s.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved {out}")

    print("Done.")


if __name__ == "__main__":
    main()
