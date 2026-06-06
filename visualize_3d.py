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

# Pull the architecture constants from the training script so the two stay in
# sync automatically -- if you bump HIDDEN_CHANNELS or N_MODES there, viz
# loads the matching checkpoint without a second edit here.
from fno_21cm_3d import (
    N_MODES, HIDDEN_CHANNELS, N_LAYERS,
    UFNO_WIDTH, UFNO_NORM, UFNO_UNET_VARIANT, UFNO_GLOBAL_RESIDUAL,
    MODEL_KIND,
)

# ------------------------------------------------------------------ config
# Default checkpoint path mirrors the training script's MODEL_KIND-aware
# CHECKPOINT_DIR so viz auto-loads from the matching directory.
_DEFAULT_CKPT = ("checkpoints_3d" if MODEL_KIND == "fno"
                 else "checkpoints_3d_ufno")
CHECKPOINT = os.environ.get("CHECKPOINT",
                            f"{_DEFAULT_CKPT}/model_state_dict.pt")

# Base directory for all viz outputs.  Each call to main() creates a fresh
# uniquely-named subfolder under this base (see make_run_folder), so
# successive viz runs never overwrite each other -- useful for comparing
# checkpoints at different training epochs, model variants, or simply
# keeping an archive of every render.
FIGURES_BASE = Path("figures")
# Data source: prefer the pre-built cube cache if it exists, otherwise stream
# from raw lightcones.  Must match what training used so the deterministic
# split (driven by len(dataset) + SPLIT_SEED) lines up.
CUBES_CACHE = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"

N_Z = 256
Z_MIN, Z_MAX = 5.0, 25.0
N_SLICES_PER_CONE = 4              # z-slices to render in the comparison panel

# Number of cones to visualize per held-out split (val + test).  Cones are
# picked to span the range of reionization behaviors -- from "barely reionized
# by z=5" to "fully reionized early" -- by ranking the held-out cones on their
# mean truth x_HI at z = STRATIFY_Z and picking at evenly spaced percentiles.
# A single representative cone (the old behavior) is a poor diagnostic because
# LHS parameter draws produce wildly different reionization histories; the
# multi-cone view is what makes architectural / loss interventions actually
# comparable.
N_CONES_PER_SPLIT = 4
STRATIFY_Z = 7.0                   # mid-reionization redshift used for ranking

# Must match fno_21cm_3d.py.
SPLIT_SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


# ------------------------------------------------- per-run output folder
def make_run_folder(base: Path = FIGURES_BASE, tag: str = "") -> Path:
    """Create a uniquely-named subfolder of *base* for one viz run.

    Folder name: ``{tag}_{timestamp}[_job{SLURM_JOB_ID}]``, where:
      * tag is "fno", "ufno", "fno-detailed", "ufno-detailed", ... -- the
        caller passes whatever identifies the variant
      * timestamp is the local time the run started (YYYYMMDD-HHMMSS)
      * job id is appended when running under SLURM so cluster runs are
        easy to correlate with sbatch log files

    Also writes a ``run_info.txt`` inside the folder summarising the run
    config -- helpful months later when you find a folder full of PNGs
    and want to remember which model / checkpoint produced them.

    Example folder names:
      ``figures/ufno_20260606-143022_job3965704/``
      ``figures/fno-detailed_20260606-145501/``        (no SLURM)
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [p for p in (tag, ts) if p]
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        parts.append(f"job{job_id}")
    folder = base / "_".join(parts)
    folder.mkdir(parents=True, exist_ok=True)

    # Drop a small breadcrumb so old folders are self-explanatory.
    info_lines = [
        f"tag:          {tag or '(unset)'}",
        f"timestamp:    {ts}",
        f"job_id:       {job_id or '(local, no SLURM)'}",
        f"MODEL_KIND:   {MODEL_KIND}",
        f"CHECKPOINT:   {CHECKPOINT}",
        f"CUBES_CACHE:  {CUBES_CACHE}",
        f"N_MODES:      {N_MODES}",
        f"HIDDEN_CHAN:  {HIDDEN_CHANNELS}",
        f"UFNO_WIDTH:   {UFNO_WIDTH}",
        f"UFNO_NORM:    {UFNO_NORM}",
        f"UFNO_UNET:    {UFNO_UNET_VARIANT}"
        + ("+global_residual" if UFNO_GLOBAL_RESIDUAL else ""),
        f"N_LAYERS:     {N_LAYERS}",
    ]
    (folder / "run_info.txt").write_text("\n".join(info_lines) + "\n")
    return folder


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


def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix):] if k.startswith(prefix) else k


# ------------------------------------------------------------------ helpers
def load_model(in_channels: int = 2) -> nn.Module:
    """Reconstruct the FNO architecture and load the latest checkpoint.

    Robust to multiple checkpoint formats:
      * single-GPU, raw FNO state dict: keys like ``lifting.fcs.0.weight``
      * single-GPU, SilentFNO state dict: ``fno.lifting.fcs.0.weight``
      * DDP-wrapped SilentFNO: ``module.fno.lifting.fcs.0.weight``

    Tries each transform, picks the one that matches the most target keys,
    and *fails loudly* if no keys match (catches the silent-random-init bug
    that ``strict=False`` was hiding).

    ``in_channels`` must match the cache used at training time -- pass
    ``dataset.in_channels`` from the caller so parameter-conditioned runs
    (where in_channels=13) load the correct lifting layer.
    """
    raw_sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    raw_sd = {k: v for k, v in raw_sd.items() if k != "_metadata"}

    if MODEL_KIND == "ufno":
        # U-FNO: SilentFNO wraps UFNOWrapped, which wraps Wen et al.'s
        # SimpleBlock3d.  Same in/out conventions; same prefix-detection
        # logic below handles the DDP module. prefix.
        from models_ufno import UFNOWrapped
        fno = UFNOWrapped(
            modes1=N_MODES[0], modes2=N_MODES[1], modes3=N_MODES[2],
            width=UFNO_WIDTH,
            in_channels=in_channels,
            out_channels=1,
            sigmoid=True,
            norm=UFNO_NORM,                     # must match training time
            unet_variant=UFNO_UNET_VARIANT,     # must match training time
            global_residual=UFNO_GLOBAL_RESIDUAL,
        )
    else:
        fno = FNO(n_modes=N_MODES, hidden_channels=HIDDEN_CHANNELS,
                  in_channels=in_channels, out_channels=1, n_layers=N_LAYERS,
                  projection_channel_ratio=2, positional_embedding="grid")
    model = SilentFNO(fno)
    target_keys = set(model.state_dict().keys())

    candidates = [
        ("as-is",
            raw_sd),
        ("add fno.",
            {f"fno.{k}": v for k, v in raw_sd.items()}),
        ("strip module.",
            {_strip_prefix(k, "module."): v for k, v in raw_sd.items()}),
        ("strip module. + add fno.",
            {f"fno.{_strip_prefix(k, 'module.')}": v for k, v in raw_sd.items()}),
    ]

    def _matches(d: dict) -> int:
        # A key matches only if the name is present AND the tensor shape
        # agrees with the model's parameter at that key.
        n = 0
        target_sd = model.state_dict()
        for k, v in d.items():
            if k in target_sd and target_sd[k].shape == v.shape:
                n += 1
        return n

    best_name, best_sd = max(candidates, key=lambda c: _matches(c[1]))
    n_matched = _matches(best_sd)
    n_target = len(target_keys)

    print(f"[load_model] checkpoint keys: {len(raw_sd)}; "
          f"best transform: {best_name!r}  "
          f"matched {n_matched}/{n_target} model params "
          f"(in_channels={in_channels})")

    if n_matched == 0:
        raise RuntimeError(
            f"No checkpoint keys match the model after trying all transforms. "
            f"Sample raw key: {next(iter(raw_sd))!r}; "
            f"sample target key: {next(iter(target_keys))!r}. "
            f"Architecture mismatch -- the model was trained with a different "
            f"in_channels/N_MODES/HIDDEN_CHANNELS than what viz is constructing."
        )

    # strict=False here because the checkpoint may contain optimizer/scheduler
    # state under arbitrary extra keys -- those should be ignored.  But the
    # n_matched check above guarantees we're not silently loading nothing.
    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    if missing:
        print(f"[load_model] WARNING: {len(missing)} parameters left at random "
              f"init: {sorted(missing)[:3]}...")
    return model.to(DEVICE).eval()


def predict_cube(model, sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on one cube.  Returns (input_density, true_xhi, pred_xhi).

    Also prints per-cube prediction stats so the SLURM log shows at a glance
    whether the output is sensible (non-constant, in [0, 1]-ish range) without
    needing to open the PNGs.
    """
    x = sample["x"].unsqueeze(0).to(DEVICE)        # (1, C, Nx, Ny, Nz)
    with torch.no_grad():
        pred = model(x=x).cpu().numpy()[0, 0]      # (Nx, Ny, Nz)
    dens = sample["x"][0].numpy()                  # density / 10 channel
    truth = sample["y"][0].numpy()                 # (Nx, Ny, Nz)

    # Quick numerical sanity: a degenerate constant prediction will show as
    # std ~ 0, while a trained model has std around 0.1-0.5 for x_HI in [0, 1].
    print(f"  pred   min/mean/max/std = "
          f"{pred.min():+.3f} / {pred.mean():+.3f} / {pred.max():+.3f} / "
          f"{pred.std():.3f}")
    print(f"  truth  min/mean/max/std = "
          f"{truth.min():+.3f} / {truth.mean():+.3f} / {truth.max():+.3f} / "
          f"{truth.std():.3f}")
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


# ---------------------------------------------------- cone-picker by behavior
def pick_cones_by_reion_behavior(split_ds, split_idx, target_z,
                                 n_cones: int,
                                 stratify_z: float) -> list[tuple[int, int, float]]:
    """Pick *n_cones* cones from the split that span the reionization range.

    Ranks every cone in the split by its mean truth-x_HI at the LOS slice
    closest to *stratify_z* (a single 2-D slice per cone, cheap to read),
    then samples at evenly spaced percentiles of that ranking.  Returns a
    list of ``(idx_in_split, global_cone_id, summary_xhi)`` tuples,
    ordered from most-reionized (low summary) to least (high summary).

    Picking a single "first cone" -- the previous behavior -- gave wildly
    different visuals run to run because LHS-sampled parameter draws produce
    very different reionization histories.  Stratifying across cones is what
    makes architectural / loss interventions diagnosable: the worst-case
    "no reionization" cone is exactly where the model's parameter-conditioning
    has to do the most work, and the typical mid-reion. cone is where bubble
    walls show the FNO's spectral-truncation effects most clearly.
    """
    n = len(split_ds)
    if n == 0:
        return []

    # Mean truth-x_HI at the slice closest to stratify_z.  One number per
    # cone, fast (single 2-D slice per cube).  Used for both ranking and
    # for the summary annotation in the rendered figures.
    z_idx = int(np.argmin(np.abs(target_z - stratify_z)))
    summaries = np.empty(n, dtype=np.float64)
    for i in range(n):
        summaries[i] = split_ds[i]["y"][0, :, :, z_idx].mean().item()

    if n_cones >= n:
        order = list(range(n))
    else:
        # Sample at evenly spaced percentiles (5..95 by default) so the
        # extremes are represented without being literal min/max outliers.
        percentiles = np.linspace(5.0, 95.0, n_cones)
        targets = np.percentile(summaries, percentiles)
        order = []
        for t in targets:
            order.append(int(np.argmin(np.abs(summaries - t))))
        # Dedupe while preserving order (np.unique would re-sort).
        order = list(dict.fromkeys(order))
        # If dedupe shrunk the list, top up with the next-closest cones.
        while len(order) < n_cones and len(order) < n:
            for k in range(n):
                if k not in order:
                    order.append(k); break

    # Sort by summary x_HI so the rendered figures step cleanly from
    # most-reionized (low <x_HI>) to least (high).
    order.sort(key=lambda i: summaries[i])
    return [(i, int(split_idx[i]), float(summaries[i])) for i in order]


# ------------------------------------------------- multi-cone summary plot
def plot_lightcone_summary_grid(per_cone, target_z, split_name):
    """One xz lightcone strip per cone, stacked vertically.

    *per_cone* is a list of ``(cone_id, summary_xhi, dens, truth, pred)``.
    Each row shows three panels (True | Pred | Pred - True) at y = Ny/2 for
    that cone, with the cone's mean truth-x_HI at z = STRATIFY_Z in the row
    label so the reionization "level" is visible at a glance.

    This is the single most useful comparison figure for the thesis: one
    image shows how the same model handles cones with qualitatively different
    reionization histories.
    """
    n = len(per_cone)
    fig, axes = plt.subplots(n, 3, figsize=(18, 2.0 * n + 1), squeeze=False)
    extent = [float(target_z[0]), float(target_z[-1]), 0, per_cone[0][2].shape[0]]

    for row, (cone_id, summ, dens, truth, pred) in enumerate(per_cone):
        j = truth.shape[1] // 2
        t_strip = truth[:, j, :]
        p_strip = pred[:, j, :]
        e_strip = p_strip - t_strip

        axes[row, 0].imshow(t_strip, cmap="viridis", aspect="auto",
                            origin="lower", extent=extent, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"cone {cone_id}\n<x_HI>={summ:.2f}",
                                fontsize=9)

        axes[row, 1].imshow(p_strip, cmap="viridis", aspect="auto",
                            origin="lower", extent=extent, vmin=0, vmax=1)

        vmax_err = max(abs(e_strip.min()), abs(e_strip.max()), 0.01)
        axes[row, 2].imshow(e_strip, cmap="RdBu_r", aspect="auto",
                            origin="lower", extent=extent,
                            vmin=-vmax_err, vmax=vmax_err)

        if row == 0:
            axes[row, 0].set_title("True x_HI")
            axes[row, 1].set_title("Predicted x_HI")
            axes[row, 2].set_title("Pred - True")
        if row == n - 1:
            for c in range(3):
                axes[row, c].set_xlabel("redshift z")
        else:
            for c in range(3):
                axes[row, c].set_xticklabels([])

    fig.suptitle(f"Lightcone-strip grid across {n} reionization regimes  "
                 f"({split_name})", y=1.0, fontsize=11)
    fig.tight_layout()
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
    # Create a unique per-run output folder so this render never overwrites
    # a prior viz job's output.  Tag is just the model kind for the standard
    # viz; detailed viz overrides with a different tag.
    figures_dir = make_run_folder(FIGURES_BASE, tag=MODEL_KIND)
    print(f"Writing figures to: {figures_dir}")

    for split_ds, split_idx, split_name in [
        (val_ds, val_idx, "validation"),
        (test_ds, test_idx, "test"),
    ]:
        if len(split_ds) == 0:
            print(f"No cones in {split_name} split; skipping")
            continue

        # Pick N cones spanning the reionization-behavior range.
        print(f"--- {split_name}: picking {N_CONES_PER_SPLIT} cones "
              f"by reionization behavior at z={STRATIFY_Z} ---")
        picks = pick_cones_by_reion_behavior(
            split_ds, split_idx, target_z,
            n_cones=N_CONES_PER_SPLIT, stratify_z=STRATIFY_Z,
        )
        for idx_in_split, cone_id, summ in picks:
            print(f"  cone {cone_id:4d}  <x_HI>(z={STRATIFY_Z}) = {summ:.3f}")

        # Render per-cone figures (individual files) AND accumulate the
        # arrays we'll need for the summary-grid plot.
        per_cone_for_grid: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []
        for idx_in_split, cone_id, summ in picks:
            print(f"--- {split_name} cone {cone_id} (idx_in_split={idx_in_split}) ---")
            sample = split_ds[idx_in_split]
            dens, truth, pred = predict_cube(model, sample)
            per_cone_for_grid.append((cone_id, summ, dens, truth, pred))

            # z-slice grid (one PNG per cone)
            idxs = np.linspace(0, N_Z - 1, N_SLICES_PER_CONE,
                               dtype=int).tolist()
            fig = plot_z_slices(dens, truth, pred, target_z, idxs, cone_id,
                                split_name)
            out = figures_dir / f"comparison_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

            # xz lightcone strip (one PNG per cone)
            fig = plot_lightcone_strip(dens, truth, pred, target_z, cone_id,
                                       split_name)
            out = figures_dir / f"lightcone_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

            # voxel scatter (one PNG per cone)
            fig = plot_scatter(truth, pred, cone_id, split_name)
            out = figures_dir / f"scatter_3d_{split_name}_cone{cone_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

        # Single summary-grid figure with all N cones' lightcone strips
        # stacked vertically -- the canonical "compare across reionization
        # regimes" plot for the thesis.
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
