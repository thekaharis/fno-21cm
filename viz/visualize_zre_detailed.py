#!/usr/bin/env python3
"""Detailed multi-cone viz for the density -> z_re(x, y) models.

Counterpart of ``viz/visualize_3d_detailed.py`` for the 2-D z_re task:

  * **16 cones per held-out split** (val + test), picked at evenly spaced
    percentiles of the cone-mean normalized target -- spanning fully-clamped
    late reionizers up to early reionizers, so the render covers the whole
    behavior range instead of whatever the split ordering happens to yield.
  * Per cone: truth / prediction / signed-error map triptych (masked pixels
    hatched out of the error panel) and a truth-vs-prediction hexbin over
    the unclamped pixels.
  * Per split: a 16-row summary grid, a radially averaged 2-D power
    spectrum of truth vs prediction (the FNO-smoothing diagnostic), and a
    ``metrics.json`` with dense + masked RMSE per cone in redshift units.

Configuration comes from the same environment contract as ``fno_zre.py``
(MODEL_KIND, CHECKPOINT_DIR, LIGHTCONE_DIR, ZRE_TARGET_CACHE, TARGET_KIND,
INPUT_FEATURES, N_Z_IN, ...) so a viz run points at a training run by
exporting the identical variables -- see ``slurm/viz_zre.sbatch``.
CHECKPOINT / CHECKPOINT_KIND select the weights file as in the 3-D viz.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fno_zre carries the whole env-driven config (model kind, checkpoint dir,
# data paths, target definition) plus build_zre_model -- reuse it wholesale
# so viz can never drift from what training ran.
import fno_zre
from fno_zre import (
    CHECKPOINT_DIR, DATA_DIR, FILE_GLOB, INPUT_FEATURES, MODEL_KIND,
    N_Z_IN, SPLIT_SEED, TARGET_CACHE, TARGET_KIND, TEST_FRACTION,
    VAL_FRACTION, Z_MAX, Z_MIN, MODEL_CONFIG,
)
from dataset.dataset_zre import ZreMapDataset, split_by_cone
from dataset import paths
from modeling import TrainerModel, build_model, load_checkpoint
from util.run_metadata import resolve_checkpoint

# ------------------------------------------------------------------ config
N_CONES_PER_SPLIT = int(os.environ.get("N_CONES_PER_SPLIT", "16"))
FIGURES_BASE = Path("figures")
VIZ_TAG = os.environ.get("VIZ_TAG", f"zre-{MODEL_KIND}")
CHECKPOINT = resolve_checkpoint(CHECKPOINT_DIR)

DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu",
)

Z_SPAN = Z_MAX - Z_MIN


# ------------------------------------------------- per-run output folder
def make_run_folder(base: Path = FIGURES_BASE, tag: str = "") -> Path:
    """Uniquely named figures folder + run_info breadcrumb (3-D viz style)."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [p for p in (tag, ts) if p]
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        parts.append(f"job{job_id}")
    folder = base / "_".join(parts)
    folder.mkdir(parents=True, exist_ok=True)
    info_lines = [
        f"tag:          {tag or '(unset)'}",
        f"timestamp:    {ts}",
        f"job_id:       {job_id or '(local, no SLURM)'}",
        f"task:         density -> z_re map",
        f"MODEL_KIND:   {MODEL_KIND}",
        f"CHECKPOINT:   {CHECKPOINT}",
        f"TARGET_KIND:  {TARGET_KIND}",
        f"TARGET_CACHE: {TARGET_CACHE}",
        f"INPUT:        {INPUT_FEATURES}",
        f"N_Z_IN:       {N_Z_IN}",
        f"DATA_DIR:     {DATA_DIR}",
    ]
    (folder / "run_info.txt").write_text("\n".join(info_lines) + "\n")
    return folder


# ------------------------------------------------------------------ helpers
def pick_cones_by_target(split_ds, n_cones: int) -> list[tuple[int, float]]:
    """Pick cones at even percentiles of the cone-mean normalized target.

    Returns ``(idx_in_split, summary)`` pairs sorted by summary (late ->
    early reionizers). The summary is the dense mean of the normalized
    target, 0 for fully clamped cones.
    """
    dataset: ZreMapDataset = split_ds.dataset
    summaries = np.array(
        [float(dataset._target[i].mean()) for i in split_ds.indices]
    )
    order = np.argsort(summaries)
    n_cones = min(n_cones, len(order))
    ranks = np.linspace(0, len(order) - 1, n_cones).round().astype(int)
    ranks = sorted(dict.fromkeys(ranks.tolist()))
    return [(int(order[r]), float(summaries[order[r]])) for r in ranks]


@torch.no_grad()
def predict_map(model, sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one cone; returns (truth_z, pred_z, valid_mask) in z units."""
    pred = model(sample["x"][None].to(DEVICE))[0, 0].cpu().numpy()
    truth = sample["y"][0].numpy()
    mask = sample["mask"][0].numpy() > 0.5
    return (
        truth * Z_SPAN + Z_MIN,
        pred * Z_SPAN + Z_MIN,
        mask,
    )


def cone_metrics(truth_z, pred_z, mask) -> dict[str, float]:
    err2 = (pred_z - truth_z) ** 2
    dense = float(np.sqrt(err2.mean()))
    masked = float(np.sqrt(err2[mask].mean())) if mask.any() else float("nan")
    return {
        "rmse_z": dense,
        "rmse_masked_z": masked,
        "valid_fraction": float(mask.mean()),
    }


def radial_power_spectrum(field: np.ndarray, n_bins: int = 30):
    """Radially averaged 2-D power spectrum (mean-subtracted, flat weights)."""
    delta = field - field.mean()
    power = np.abs(np.fft.fftn(delta)) ** 2
    kx = np.fft.fftfreq(field.shape[0])
    ky = np.fft.fftfreq(field.shape[1])
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    bins = np.linspace(0, kk.max(), n_bins + 1)
    which = np.digitize(kk.ravel(), bins) - 1
    spectrum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(spectrum, which.clip(0, n_bins - 1), power.ravel())
    np.add.at(counts, which.clip(0, n_bins - 1), 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    valid = counts > 0
    return centers[valid], spectrum[valid] / counts[valid]


# ------------------------------------------------------------------ plots
def plot_map_triptych(truth_z, pred_z, mask, cone_stem, split, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=150)
    finite_lo = truth_z.min()
    vmax = max(truth_z.max(), finite_lo + 0.5)
    err = pred_z - truth_z
    lim = max(0.1, float(np.abs(err).max()))
    err_shown = np.where(mask, err, np.nan)
    cmap_err = plt.get_cmap("RdBu_r").copy()
    cmap_err.set_bad("0.85")

    panels = (
        (truth_z, "magma", dict(vmin=finite_lo, vmax=vmax), "truth"),
        (pred_z, "magma", dict(vmin=finite_lo, vmax=vmax), "prediction"),
        (err_shown, cmap_err, dict(vmin=-lim, vmax=lim),
         f"error (gray = clamped, {100 * (1 - mask.mean()):.0f}%)"),
    )
    for ax, (img, cmap, kw, title) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap,
                       interpolation="nearest", **kw)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{split} {cone_stem}   rmse_z = {metrics['rmse_z']:.3f}   "
        f"masked = {metrics['rmse_masked_z']:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_scatter(truth_z, pred_z, mask, cone_stem, split, metrics):
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=150)
    if mask.any():
        hb = ax.hexbin(truth_z[mask], pred_z[mask], gridsize=60,
                       cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="pixels")
        lo = min(truth_z[mask].min(), pred_z[mask].min())
        hi = max(truth_z[mask].max(), pred_z[mask].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    else:
        ax.text(0.5, 0.5, "fully clamped cone\n(no unclamped pixels)",
                transform=ax.transAxes, ha="center", va="center",
                color="0.4")
    ax.set_xlabel("truth $z_{re}$")
    ax.set_ylabel("predicted $z_{re}$")
    ax.set_title(
        f"{split} {cone_stem}\nmasked rmse_z = "
        f"{metrics['rmse_masked_z']:.3f}",
        fontsize=10.5,
    )
    fig.tight_layout()
    return fig


def plot_summary_grid(per_cone, split):
    n = len(per_cone)
    fig, axes = plt.subplots(n, 3, figsize=(10.8, 3.35 * n), dpi=130,
                             squeeze=False)
    for row, entry in enumerate(per_cone):
        stem, summary, truth_z, pred_z, mask, metrics = entry
        lo, hi = truth_z.min(), max(truth_z.max(), truth_z.min() + 0.5)
        err = np.where(mask, pred_z - truth_z, np.nan)
        lim = max(0.1, float(np.nanmax(np.abs(err))) if mask.any() else 0.1)
        cmap_err = plt.get_cmap("RdBu_r").copy()
        cmap_err.set_bad("0.85")
        images = (
            (truth_z, "magma", dict(vmin=lo, vmax=hi)),
            (pred_z, "magma", dict(vmin=lo, vmax=hi)),
            (err, cmap_err, dict(vmin=-lim, vmax=lim)),
        )
        for col, (img, cmap, kw) in enumerate(images):
            ax = axes[row][col]
            im = ax.imshow(img, origin="lower", cmap=cmap,
                           interpolation="nearest", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
            if row == 0:
                ax.set_title(("truth", "prediction", "error")[col],
                             fontsize=11)
        axes[row][0].set_ylabel(
            f"{stem.replace('21cmfast_11d_', '')}\n"
            f"$\\langle y \\rangle$={summary:.2f}  "
            f"rmse={metrics['rmse_z']:.2f}",
            fontsize=8.5,
        )
    fig.suptitle(f"z_re maps, {split} split ({n} cones, late -> early)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    return fig


def plot_power_spectra(per_cone, split):
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(6.4, 6.6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    ratios = []
    k_ref = None
    for stem, _, truth_z, pred_z, mask, _ in per_cone:
        if not mask.any():
            continue
        k, p_true = radial_power_spectrum(truth_z)
        _, p_pred = radial_power_spectrum(pred_z)
        k_ref = k
        ratios.append(p_pred / np.maximum(p_true, 1e-30))
        ax.loglog(k, p_true, color="#41597A", alpha=0.25, lw=1)
        ax.loglog(k, p_pred, color="#C4622D", alpha=0.25, lw=1)
    if k_ref is None:
        plt.close(fig)
        return None
    ax.loglog([], [], color="#41597A", label="truth")
    ax.loglog([], [], color="#C4622D", label="prediction")
    ax.set_ylabel("radially averaged $P(k)$")
    ax.legend(frameon=False)
    ax.set_title(f"z_re map power spectra, {split} split", fontsize=12)

    ratios = np.array(ratios)
    median = np.median(ratios, axis=0)
    ax_ratio.semilogx(k_ref, median, color="0.2", lw=1.6)
    ax_ratio.fill_between(k_ref, np.percentile(ratios, 16, axis=0),
                          np.percentile(ratios, 84, axis=0),
                          color="0.5", alpha=0.3, lw=0)
    ax_ratio.axhline(1.0, color="r", ls="--", lw=0.8)
    ax_ratio.set_ylim(0, 1.6)
    ax_ratio.set_xlabel("$k$ [pixel$^{-1}$]")
    ax_ratio.set_ylabel("pred / truth")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ main
def main():
    print(f"[visualize_zre_detailed]  MODEL_KIND={MODEL_KIND}  "
          f"TARGET_KIND={TARGET_KIND}  N_CONES_PER_SPLIT={N_CONES_PER_SPLIT}")
    print(f"Device: {DEVICE}")
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found: {CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        print(f"No lightcone files under {DATA_DIR}/{FILE_GLOB}",
              file=sys.stderr)
        sys.exit(1)
    print(f"{len(files)} lightcones in {DATA_DIR}")

    dataset = ZreMapDataset(
        files,
        target_cache=TARGET_CACHE,
        target_kind=TARGET_KIND,
        n_z_in=N_Z_IN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        use_params=(INPUT_FEATURES == "density_params"),
        preload=False,
        density_cache=os.environ.get("ZRE_INPUT_CACHE", paths.ZRE_INPUTS),
    )
    train_ds, val_ds, test_ds = split_by_cone(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION,
        seed=SPLIT_SEED,
    )
    # The training script fits normalization on the (deterministic) train
    # split at run time and stores nothing, so refitting here reproduces
    # the training-time statistics exactly.
    dataset.set_parameter_normalization(
        dataset.fit_parameter_normalization(train_ds.indices)
    )

    inner = build_model(MODEL_CONFIG, dataset.in_channels)
    description = MODEL_CONFIG.describe()
    model = TrainerModel(inner)
    report = load_checkpoint(model, CHECKPOINT)
    print(f"[load_model] transform: {report.transform!r}; matched "
          f"{report.matched}/{report.total} params ({description})")
    if report.missing:
        print(f"[load_model] WARNING: {len(report.missing)} params left at "
              f"random init: {sorted(report.missing)[:3]}...")
    model = model.to(DEVICE).eval()

    figures_dir = make_run_folder(FIGURES_BASE, tag=f"{VIZ_TAG}-detailed")
    print(f"Writing figures to: {figures_dir}")

    all_metrics: dict[str, dict] = {}
    for split_ds, split_name in [(val_ds, "validation"), (test_ds, "test")]:
        if len(split_ds) == 0:
            print(f"No cones in {split_name} split; skipping")
            continue
        picks = pick_cones_by_target(split_ds, N_CONES_PER_SPLIT)
        print(f"--- {split_name}: {len(picks)} cones by mean target ---")

        per_cone = []
        for idx_in_split, summary in picks:
            global_idx = split_ds.indices[idx_in_split]
            stem = dataset.file_paths[global_idx].stem
            sample = dataset[global_idx]
            truth_z, pred_z, mask = predict_map(model, sample)
            metrics = cone_metrics(truth_z, pred_z, mask)
            per_cone.append((stem, summary, truth_z, pred_z, mask, metrics))
            all_metrics[f"{split_name}/{stem}"] = metrics
            print(f"  {stem}  <y>={summary:.3f}  "
                  f"rmse_z={metrics['rmse_z']:.3f}  "
                  f"masked={metrics['rmse_masked_z']:.3f}")

            short = stem.replace("21cmfast_11d_", "")
            fig = plot_map_triptych(truth_z, pred_z, mask, short,
                                    split_name, metrics)
            fig.savefig(figures_dir / f"maps_{split_name}_{short}.png",
                        bbox_inches="tight")
            plt.close(fig)

            fig = plot_scatter(truth_z, pred_z, mask, short, split_name,
                               metrics)
            fig.savefig(figures_dir / f"scatter_{split_name}_{short}.png",
                        bbox_inches="tight")
            plt.close(fig)

        fig = plot_summary_grid(per_cone, split_name)
        fig.savefig(figures_dir / f"summary_grid_{split_name}.png",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved summary grid for {split_name}")

        fig = plot_power_spectra(per_cone, split_name)
        if fig is not None:
            fig.savefig(figures_dir / f"power_spectrum_{split_name}.png",
                        bbox_inches="tight")
            plt.close(fig)
            print(f"Saved power spectra for {split_name}")

        rmse = [m["rmse_z"] for m in
                (entry[5] for entry in per_cone)]
        masked = [entry[5]["rmse_masked_z"] for entry in per_cone
                  if np.isfinite(entry[5]["rmse_masked_z"])]
        all_metrics[f"{split_name}/aggregate"] = {
            "mean_rmse_z": float(np.mean(rmse)),
            "mean_rmse_masked_z": float(np.mean(masked)) if masked else None,
            "n_cones_rendered": len(per_cone),
            "n_cones_in_split": len(split_ds),
        }

    (figures_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2)
    )
    print(f"Metrics: {figures_dir / 'metrics.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
