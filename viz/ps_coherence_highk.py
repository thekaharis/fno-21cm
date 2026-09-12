#!/usr/bin/env python3
"""Coherence r(k) against small-scale power fidelity, for 2-D x_HI models.

Two independent failure modes get conflated by any single scalar:

* **Coherence** `r(k) = <P_x P_y*> / sqrt(P_x P_y)` -- is the structure in the
  *right place*? Falls when the prediction misplaces fronts, regardless of how
  much power it carries.
* **Small-scale power fidelity** -- is there the right *amount* of structure?
  This is what `losses.HighKPowerRatio` scores: the mean squared log ratio of
  radially binned power above `k_min` (0.2 cycles/px ~ 0.9 Mpc^-1 on the
  200 Mpc / 140 px grid). Being built from |FFT|^2 it is translation invariant,
  so it cannot be reduced by hedging on edge position.

The 3-D matrix found these **anti-correlated** across architectures -- U-FNO
held coherence to the highest k while being wrong about amplitude by 48%, and
the Walsh models did the reverse. This renders the same plane for the 2-D runs
so the learned-waveform models can be placed on it.

Panels: r(k) curves, the power ratio P_pred/P_true, and the summary plane
(coherence scale vs highk) where lower-right is better on both axes.

Run:
    python -m viz.ps_coherence_highk --runs lwf/lwf=checkpoints/... whno=... \
        --n-slices 400 --out figures/summary/ps_coherence_highk.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.slices import SliceCache
from dataset.dataset_3d import ParameterNormalization
from losses import HighKPowerRatio
from modeling import ModelConfig, TrainerModel, build_model as build_from_config, load_checkpoint
from viz.compare_xhi2d_models import load_run, parse_run

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#9d4edd", "#d4a017", "#00a6a6",
           "#c2255c", "#495057"]


def radial_bins(n: int, n_bins: int, k_max: float = 0.5):
    """Radial bins up to the per-axis Nyquist limit.

    Modes beyond k = 0.5 exist only in the corners of the 2-D spectrum, so
    their bins average very few modes and are not isotropic; the DC mode is
    removed by mean subtraction and would only add zeros. Both are dropped
    (index -1).
    """
    ky = np.fft.fftfreq(n)
    kx = np.fft.rfftfreq(n)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = np.linspace(0.0, k_max, n_bins + 1)
    index = np.digitize(radius, edges) - 1
    index[(radius >= k_max) | (radius == 0) | (index < 0)] = -1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return index, centers


def spectra(pred: np.ndarray, truth: np.ndarray, n_bins: int = 24):
    """Radially binned auto- and cross-power, accumulated over all slices."""
    n = pred.shape[-1]
    index, centers = radial_bins(n, n_bins)
    fp = np.fft.rfft2(pred - pred.mean(axis=(-2, -1), keepdims=True), norm="ortho")
    ft = np.fft.rfft2(truth - truth.mean(axis=(-2, -1), keepdims=True), norm="ortho")
    pp = (fp * fp.conj()).real
    tt = (ft * ft.conj()).real
    pt = (fp * ft.conj()).real
    flat = index.ravel()
    keep = flat >= 0
    out = []
    for field in (pp, tt, pt):
        acc = np.zeros(n_bins)
        f = field.reshape(field.shape[0], -1).sum(axis=0)
        np.add.at(acc, flat[keep], f[keep])
        counts = np.bincount(flat[keep], minlength=n_bins).astype(float)
        out.append(acc / np.maximum(counts, 1.0))
    pp_b, tt_b, pt_b = out
    coherence = pt_b / np.sqrt(np.maximum(pp_b * tt_b, 1e-30))
    ratio = pp_b / np.maximum(tt_b, 1e-30)
    return centers, coherence, ratio


def coherence_scale(centers, coherence, threshold=0.9):
    """First k where r drops below the threshold; NaN if it never does."""
    below = np.where(coherence < threshold)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(centers[0])
    x0, x1 = centers[i - 1], centers[i]
    y0, y1 = coherence[i - 1], coherence[i]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def rebuild(metadata):
    """Build from run metadata, tolerating both metadata shapes.

    `viz.compare_xhi2d_models.build_model` reads `model_config["in_channels"]`,
    which newer runs no longer record -- they carry it under `input_features`
    instead. Reading only the old location raises KeyError on every run since
    that change, so try both.
    """
    config = metadata["model_config"]
    channels = (config.get("in_channels")
                or metadata.get("input_features", {}).get("in_channels"))
    if channels is None:
        raise KeyError("in_channels missing from model_config and input_features")
    inner = build_from_config(ModelConfig.from_dict(config), int(channels))
    # Mirror viz.compare_xhi2d_models.build_model from here on: optional contrast
    # wrapper, then TrainerModel. Checkpoints are saved from the TrainerModel, so
    # their keys carry an `fno.` prefix that the bare model would not match.
    if str(config.get("contrast_mode", "off")) != "off":
        from contrast import ContrastComposed
        inner = ContrastComposed(
            inner, str(config["contrast_mode"]),
            schedule=config.get("contrast_schedule"),
            freeze=bool(config.get("contrast_freeze", False)),
            schedule_kind=str(config.get("contrast_schedule_kind", "sigmoid")),
            n_bins=int(config.get("contrast_bins", 14)))
    return TrainerModel(inner)


@torch.inference_mode()
def predict_all(run, inputs, device, batch=64):
    model = rebuild(run.metadata)
    result = load_checkpoint(model, run.checkpoint)
    # Waveform checkpoints from before phase mixing lack `phase_weight`. The
    # tensor is zero-initialized, so a zero-filled cross-phase block reproduces
    # the architecture they were trained as exactly. Tolerate that and only that.
    missing = [k for k in (result.missing or []) if not k.endswith("phase_weight")]
    if missing or result.unexpected:
        raise RuntimeError(f"incomplete load for {run.label}: "
                           f"{result.matched}/{result.total}, missing={missing}, "
                           f"unexpected={result.unexpected}")
    if result.missing:
        stale = [name for name, p in model.named_parameters()
                 if name.endswith("phase_weight") and p.detach().abs().max() > 0]
        if stale:
            raise RuntimeError(f"{run.label}: absent phase_weight is nonzero: {stale}")
        print(f"[ps] {run.label}: pre-phase-mixing checkpoint, "
              f"{len(result.missing)} phase_weight tensors zero-filled", flush=True)
    model = model.to(device).eval()
    chunks = [model(inputs[i:i + batch].to(device)).cpu().numpy()[:, 0]
              for i in range(0, len(inputs), batch)]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(chunks)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", nargs="+", required=True, type=parse_run,
                    metavar="LABEL=CHECKPOINT_DIR")
    ap.add_argument("--n-slices", type=int, default=400)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--k-min", type=float, default=0.2,
                    help="highk cutoff in cycles/px, matching losses.HighKPowerRatio")
    ap.add_argument("--out", type=Path, default=Path("figures/summary/ps_coherence_highk.png"))
    args = ap.parse_args(argv)

    runs = [r for r in (load_run(label, path) for label, path in args.runs) if r]
    if not runs:
        raise SystemExit("no loadable runs")

    ref = runs[0].metadata
    cache = SliceCache(ref["dataset"]["cache_file"],
                       input_features=ref["input_features"]["name"],
                       parameter_normalization=ParameterNormalization.from_dict(
                           ref["parameter_normalization"])
                       if ref.get("parameter_normalization") else None)
    # The 2-D split is recorded by CONE id, not slice row: a cone contributes
    # many slices, so selecting rows directly would leak cones across splits.
    test_cones = np.asarray(ref["split"]["test_cone_ids"], dtype=np.int64)
    test_idx = np.flatnonzero(np.isin(cache.cone_id, test_cones))
    if args.n_slices < len(test_idx):
        # Evenly spaced rather than the first N, which would be one end of the
        # redshift range and unrepresentative of the reionization history.
        test_idx = test_idx[np.linspace(0, len(test_idx) - 1, args.n_slices).astype(int)]
    items = [cache[int(i)] for i in test_idx]
    # Fully neutral / fully ionized slices have ~zero true small-scale power, so
    # a per-slice log power ratio diverges on them and dominates the mean.
    xhi_mean = np.asarray(cache.xHI_mean)[test_idx]
    active = torch.from_numpy((xhi_mean > 0.001) & (xhi_mean < 0.999))
    inputs = torch.stack([it["x"] for it in items])
    truth = np.stack([it["y"].numpy()[0] for it in items])
    print(f"[ps] {len(items)} test slices, shape {truth.shape[-2:]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    highk = HighKPowerRatio(k_min=args.k_min)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    for i, run in enumerate(runs):
        pred = predict_all(run, inputs, device)
        centers, coh, ratio = spectra(pred, truth, args.n_bins)
        p_t = torch.from_numpy(pred).unsqueeze(1)
        t_t = torch.from_numpy(truth).unsqueeze(1)
        hk_all = float(highk(p_t, t_t))
        hk = float(highk(p_t[active], t_t[active]))
        scale = coherence_scale(centers, coh)
        band = centers >= args.k_min
        coh_hk = float(coh[band].mean())
        colour = PALETTE[i % len(PALETTE)]
        axes[0].plot(centers, coh, "-", color=colour, lw=1.8, label=run.label)
        axes[1].plot(centers, ratio, "-", color=colour, lw=1.8, label=run.label)
        axes[2].plot(coh_hk, hk, "o", ms=11, color=colour, mec=SURFACE, mew=2)
        axes[2].annotate(run.label, (coh_hk, hk), textcoords="offset points",
                         xytext=(0, 13 if i % 2 == 0 else -19), ha="center",
                         fontsize=8.5, color=INK_2)
        rows.append({"model": run.label, "coherence_mean_k_ge_kmin": coh_hk,
                     "coherence_k_r0.9": scale, "highk_active": hk,
                     "highk_all_slices": hk_all, "n_slices": len(items),
                     "n_active": int(active.sum())})
        print(f"[ps] {run.label:24s} r(k>={args.k_min})={coh_hk:.4f}  "
              f"highk active={hk:.4f} all={hk_all:.4f}  k(r<0.9)={scale:.4f}", flush=True)

    axes[0].axhline(0.9, color=INK_MUTED, lw=1.0, ls="--")
    axes[0].set_ylabel("coherence  $r(k)$", fontsize=10, color=INK_2)
    axes[0].set_title("Is the structure in the right place?", fontsize=11,
                      color=INK, loc="left")
    axes[0].set_ylim(0, 1.02)
    axes[1].axhline(1.0, color=INK_MUTED, lw=1.0, ls="--")
    axes[1].axvline(args.k_min, color=INK_MUTED, lw=1.0, ls=":")
    axes[1].set_ylabel("$P_\\mathrm{pred}/P_\\mathrm{true}$", fontsize=10, color=INK_2)
    axes[1].set_title(f"Is there the right amount?  (dotted: highk cutoff "
                      f"{args.k_min})", fontsize=11, color=INK, loc="left")
    axes[1].set_yscale("log")
    for ax in axes[:2]:
        ax.set_xlabel("$k$  (cycles / pixel)", fontsize=10, color=INK_2)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    axes[2].set_xlabel(f"mean coherence $r(k)$ for $k \\geq {args.k_min}$  (higher better)",
                       fontsize=10, color=INK_2)
    axes[2].set_ylabel("highk on active slices  (lower better)", fontsize=10, color=INK_2)
    axes[2].set_title("Placement vs amount of small-scale structure", fontsize=11,
                      color=INK, loc="left")

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
