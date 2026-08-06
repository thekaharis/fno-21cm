#!/usr/bin/env python3
"""Parity / calibration diagnostic for x_HI lightcone predictions.

Motivation
----------
Whole-volume metrics say *how much* error there is; the boundary-band
diagnostic says *where in space* it lives. This diagnostic asks a third
question: *which part of the x_HI value range does the model get wrong?*

For every voxel of every requested cone we pair (true x_HI, predicted x_HI),
bin by the TRUE value, and summarise the distribution of predictions per bin
(mean, median, 16-84 percentile band, RMSE). Plotted as predicted-vs-true
against the y=x identity line this is a **binned parity plot** (a.k.a. a
regression calibration curve): systematic departure of the median from the
diagonal is conditional bias E[pred|true]-true (e.g. hedging toward the mean
in the transition range 0 < x_HI < 1), and the width of the percentile band
is conditional spread.

Because the truth is heavily bimodal (~74% of voxels >= 0.999, ~6% exactly 0),
a per-bin occupancy histogram is drawn under the parity panel -- without it
the sparse mid-range bins would look as trustworthy as the saturated ones.

The accumulation is streaming (per-bin count/sum/sum-sq + a coarse 2-D
histogram for percentiles), so memory stays a few MB regardless of how many
cones are processed.

Typical use
-----------
Cluster (predicts cones with each checkpoint's own architecture)::

    python -m viz.parity_diagnostic --checkpoints \
        localfno=checkpoints/checkpoints_3d_localfno_1gpu/best_model_state_dict.pt \
        ufno=checkpoint-archive/checkpoints_3d_ufno/best_model_state_dict.pt \
        --split test --n-cones 200 --out parity_out/

Self-test (no data / torch needed)::

    python -m viz.parity_diagnostic --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Core accumulator (torch-free)
# --------------------------------------------------------------------------- #
class ParityAccumulator:
    """Streaming per-true-bin statistics of predictions.

    True axis: ``n_true`` uniform bins on [0, 1] (values clipped in).
    Percentiles come from a (n_true x n_pred) count histogram, so their
    resolution is 1/n_pred (default 0.0025) -- ample for this purpose.
    """

    def __init__(self, n_true: int = 50, n_pred: int = 400):
        self.n_true = int(n_true)
        self.n_pred = int(n_pred)
        self.count = np.zeros(self.n_true, dtype=np.int64)
        self.sum_p = np.zeros(self.n_true, dtype=np.float64)
        self.sum_p2 = np.zeros(self.n_true, dtype=np.float64)
        self.sum_t = np.zeros(self.n_true, dtype=np.float64)
        self.sum_sqerr = np.zeros(self.n_true, dtype=np.float64)
        self.h2d = np.zeros(self.n_true * self.n_pred, dtype=np.int64)

    def update(self, truth: np.ndarray, pred: np.ndarray) -> None:
        t = np.clip(np.asarray(truth, np.float64).ravel(), 0.0, 1.0)
        p = np.clip(np.asarray(pred, np.float64).ravel(), 0.0, 1.0)
        ti = np.minimum((t * self.n_true).astype(np.int64), self.n_true - 1)
        pi = np.minimum((p * self.n_pred).astype(np.int64), self.n_pred - 1)
        nb = self.n_true
        self.count += np.bincount(ti, minlength=nb)
        self.sum_p += np.bincount(ti, weights=p, minlength=nb)
        self.sum_p2 += np.bincount(ti, weights=p * p, minlength=nb)
        self.sum_t += np.bincount(ti, weights=t, minlength=nb)
        self.sum_sqerr += np.bincount(ti, weights=(p - t) ** 2, minlength=nb)
        self.h2d += np.bincount(ti * self.n_pred + pi,
                                minlength=nb * self.n_pred)

    # -- derived ------------------------------------------------------------ #
    @property
    def true_edges(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_true + 1)

    def _percentile_rows(self, qs: tuple[float, ...]) -> np.ndarray:
        """Per-true-bin percentiles of the prediction (bin-center resolution)."""
        h = self.h2d.reshape(self.n_true, self.n_pred)
        centers = (np.arange(self.n_pred) + 0.5) / self.n_pred
        out = np.full((self.n_true, len(qs)), np.nan)
        for i in range(self.n_true):
            tot = h[i].sum()
            if tot == 0:
                continue
            cum = np.cumsum(h[i])
            for j, q in enumerate(qs):
                k = int(np.searchsorted(cum, q / 100.0 * tot))
                out[i, j] = centers[min(k, self.n_pred - 1)]
        return out

    def stats(self) -> dict:
        n = np.maximum(self.count, 1).astype(np.float64)
        mean = self.sum_p / n
        var = np.clip(self.sum_p2 / n - mean ** 2, 0.0, None)
        true_mean = self.sum_t / n
        pcts = self._percentile_rows((16.0, 50.0, 84.0))
        empty = self.count == 0
        for arr in (mean, var, true_mean):
            arr[empty] = np.nan
        return {
            "true_edges": self.true_edges,
            "true_mean": true_mean,             # mean TRUE value per bin
            "count": self.count.copy(),
            "pred_mean": mean,
            "pred_std": np.sqrt(var),
            "pred_p16": pcts[:, 0],
            "pred_p50": pcts[:, 1],
            "pred_p84": pcts[:, 2],
            "bias": mean - true_mean,           # E[pred|true] - true
            "rmse": np.sqrt(self.sum_sqerr / n),
        }

    def save_npz(self, path: Path) -> None:
        np.savez_compressed(
            path, n_true=self.n_true, n_pred=self.n_pred, count=self.count,
            sum_p=self.sum_p, sum_p2=self.sum_p2, sum_t=self.sum_t,
            sum_sqerr=self.sum_sqerr, h2d=self.h2d)


# --------------------------------------------------------------------------- #
# Reporting / plotting
# --------------------------------------------------------------------------- #
def worst_bins(stats: dict, top: int = 3, min_count: int = 1000) -> list[str]:
    """Bins with the largest |median - true|, ignoring under-populated bins."""
    dev = np.abs(stats["pred_p50"] - stats["true_mean"])
    dev[stats["count"] < min_count] = -np.inf
    order = np.argsort(dev)[::-1][:top]
    edges = stats["true_edges"]
    lines = []
    for i in order:
        if not np.isfinite(dev[i]) or dev[i] < 0:
            continue
        lines.append(
            f"true x_HI in [{edges[i]:.2f},{edges[i+1]:.2f}): "
            f"median pred {stats['pred_p50'][i]:.3f} vs true {stats['true_mean'][i]:.3f} "
            f"(bias {stats['bias'][i]:+.3f}, rmse {stats['rmse'][i]:.3f}, "
            f"n={stats['count'][i]:,})")
    return lines


def plot_parity(results: dict[str, dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(7.5, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})

    ax.plot([0, 1], [0, 1], color="0.4", lw=1.0, ls="--", label="identity (y = x)")
    for idx, (name, st) in enumerate(results.items()):
        c = f"C{idx}"
        x = st["true_mean"]
        ax.fill_between(x, st["pred_p16"], st["pred_p84"],
                        color=c, alpha=0.22, linewidth=0,
                        label=f"{name} 16-84%")
        ax.plot(x, st["pred_p50"], color=c, lw=1.8, label=f"{name} median")
        ax.plot(x, st["pred_mean"], color=c, lw=1.0, ls=":",
                label=f"{name} mean")
    ax.set_ylabel("predicted x_HI")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Binned parity plot: E[predicted x_HI | true x_HI]")

    for idx, (name, st) in enumerate(results.items()):
        edges = st["true_edges"]
        axn.step(edges[:-1], np.maximum(st["count"], 0.5), where="post",
                 color=f"C{idx}", lw=1.2, label=name)
        break  # counts are truth-only: identical for every model
    axn.set_yscale("log")
    axn.set_xlabel("true x_HI")
    axn.set_ylabel("voxels / bin")
    axn.grid(alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(results: dict[str, dict], out_path: Path) -> None:
    cols = ("true_lo", "true_hi", "true_mean", "count", "pred_mean", "pred_std",
            "pred_p16", "pred_p50", "pred_p84", "bias", "rmse")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model",) + cols)
        for name, st in results.items():
            edges = st["true_edges"]
            for i in range(len(st["count"])):
                w.writerow([name, f"{edges[i]:.4f}", f"{edges[i+1]:.4f}",
                            f"{st['true_mean'][i]:.6f}", int(st["count"][i])]
                           + [f"{st[k][i]:.6f}" for k in cols[4:]])


# --------------------------------------------------------------------------- #
# Cluster driver (lazy torch / project imports)
# --------------------------------------------------------------------------- #
def run_from_checkpoints(checkpoints: dict[str, str], out_dir: Path,
                         n_cones: int, split: str,
                         n_true: int, n_pred: int,
                         z_window: tuple[float, float] | None) -> dict[str, dict]:
    import torch
    from dataset.dataset_3d import (
        InputFeatures,
        LightconeCubeCache,
        ParameterNormalization,
        resolve_split,
    )
    from modeling import ModelConfig
    from viz.visualize_3d import load_model, predict_cube
    from util.run_metadata import load_run_metadata

    from dataset import paths
    cache = Path(os.environ.get("CUBES_CACHE", paths.CUBES))
    first_checkpoint = Path(next(iter(checkpoints.values())))
    first_meta = load_run_metadata(first_checkpoint.parent)
    input_features = InputFeatures(
        first_meta["input_features"]["name"]
        if first_meta and "input_features" in first_meta
        else "density_z_params"
    )
    dataset = LightconeCubeCache(cache, input_features=input_features)
    if first_meta and first_meta.get("parameter_normalization"):
        dataset.set_parameter_normalization(
            ParameterNormalization.from_dict(first_meta["parameter_normalization"])
        )
    z_grid = np.asarray(dataset.target_z, dtype=float)
    los_mask = None
    if z_window is not None:
        los_mask = (z_grid >= z_window[0]) & (z_grid <= z_window[1])
        if not los_mask.any():
            raise SystemExit(f"z window {z_window} selects no LOS slices")

    train_idx, val_idx, test_idx, _ = resolve_split(dataset, first_meta)
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    rows = rows[:n_cones]
    print(f"[parity] {split} split: {len(rows)} cones, "
          f"z window: {z_window or 'all'}")

    results = {}
    for name, ckpt_path in checkpoints.items():
        checkpoint = Path(ckpt_path)
        metadata = load_run_metadata(checkpoint.parent)
        config = (ModelConfig.from_dict(metadata["model_config"])
                  if metadata and "model_config" in metadata else None)
        if metadata and "input_features" in metadata:
            expected = metadata["input_features"]["name"]
            if expected != input_features.name:
                raise ValueError(
                    f"checkpoint {checkpoint} expects input features "
                    f"{expected!r}, but comparison dataset uses "
                    f"{input_features.name!r}")
        model = load_model(in_channels=dataset.in_channels,
                           checkpoint=checkpoint, model_config=config)
        acc = ParityAccumulator(n_true=n_true, n_pred=n_pred)
        for j, r in enumerate(rows):
            _dens, truth, pred = predict_cube(model, dataset[r])
            if los_mask is not None:
                truth, pred = truth[..., los_mask], pred[..., los_mask]
            acc.update(truth, pred)
            if (j + 1) % 25 == 0 or j + 1 == len(rows):
                print(f"[parity] {name}: {j + 1}/{len(rows)} cones")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        acc.save_npz(out_dir / f"parity_hist_{name}.npz")
        results[name] = acc.stats()
    return results


# --------------------------------------------------------------------------- #
# Self-test (synthetic, no data / torch)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    rng = np.random.default_rng(0)
    # bimodal truth like the lightcones: mass at 0 and ~1, thin transition
    t = np.concatenate([
        np.zeros(60_000),
        rng.uniform(0, 1, 30_000),
        np.full(200_000, 0.9995),
    ])
    # model hedges mid-range toward 0.5 and adds noise
    hedge = 0.3 * (0.5 - t) * ((t > 0.05) & (t < 0.95))
    p = np.clip(t + hedge + rng.normal(0, 0.02, t.shape), 0, 1)

    acc = ParityAccumulator(n_true=20, n_pred=400)
    # feed in chunks to exercise streaming
    for lo in range(0, len(t), 50_000):
        acc.update(t[lo:lo + 50_000], p[lo:lo + 50_000])
    st = acc.stats()

    assert int(st["count"].sum()) == len(t), "voxel count mismatch"
    # saturated-low bin: no hedge applied -> tiny bias (noise clipped at 0)
    assert abs(st["bias"][0]) < 0.03, f"low-bin bias {st['bias'][0]}"
    # mid bin (~0.25): hedge = +0.075 -> bias should recover it
    mid = 5   # bin [0.25, 0.30)
    expect = 0.3 * (0.5 - st["true_mean"][mid])
    assert abs(st["bias"][mid] - expect) < 0.02, \
        f"mid-bin bias {st['bias'][mid]:.4f} vs expected {expect:.4f}"
    # percentile band should bracket the median
    ok = st["count"] > 100
    assert np.all(st["pred_p16"][ok] <= st["pred_p50"][ok] + 1e-9)
    assert np.all(st["pred_p50"][ok] <= st["pred_p84"][ok] + 1e-9)
    print("[selftest] bias recovery:",
          f"low bin {st['bias'][0]:+.4f}, mid bin {st['bias'][mid]:+.4f} "
          f"(expected {expect:+.4f})")

    # plotting + csv smoke test
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        plot_parity({"toy": st}, Path(tmp) / "parity_overlay.png")
        write_csv({"toy": st}, Path(tmp) / "parity_metrics.csv")
    print("[selftest] OK")
    return 0


# --------------------------------------------------------------------------- #
def _parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoints entries must be name=path, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH",
                    help="model checkpoints; each dir needs run_metadata.json")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--n-cones", type=int, default=200)
    ap.add_argument("--true-bins", type=int, default=50)
    ap.add_argument("--pred-bins", type=int, default=400)
    ap.add_argument("--z-window", nargs=2, type=float, metavar=("LO", "HI"),
                    help="restrict to LOS slices with z in [LO, HI]")
    ap.add_argument("--min-count", type=int, default=1000,
                    help="ignore bins below this population in the summary")
    ap.add_argument("--out", type=Path, default=Path("figures/parity_out"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())
    if not args.checkpoints:
        ap.error("--checkpoints is required (or use --selftest)")

    args.out.mkdir(parents=True, exist_ok=True)
    results = run_from_checkpoints(
        _parse_kv(args.checkpoints), args.out, args.n_cones, args.split,
        args.true_bins, args.pred_bins,
        tuple(args.z_window) if args.z_window else None)

    plot_parity(results, args.out / "parity_overlay.png")
    write_csv(results, args.out / "parity_metrics.csv")
    summary = {}
    for name, st in results.items():
        lines = worst_bins(st, min_count=args.min_count)
        summary[name] = lines
        print(f"\n[{name}] hardest x_HI ranges (|median - true|):")
        for line in lines:
            print("  " + line)
    (args.out / "parity_summary.json").write_text(json.dumps(
        {name: lines for name, lines in summary.items()}, indent=2) + "\n")
    print(f"\nParity diagnostic complete: {args.out}")


if __name__ == "__main__":
    main()
