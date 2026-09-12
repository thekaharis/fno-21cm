#!/usr/bin/env python3
"""Model size vs accuracy (test R^2) for the 3-D matrix runs.

Parameter counts come from the saved checkpoints, not from rebuilding the
model: `run_metadata.json` does not always record `ndim`, so `from_dict` can
silently produce a 2-D model of a 3-D run. Counting the checkpoint tensors is
what was actually trained, and it reproduces the "Model: N parameters" line in
the training logs exactly (verified for ufno, cnn_whno, cnn_swhno).

Complex weights count as two, matching neuralop's count_model_params and the
training logs.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_ROOT = Path("checkpoints")
RMSE_CSV = Path("figures/shared/eval/final_eval/matrix/rmse/rmse_r2.csv")
OUT_BY_METRIC = {"r2": Path("figures/summary/params_vs_accuracy.png"),
                 "rmse": Path("figures/summary/params_vs_rmse.png")}
# (axis label, lower_is_better)
METRICS = {"r2": ("Test $R^2$  (200 cones, all voxels)", False),
           "rmse": ("Test RMSE  (200 cones, all voxels)", True)}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
HUE = {"fourier": "#2a78d6", "wavelet": "#eb6834", "hadamard": "#1baf7a",
       "siren_hadamard": "#9d4edd", "cnn": "#d4a017",
       "siren_fourier": "#00a6a6", None: "#8a8880"}
NAME = {"fourier": "local FNO", "wavelet": "local WNO",
        "hadamard": "local WHNO", "siren_hadamard": "local SWHNO",
        "cnn": "local CNN (U-Net path)", "siren_fourier": "local SirenFNO",
        None: "U-FNO (whole-volume)"}
TAG = {"fourier": "fno", "wavelet": "wno", "hadamard": "whno",
       "siren_hadamard": "swhno", "siren_fourier": "sfno", "cnn": "cnn"}


def params_of(run: str) -> int:
    import torch
    sd = torch.load(CKPT_ROOT / f"checkpoints_3d_{run}" / "best_model_state_dict.pt",
                    map_location="cpu", weights_only=True)
    return sum(v.numel() * (2 if v.is_complex() else 1) for v in sd.values()
               if v.is_floating_point() or v.is_complex())


def label_of(run: str, cfg: dict) -> str:
    loc, glo = cfg.get("local_operator"), cfg.get("global_operator")
    if cfg.get("kind") == "ufno":
        return "U-FNO"
    return f"{TAG.get(loc, loc)} / {TAG.get(glo, glo)}"


def best_legend_loc(xs, ys) -> str:
    """Put the legend in the corner with the most clearance.

    Hardcoding a corner breaks whenever the metric flips the plot vertically:
    "lower right" is empty for R^2 but is exactly where the U-FNO sits for
    RMSE. Footprints are generous so a near-miss still loses to a clear corner.
    """
    if not xs:
        return "lower right"
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx = [(x - x0) / (x1 - x0) if x1 > x0 else 0.5 for x in xs]
    ny = [(y - y0) / (y1 - y0) if y1 > y0 else 0.5 for y in ys]
    boxes = {"lower left":  (0.00, 0.40, 0.00, 0.34),
             "lower right": (0.60, 1.00, 0.00, 0.34),
             "upper left":  (0.00, 0.40, 0.66, 1.00),
             "upper right": (0.60, 1.00, 0.66, 1.00)}
    scored = []
    for loc, (bx0, bx1, by0, by1) in boxes.items():
        inside = sum(1 for a, b in zip(nx, ny)
                     if bx0 <= a <= bx1 and by0 <= b <= by1)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        clearance = min(((a - cx) ** 2 + (b - cy) ** 2) ** 0.5
                        for a, b in zip(nx, ny))
        scored.append((inside, -clearance, loc))
    return min(scored)[2]


def pareto(points, metric: str, lower_is_better: bool):
    """Runs that nothing beats on both axes: fewer parameters AND better score.

    The direction matters -- for RMSE "better" is lower, so reusing the R^2
    comparison here would return the exactly wrong set.
    """
    front = []
    best = float("inf") if lower_is_better else float("-inf")
    for p in sorted(points, key=lambda p: p["params"]):
        v = p[metric]
        if (v < best) if lower_is_better else (v > best):
            front.append(p)
            best = v
    return front


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--metric", choices=sorted(METRICS), default="r2")
    args = ap.parse_args(argv)
    metric = args.metric
    ylabel, lower_better = METRICS[metric]
    out = OUT_BY_METRIC[metric]

    rows = list(csv.DictReader(open(RMSE_CSV)))
    pts = []
    for r in rows:
        run = r["model"]
        meta = json.load(open(CKPT_ROOT / f"checkpoints_3d_{run}" / "run_metadata.json"))
        cfg = meta["model_config"]
        pts.append({"run": run, "r2": float(r["r2"]), "rmse": float(r["rmse"]),
                    "params": params_of(run), "local": cfg.get("local_operator"),
                    "label": label_of(run, cfg)})

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    front = pareto(pts, metric, lower_better)
    ax.plot([p["params"] for p in front], [p[metric] for p in front],
            "-", color=INK_MUTED, lw=1.1, alpha=0.7, zorder=1,
            label="Pareto front")

    seen = set()
    for p in sorted(pts, key=lambda p: p["params"]):
        loc = p["local"]
        lbl = None
        if loc not in seen:
            lbl = NAME.get(loc, str(loc))
            seen.add(loc)
        ax.plot(p["params"], p[metric], "o", ms=10, color=HUE.get(loc, INK_MUTED),
                mec=SURFACE, mew=2, zorder=3, linestyle="none", label=lbl)

    # Stagger labels: the cluster below 3e6 is dense on the x axis.
    for i, p in enumerate(sorted(pts, key=lambda p: p["params"])):
        dy = 13 if i % 2 == 0 else -20
        ax.annotate(p["label"], (p["params"], p[metric]),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=INK_2, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters  (complex weights counted as two)",
                  fontsize=10, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.set_title("Model size vs accuracy", fontsize=13, color=INK, pad=34,
                 loc="left")

    best = (min if lower_better else max)(pts, key=lambda p: p[metric])
    small = min(pts, key=lambda p: p["params"])
    gap = abs(best[metric] - small[metric])
    unit = f"{gap:.5f} RMSE" if metric == "rmse" else f"{100*gap:.2f} points of $R^2$"
    ax.text(0.0, 1.012,
            f"3-D matrix runs, best checkpoint per architecture; colour is the "
            f"LOCAL operator. {small['label']} is "
            f"{best['params']/small['params']:.0f}x smaller than "
            f"{best['label']} for {unit}",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")

    lo = min(p[metric] for p in pts)
    hi = max(p[metric] for p in pts)
    pad = 0.16 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    by = {l: h for h, l in zip(handles, labels)}
    order = [NAME[k] for k in ("fourier", "wavelet", "hadamard",
                               "siren_hadamard", "siren_fourier", "cnn", None)]
    order += [l for l in by if l not in order]
    loc = best_legend_loc([p["params"] for p in pts], [p[metric] for p in pts])
    leg = ax.legend([by[l] for l in order if l in by],
                    [l for l in order if l in by],
                    loc=loc, frameon=True, fontsize=9.5,
                    facecolor=SURFACE, edgecolor=GRID, borderpad=0.8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"\n{'model':<24}{'params':>14}{'R^2':>10}{'RMSE':>10}")
    for p in sorted(pts, key=lambda p: p[metric] if lower_better else -p[metric]):
        mark = " *" if p in front else ""
        print(f"{p['label']:<24}{p['params']:>14,}{p['r2']:>10.5f}"
              f"{p['rmse']:>10.5f}{mark}")
    print("\n* = on the Pareto front (nothing is both smaller and more accurate)")


if __name__ == "__main__":
    main()
