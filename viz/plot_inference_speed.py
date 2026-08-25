#!/usr/bin/env python3
"""Inference cost of the trained 3-D matrix models, two ways.

    --mode accuracy   throughput vs test RMSE  (is the cheap model also good?)
    --mode size       parameters vs throughput (does size predict cost?)

The size view deliberately mirrors `plot_operator_benchmark.py`, which plots
the same axes for freshly built *2-D slice* variants at matched width. The
comparison between the two figures is the point: that benchmark isolates the
operator, this one measures whole trained cubes, and they disagree about the
U-FNO because its cost is dominated by a dense U-Net path that a matched-width
slice probe does not reproduce.

Both read `figures/final_eval/matrix/speed/inference_speed.csv`
(`viz.inference_speed_eval`) and, for the accuracy view,
`figures/final_eval/matrix/rmse/rmse_r2.csv`.

Colour encodes the LOCAL operator throughout the campaign's figures; kept here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_ROOT = Path("checkpoints")
SPEED_CSV = Path("figures/final_eval/matrix/speed/inference_speed.csv")
SPEED_JSON = Path("figures/final_eval/matrix/speed/inference_speed.json")
RMSE_CSV = Path("figures/final_eval/matrix/rmse/rmse_r2.csv")
OUT = {"accuracy": Path("figures/inference_speed_vs_rmse.png"),
       "size": Path("figures/inference_speed_vs_size.png")}

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


def label_of(cfg: dict) -> str:
    if cfg.get("kind") == "ufno":
        return "U-FNO"
    loc, glo = cfg.get("local_operator"), cfg.get("global_operator")
    return f"{TAG.get(loc, loc)} / {TAG.get(glo, glo)}"


def best_legend_loc(xs, ys) -> str:
    """Corner with no points in it, preferring the one furthest from any point."""
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


def pareto_fast_accurate(pts):
    """Nothing is both faster and more accurate.

    Walk from fastest to slowest keeping any run whose RMSE beats everything
    faster than it.
    """
    front = []
    best = float("inf")
    for p in sorted(pts, key=lambda p: -p["cubes_per_s"]):
        if p["rmse"] < best:
            front.append(p)
            best = p["rmse"]
    return sorted(front, key=lambda p: p["cubes_per_s"])


def load() -> tuple[list[dict], dict]:
    if not SPEED_CSV.exists():
        raise SystemExit(f"missing {SPEED_CSV} -- run the speed suite first:\n"
                         f"  sbatch --export=ALL,SUITE=speed,"
                         f"SPECS_FILE=figures/final_eval/specs_matrix.txt,"
                         f"OUT_ROOT=figures/final_eval/matrix "
                         f"slurm/final_eval_suite.sbatch")
    speed = {r["model"]: r for r in csv.DictReader(open(SPEED_CSV))}
    rmse = {r["model"]: r for r in csv.DictReader(open(RMSE_CSV))}
    env = json.load(open(SPEED_JSON))["env"] if SPEED_JSON.exists() else {}

    pts = []
    for run, s in speed.items():
        meta = json.load(open(CKPT_ROOT / f"checkpoints_3d_{run}" / "run_metadata.json"))
        cfg = meta["model_config"]
        pts.append({"run": run, "label": label_of(cfg),
                    "local": cfg.get("local_operator"),
                    "params": int(s["params"]),
                    "ms": float(s["ms_per_cube"]),
                    "cubes_per_s": float(s["cubes_per_s"]),
                    "slices_per_s": float(s["slices_per_s"]),
                    "peak_mib": float(s["peak_mib"]),
                    "rmse": float(rmse[run]["rmse"]) if run in rmse else float("nan")})
    missing = [p["run"] for p in pts if p["rmse"] != p["rmse"]]
    if missing:
        print(f"warning: no RMSE for {missing} -- excluded from the accuracy view")
    return pts, env


def style(ax) -> None:
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def legend(ax, xs, ys) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by = {l: h for h, l in zip(handles, labels)}
    order = [NAME[k] for k in ("fourier", "wavelet", "hadamard",
                               "siren_hadamard", "siren_fourier", "cnn", None)]
    order += [l for l in by if l not in order]
    leg = ax.legend([by[l] for l in order if l in by],
                    [l for l in order if l in by],
                    loc=best_legend_loc(xs, ys), frameon=True, fontsize=9.5,
                    facecolor=SURFACE, edgecolor=GRID, borderpad=0.8)
    for t in leg.get_texts():
        t.set_color(INK_2)


def scatter(ax, pts, xkey, ykey) -> None:
    seen = set()
    for p in sorted(pts, key=lambda p: p[xkey]):
        loc = p["local"]
        lbl = None
        if loc not in seen:
            lbl = NAME.get(loc, str(loc))
            seen.add(loc)
        ax.plot(p[xkey], p[ykey], "o", ms=10, color=HUE.get(loc, INK_MUTED),
                mec=SURFACE, mew=2, zorder=3, linestyle="none", label=lbl)
    for i, p in enumerate(sorted(pts, key=lambda p: p[xkey])):
        dy = 13 if i % 2 == 0 else -20
        ax.annotate(p["label"], (p[xkey], p[ykey]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5, color=INK_2,
                    zorder=4)


def pad_axis(ax, vals, which: str, frac: float = 0.16) -> None:
    lo, hi = min(vals), max(vals)
    pad = frac * (hi - lo)
    (ax.set_ylim if which == "y" else ax.set_xlim)(lo - pad, hi + pad)


def device_note(env: dict) -> str:
    """Provenance footer. Kept off the subtitle line -- it is caveat, not
    finding, and the subtitle is where the reader looks first."""
    dev = env.get("device", "unknown GPU")
    return (f"{dev} | batch 1, whole 140x140x256 cubes | median of "
            f"{env.get('iters', '?')} timed passes after "
            f"{env.get('warmup', '?')} warmup | patch chunk "
            f"{env.get('patch_chunk_size', '?')}")


def footer(fig, env: dict) -> None:
    fig.text(0.995, 0.006, device_note(env), ha="right", va="bottom",
             fontsize=7.5, color=INK_MUTED)


def plot_accuracy(pts, env) -> None:
    pts = [p for p in pts if p["rmse"] == p["rmse"]]
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    front = pareto_fast_accurate(pts)
    ax.plot([p["cubes_per_s"] for p in front], [p["rmse"] for p in front],
            "-", color=INK_MUTED, lw=1.1, alpha=0.7, zorder=1,
            label="Pareto front")
    scatter(ax, pts, "cubes_per_s", "rmse")

    ax.set_xlabel("Inference throughput  (cubes / s, higher is better)",
                  fontsize=10, color=INK_2)
    ax.set_ylabel("Test RMSE  (200 cones, all voxels)", fontsize=10, color=INK_2)
    ax.set_title("Inference speed vs accuracy", fontsize=13, color=INK,
                 pad=34, loc="left")

    fastest = max(pts, key=lambda p: p["cubes_per_s"])
    best = min(pts, key=lambda p: p["rmse"])
    ax.text(0.0, 1.012,
            f"Lower right is better; colour is the LOCAL operator. "
            f"{fastest['label']} is "
            f"{fastest['cubes_per_s']/best['cubes_per_s']:.1f}x faster than "
            f"{best['label']} for {p_pct(fastest['rmse'], best['rmse'])} RMSE",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")

    pad_axis(ax, [p["rmse"] for p in pts], "y")
    pad_axis(ax, [p["cubes_per_s"] for p in pts], "x", 0.10)
    style(ax)
    legend(ax, [p["cubes_per_s"] for p in pts], [p["rmse"] for p in pts])
    footer(fig, env)
    save(fig, OUT["accuracy"])


def p_pct(a: float, b: float) -> str:
    return f"{100 * (a / b - 1):+.1f}%"


def plot_size(pts, env) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    scatter(ax, pts, "params", "slices_per_s")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters  (complex weights counted as two)",
                  fontsize=10, color=INK_2)
    ax.set_ylabel("Inference throughput  (LOS slices / s)",
                  fontsize=10, color=INK_2)
    ax.set_title("Model size vs inference speed", fontsize=13, color=INK,
                 pad=34, loc="left")

    big = max(pts, key=lambda p: p["params"])
    small = min(pts, key=lambda p: p["params"])
    ax.text(0.0, 1.012,
            f"Size does not predict cost: {big['label']} is "
            f"{big['params']/small['params']:.0f}x the parameters of "
            f"{small['label']} and {p_pct(big['slices_per_s'], small['slices_per_s'])} "
            f"the throughput; colour is the LOCAL operator",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")

    pad_axis(ax, [p["slices_per_s"] for p in pts], "y")
    style(ax)
    legend(ax, [p["params"] for p in pts], [p["slices_per_s"] for p in pts])
    footer(fig, env)
    save(fig, OUT["size"])


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=("accuracy", "size", "both"), default="both")
    args = ap.parse_args(argv)

    pts, env = load()
    if args.mode in ("accuracy", "both"):
        plot_accuracy(pts, env)
    if args.mode in ("size", "both"):
        plot_size(pts, env)

    print(f"\n{'model':<24}{'params':>14}{'ms/cube':>10}{'cubes/s':>10}"
          f"{'slices/s':>10}{'RMSE':>10}{'peak MiB':>10}")
    for p in sorted(pts, key=lambda p: p["ms"]):
        print(f"{p['label']:<24}{p['params']:>14,}{p['ms']:>10.1f}"
              f"{p['cubes_per_s']:>10.2f}{p['slices_per_s']:>10.1f}"
              f"{p['rmse']:>10.5f}{p['peak_mib']:>10.0f}")


if __name__ == "__main__":
    main()
