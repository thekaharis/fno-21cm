"""One table for every completed 2-D x_HI run, on identical held-out slices.

Scraping each run's metrics.jsonl would not be comparable: loss terms were added
over the course of the sweeps, so early runs have no wall/expwall/highk columns
at all, and each run's own training loss differs. Every model here is instead
re-evaluated on the same test slices with the same metrics.

Reported per run:
  train loss   what it optimised (from run metadata), which is the thing that
               makes RMSE an unfair headline for the L2-free runs
  params       trainable, plus the training-log convention in brackets
               (complex weights counted twice)
  RMSE         pixel error on the shared slices
  wall/expwall wall-placement error -- distance-weighted, so misplacement costs
               grow with distance rather than saturating
  width/blur   sharpness; compare against the TRUTH row, not against each other

Run: sbatch slurm/compare_all_2d.sbatch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch

from losses import ExponentialWallDistance, WallPlacementLoss
from legacy.xhi2d import field_metrics as fm
from legacy.xhi2d.slice_eval import gather, open_run

N_SLICES = 384
OUT_MD = "figures/summary/xhi2d_all_variants.md"
OUT_CSV = "figures/summary/xhi2d_all_variants.csv"

TAGS = {"fourier": "fno", "wavelet": "wno", "hadamard": "whno",
        "siren_fourier": "sirenfno", "cnn": "cnn"}


def arch_label(cfg: dict) -> str:
    kind = cfg.get("kind", "?")
    if kind == "localop":
        return (f"local {TAGS.get(cfg.get('local_operator'), '?')}"
                f" / global {TAGS.get(cfg.get('global_operator'), '?')}")
    return {"ufno": "U-FNO", "fno": "FNO", "localwno": "local wno / global fno",
            "localfno": "local fno / global fno",
            "localwhno": "local whno / global fno"}.get(kind, kind)


def loss_label(meta: dict) -> str:
    tr = meta.get("training", {})
    w = tr.get("loss_weights", {})
    if not isinstance(w, dict):
        return "?"
    named = {"l2": "L2", "h1": "H1", "bce": "BCE", "swd": "SWD",
             "highk": "highK", "wall": "wall", "h1semi": "H1semi",
             "expwall": "expwall"}
    parts = [f"{v:g}*{named.get(k, k)}" for k, v in sorted(w.items()) if v]
    base = " + ".join(parts) or "?"
    mc = meta.get("model_config", {})
    if mc.get("contrast_mode", "off") != "off":
        kind = mc.get("contrast_schedule_kind", "sigmoid")
        base += f"  [contrast {mc['contrast_mode']}/{kind}]"
    return base


def best_val(run: Path):
    p = run / "metrics.jsonl"
    if not p.exists():
        return None, None
    rows = [json.loads(l) for l in p.read_text().splitlines()
            if l.strip() and '"val_l2"' in l]
    if not rows:
        return None, None
    b = min(rows, key=lambda r: r["val_l2"])
    return b["val_l2"], b["epoch"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=N_SLICES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    runs = sorted(d for d in Path("checkpoints").glob("xhi2d_*")
                  if (d / "final_model_state_dict.pt").is_file()
                  and (d / "run_metadata.json").is_file())
    print(f"{len(runs)} completed runs; {args.slices} shared held-out slices\n")

    wall = WallPlacementLoss(cap=32)
    expw = ExponentialWallDistance(scale=16.0, cap=32)
    rows, truth_ref = [], None
    for d in runs:
        try:
            s = gather(str(d), split="test", max_slices=args.slices,
                       device=args.device)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [skip] {d.name}: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        meta = json.loads((d / "run_metadata.json").read_text())
        model, _, _ = open_run(str(d), device=args.device)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_real = sum(p.numel() * (2 if p.is_complex() else 1)
                     for p in model.parameters() if p.requires_grad)
        p4, t4 = s.pred.unsqueeze(1), s.truth.unsqueeze(1)
        v, e = best_val(d)
        truth_ref = s.truth
        rows.append({
            "run": d.name.replace("xhi2d_", ""),
            "architecture": arch_label(meta.get("model_config", {})),
            "train_loss": loss_label(meta),
            "params": n, "params_real": n_real,
            "best_val_l2": v, "best_epoch": e,
            "rmse": float(fm.rmse(s.pred, s.truth)),
            "wall": float(wall(p4, t4)), "expwall": float(expw(p4, t4)),
            "width_px": float(fm.width_px(s.pred).mean()),
            "blur_frac": float(fm.blur_frac(s.pred).mean()),
        })
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["rmse"])
    truth = {"width_px": float(fm.width_px(truth_ref).mean()),
             "blur_frac": float(fm.blur_frac(truth_ref).mean())}

    hdr = (f"{'run':<24} {'architecture':<24} {'train loss':<30} "
           f"{'params':>9} {'RMSE':>8} {'wall':>8} {'expwall':>8} "
           f"{'width':>6} {'blur':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<24} {r['architecture']:<24} {r['train_loss'][:29]:<30} "
              f"{r['params']:9,} {r['rmse']:8.5f} {r['wall']:8.5f} "
              f"{r['expwall']:8.5f} {r['width_px']:6.2f} {r['blur_frac']:7.4f}")
    print("-" * len(hdr))
    print(f"{'TRUTH':<24} {'':<24} {'':<30} {'':>9} {'':>8} {'':>8} {'':>8} "
          f"{truth['width_px']:6.2f} {truth['blur_frac']:7.4f}")

    os.makedirs("figures", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT_MD, "w") as f:
        f.write(f"# 2-D x_HI model variants\n\n")
        f.write(f"All {len(rows)} completed runs re-evaluated on the same "
                f"{args.slices} held-out test slices. Sorted by RMSE.\n\n")
        f.write("RMSE is not the whole story: the L2-free runs are not "
                "optimising it, and `width`/`blur` should be compared against "
                "the TRUTH row rather than between models.\n\n")
        f.write("| run | architecture | train loss | params | params (log) | "
                "best val L2 | RMSE | wall | expwall | width | blur |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            bv = f"{r['best_val_l2']:.4f}" if r["best_val_l2"] else "--"
            f.write(f"| {r['run']} | {r['architecture']} | {r['train_loss']} | "
                    f"{r['params']:,} | {r['params_real']:,} | {bv} | "
                    f"{r['rmse']:.5f} | {r['wall']:.5f} | {r['expwall']:.5f} | "
                    f"{r['width_px']:.2f} | {r['blur_frac']:.4f} |\n")
        f.write(f"| **TRUTH** | | | | | | | | | **{truth['width_px']:.2f}** | "
                f"**{truth['blur_frac']:.4f}** |\n")
    print(f"\nwrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
