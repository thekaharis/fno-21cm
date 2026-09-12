#!/usr/bin/env python3
"""Run-level results table and paired seed contrasts for the 2-D waveform controls.

Matrix: square and sawtooth x {frozen from init, adapt 25 then freeze, continuous
joint}, plus frozen sine as the fixed-Fourier reference; three training seeds
each, all on the same data split (SPLIT_SEED 42). Conditions are paired by seed:
a given RUN_SEED gives identical non-waveform initialization and batch order in
every condition (verified before launch), so differences are taken WITHIN a
seed and then summarized across seeds.

Recorded per run: best val_l2 and its epoch, final val_l2 (epoch 99), training
error at both, test_l2 at the best epoch, best val_l2 after epoch 24 (the
post-freeze window of the adapt runs, reported for every condition so the
windows are comparable), and which waveform artifacts exist (initial tables,
epoch-25/50/75 snapshots, final checkpoint).

Epoch-level gaps between two runs are correlated through training and are not
replicates; only the per-seed run-level differences below are treated as
independent observations (n = number of complete seeds, at most 3).

Run: python -m viz.waveform_seed_controls_report  [--out figures/shared/diagnostics/waveform_seed_controls]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

CKPT = Path("checkpoints")
FREEZE_EPOCH = 25
LAST_EPOCH = 99
# Seed-0 runs predate the seeded naming scheme; their compatibility (seed,
# split, model config, batch, lr, epochs, waveform ratio) was verified before reuse.
LEGACY_SEED0 = {
    ("square", "joint"): "xhi2d_lwf_both_ph_square_lr1",
    ("sawtooth", "joint"): "xhi2d_lwf_both_ph_sawtooth_lr1",
    ("square", "adapt25"): "xhi2d_lwf_both_ph_square_adapt25",
    ("sawtooth", "adapt25"): "xhi2d_lwf_both_ph_sawtooth_adapt25",
    ("sine", "frozen"): "xhi2d_lwf_both_ph_sine_frozen",
}
CONDITIONS = [("square", "frozen"), ("square", "adapt25"), ("square", "joint"),
              ("sawtooth", "frozen"), ("sawtooth", "adapt25"), ("sawtooth", "joint"),
              ("sine", "frozen")]
SEEDS = (0, 1, 2)
CONTRASTS = [  # (label, condition A, condition B): reported as A - B
    ("square: frozen - joint", ("square", "frozen"), ("square", "joint")),
    ("square: adapt25 - joint", ("square", "adapt25"), ("square", "joint")),
    ("square: adapt25 - frozen", ("square", "adapt25"), ("square", "frozen")),
    ("sawtooth: frozen - joint", ("sawtooth", "frozen"), ("sawtooth", "joint")),
    ("sawtooth: adapt25 - joint", ("sawtooth", "adapt25"), ("sawtooth", "joint")),
    ("sawtooth: adapt25 - frozen", ("sawtooth", "adapt25"), ("sawtooth", "frozen")),
    ("frozen square - frozen sine", ("square", "frozen"), ("sine", "frozen")),
    ("frozen sawtooth - frozen sine", ("sawtooth", "frozen"), ("sine", "frozen")),
]
METRICS = ("best_val", "final_val", "best_post_freeze_val", "train_at_best", "train_final", "test_at_best")


def run_dir(preset: str, cond: str, seed: int) -> str:
    if seed == 0 and (preset, cond) in LEGACY_SEED0:
        return LEGACY_SEED0[(preset, cond)]
    return f"xhi2d_lwf_both_ph_{preset}_{cond}_s{seed}"


def summarize(preset: str, cond: str, seed: int) -> dict:
    name = run_dir(preset, cond, seed)
    d = CKPT / name
    row = {"preset": preset, "condition": cond, "seed": seed, "run_dir": name}
    metrics = d / "metrics.jsonl"
    if not metrics.is_file():
        row["status"] = "missing"
        return row
    ev = [x for x in (json.loads(l) for l in metrics.open() if l.strip()) if "val_l2" in x]
    if not ev:
        row["status"] = "no evaluations"
        return row
    meta = json.loads((d / "run_metadata.json").read_text()) if (d / "run_metadata.json").is_file() else {}
    best = min(ev, key=lambda x: x["val_l2"])
    post = [x for x in ev if x["epoch"] >= FREEZE_EPOCH]
    bp = min(post, key=lambda x: x["val_l2"]) if post else None
    last = ev[-1]
    done = (d / "final_model_state_dict.pt").is_file() and last["epoch"] >= LAST_EPOCH
    snaps = sorted(p.name[2:5] for p in (d / "snapshots").glob("ep*_model_state_dict.pt")) if (d / "snapshots").is_dir() else []
    row.update({
        "status": "complete" if done else f"running (epoch {last['epoch']})",
        "recorded_run_seed": meta.get("training", {}).get("run_seed"),
        "recorded_split_seed": meta.get("split", {}).get("seed"),
        "best_val": best["val_l2"], "best_epoch": best["epoch"],
        "final_val": last["val_l2"] if done else None,
        "train_at_best": best.get("train_err"), "train_final": last.get("train_err") if done else None,
        "test_at_best": best.get("test_l2"),
        "best_post_freeze_val": bp["val_l2"] if bp else None,
        "best_post_freeze_epoch": bp["epoch"] if bp else None,
        "initial_tables": (d / "waveform_initial_tables.pt").is_file(),
        "snapshots": "/".join(snaps),
        "final_checkpoint": (d / "final_model_state_dict.pt").is_file(),
    })
    return row


def mean_sd(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, sd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=Path("figures/shared/diagnostics/waveform_seed_controls"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [summarize(p, c, s) for p, c in CONDITIONS for s in SEEDS]
    cols = ["preset", "condition", "seed", "run_dir", "status", "recorded_run_seed", "recorded_split_seed",
            "best_val", "best_epoch", "final_val", "train_at_best", "train_final", "test_at_best",
            "best_post_freeze_val", "best_post_freeze_epoch", "initial_tables", "snapshots", "final_checkpoint"]
    with (args.out / "runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    fmt = lambda v, n=4: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
    print(f"{'preset':9s}{'cond':9s}{'seed':>5}  {'status':20s}{'best val':>9}{'@':>4}{'final':>8}{'post-frz':>9}{'@':>4}{'trn@best':>9}{'trn fin':>8}{'test@b':>8}  artifacts")
    for r in rows:
        art = f"init={'y' if r.get('initial_tables') else 'n'} snaps={r.get('snapshots') or '-'} final={'y' if r.get('final_checkpoint') else 'n'}"
        print(f"{r['preset']:9s}{r['condition']:9s}{r['seed']:>5}  {r['status']:20s}{fmt(r.get('best_val')):>9}{fmt(r.get('best_epoch')):>4}"
              f"{fmt(r.get('final_val')):>8}{fmt(r.get('best_post_freeze_val')):>9}{fmt(r.get('best_post_freeze_epoch')):>4}"
              f"{fmt(r.get('train_at_best'),3):>9}{fmt(r.get('train_final'),3):>8}{fmt(r.get('test_at_best')):>8}  {art}")

    index = {(r["preset"], r["condition"], r["seed"]): r for r in rows}
    out_rows = []
    print(f"\nPaired seed contrasts (A - B within each seed; complete runs only). n = seeds with both runs complete.")
    for metric in ("best_val", "final_val", "best_post_freeze_val"):
        print(f"\n  {metric}")
        for label, a, b in CONTRASTS:
            per_seed = {}
            for s in SEEDS:
                ra, rb = index[(*a, s)], index[(*b, s)]
                if ra.get("status") == "complete" and rb.get("status") == "complete" and ra.get(metric) is not None and rb.get(metric) is not None:
                    per_seed[s] = ra[metric] - rb[metric]
            m, sd = mean_sd(list(per_seed.values()))
            signs = {("+" if v > 0 else "-" if v < 0 else "0") for v in per_seed.values()}
            print(f"    {label:32s} per seed {{{', '.join(f's{s}: {v:+.4f}' for s, v in per_seed.items())}}}  "
                  f"mean {fmt(m, 4) if m is not None else '-'}  sd {fmt(sd, 4) if sd is not None else '-'}  n={len(per_seed)}"
                  f"  {'same sign' if len(signs) == 1 and len(per_seed) > 1 else ''}")
            out_rows.append({"metric": metric, "contrast": label, "n": len(per_seed), "mean": m, "sd": sd,
                             **{f"seed{s}": per_seed.get(s) for s in SEEDS}})
    print("\n  seed-to-seed spread within each condition (sd of best_val across complete seeds)")
    for p, c in CONDITIONS:
        vals = [index[(p, c, s)]["best_val"] for s in SEEDS if index[(p, c, s)].get("status") == "complete"]
        m, sd = mean_sd(vals)
        print(f"    {p:9s}{c:9s} n={len(vals)}  mean {fmt(m)}  sd {fmt(sd)}")
    with (args.out / "paired_contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "contrast", "n", "mean", "sd", "seed0", "seed1", "seed2"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {args.out}/runs.csv and {args.out}/paired_contrasts.csv")


if __name__ == "__main__":
    main()
