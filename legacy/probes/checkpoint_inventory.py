"""Inventory every checkpoint directory and rank runs within each task.

Walks checkpoints/ and checkpoint-archive/, classifies each run by task, pulls
whatever metrics that task's era recorded, and writes one sorted table per task.

Three things this deliberately does NOT do.

*It does not compare across tasks.* 2-D slices, 3-D cubes, warped-grid cubes and
z_re maps have different targets, different shapes and different normalisations;
their L2 columns are not the same quantity. Ranking is within a task only.

*It does not mix absolute and relative L2.* The 3-D runs trained under
``loss_modes={'l2': 'relative'}`` and log both ``val_l2`` (absolute) and
``val_l2_rel``; the 2-D runs log absolute only. Both are shown for 3-D, and the
mode used for training is named.

*It does not infer a grid variant it cannot verify.* 3-D metadata never recorded
which cube cache was used, so warped/envelope is read off the directory name and
flagged as such rather than presented as a recorded fact.

    python -m util.checkpoint_inventory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOTS = ("checkpoints", "checkpoint-archive")
OUT_MD = "figures/checkpoint_inventory.md"
OUT_CSV = "figures/checkpoint_inventory.csv"

OP_TAG = {"fourier": "fno", "wavelet": "wno", "hadamard": "whno",
          "siren_fourier": "sirenfno", "cnn": "cnn"}
# Runs that exist only to check the pipeline, not to be ranked.
SMOKE = ("smoke", "bench", "_valonly", "wrongLR", "fail")


def classify(name: str, meta: dict) -> str:
    n = name.lower()
    task = (meta.get("task") or "").lower()
    if task == "zre" or "zre" in n:
        return "z_re maps (2-D)"
    if task == "2d" or n.startswith("xhi2d"):
        return "x_HI slices (2-D)"
    if "warped" in n:
        return "x_HI cubes (3-D, warped grid)"
    if "envelope" in n:
        return "x_HI cubes (3-D, envelope grid)"
    if "3d" in n or "ufno" in n:
        return "x_HI cubes (3-D, uniform grid)"
    return "unclassified"


def arch(meta: dict, name: str = "") -> str:
    """Architecture from metadata, falling back to the directory name.

    The oldest archive runs have no run_metadata.json at all, so nothing is
    recorded. Their names do say which architecture they were, but a name is
    not a record -- inferred values are marked so they are never mistaken for
    logged ones.
    """
    mc = meta.get("model_config", {}) or {}
    kind = mc.get("kind")
    if not kind:
        n = name.lower()
        for token, label in (("localsirenfno", "local sirenfno / global sirenfno"),
                             ("localwhno", "local whno / global fno"),
                             ("localwno", "local wno / global fno"),
                             ("localfno", "local fno / global fno"),
                             ("sirenfno", "SirenFNO"), ("ufno", "U-FNO"),
                             ("fno", "FNO")):
            if token in n:
                return f"{label} ?"
        return "unrecorded"
    
    if kind == "localop":
        return (f"local {OP_TAG.get(mc.get('local_operator'), '?')}"
                f" / global {OP_TAG.get(mc.get('global_operator'), '?')}")
    return {"ufno": "U-FNO", "fno": "FNO",
            "localfno": "local fno / global fno",
            "localwno": "local wno / global fno",
            "localwhno": "local whno / global fno",
            "localsirenfno": "local sirenfno / global sirenfno",
            "sirenfno": "SirenFNO"}.get(kind, kind)


def loss_label(meta: dict) -> str:
    tr = meta.get("training", {}) or {}
    w = tr.get("loss_weights")
    if not isinstance(w, dict):
        return "unrecorded"
    names = {"l2": "L2", "h1": "H1", "bce": "BCE", "swd": "SWD",
             "highk": "highK", "wall": "wall", "h1semi": "H1semi",
             "expwall": "expwall", "ionized_wall": "ionWall"}
    parts = [f"{v:g}*{names.get(k, k)}" for k, v in sorted(w.items()) if v]
    out = " + ".join(parts) or "?"
    modes = tr.get("loss_modes")
    if isinstance(modes, dict) and set(modes.values()) == {"relative"}:
        out += " (rel)"
    elif isinstance(modes, dict) and "relative" in modes.values():
        out += " (mixed abs/rel)"
    mc = meta.get("model_config", {}) or {}
    if mc.get("contrast_mode", "off") not in ("off", None):
        out += f" [contrast {mc['contrast_mode']}/" \
               f"{mc.get('contrast_schedule_kind', 'sigmoid')}]"
    return out


def eval_rows(run: Path) -> list[dict]:
    p = run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and any(k.startswith("val_") for k in r):
            out.append(r)
    return out


def finite(v):
    return isinstance(v, (int, float)) and v == v


def collect(run: Path) -> dict | None:
    meta_p = run / "run_metadata.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    rows = eval_rows(run)
    report_p = run / "final_report.json"
    report = json.loads(report_p.read_text()) if report_p.exists() else {}
    if not rows and not report:
        return None

    rec = {
        "run": run.name, "root": run.parent.name,
        "task": classify(run.name, meta),
        "architecture": arch(meta, run.name),
        "train_loss": loss_label(meta),
        "epochs": (meta.get("training", {}) or {}).get("epochs"),
        "smoke": any(s.lower() in run.name.lower() for s in SMOKE),
        "complete": (run / "final_model_state_dict.pt").is_file(),
    }
    if rows:
        best = min((r for r in rows if finite(r.get("val_l2"))),
                   key=lambda r: r["val_l2"], default=None)
        last = rows[-1]
        rec["best_val_l2"] = best["val_l2"] if best else None
        rec["best_epoch"] = best["epoch"] if best else None
        rec["last_epoch"] = last.get("epoch")
        for k in ("val_l2", "test_l2", "val_l2_rel", "test_l2_rel", "val_h1",
                  "test_h1", "val_bce", "val_ionized_wall", "val_pred_mean",
                  "val_pred_std", "val_wall", "val_expwall", "val_highk",
                  "val_swd"):
            if finite(last.get(k)):
                rec[k] = last[k]
    for k, v in report.items():
        if finite(v) and k.startswith(("val_", "test_")):
            rec.setdefault(k, v)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()

    recs = []
    for root in ROOTS:
        for run in sorted(Path(root).glob("*")):
            if not run.is_dir():
                continue
            try:
                r = collect(run)
            except Exception as exc:                            # noqa: BLE001
                print(f"  [skip] {run}: {type(exc).__name__}: {exc}")
                continue
            if r:
                recs.append(r)
    print(f"{len(recs)} runs with recorded metrics "
          f"({sum(r['smoke'] for r in recs)} smoke/bench)\n")

    ranked = [r for r in recs if args.include_smoke or not r["smoke"]]
    tasks = sorted({r["task"] for r in ranked})
    os.makedirs("figures", exist_ok=True)

    # per-task metric columns, in the order they should be shown
    COLS = {
        "x_HI slices (2-D)": [("best_val_l2", "best val L2"), ("test_l2", "test L2"),
                              ("test_rmse", "test RMSE"),
                              ("test_gradient_rmse", "grad RMSE"),
                              ("test_high_k_power_ratio", "highK ratio"),
                              ("test_high_k_cross_correlation", "highK xcorr"),
                              ("test_mean_xhi_mae", "mean x_HI MAE")],
        "z_re maps (2-D)": [("val_mse_norm", "val MSE"), ("test_mse_norm", "test MSE"),
                            ("test_mse_masked_norm", "test MSE masked"),
                            ("test_rmse_z", "test RMSE z"),
                            ("test_rmse_masked_z", "RMSE z masked")],
    }
    THREE_D = [("best_val_l2", "best val L2"), ("test_l2", "test L2"),
               ("val_l2_rel", "val L2 rel"), ("test_h1", "test H1"),
               ("val_bce", "val BCE"), ("val_ionized_wall", "ion wall"),
               ("val_pred_mean", "pred mean"), ("val_pred_std", "pred std")]

    lines = ["# Checkpoint inventory\n",
             f"All {len(recs)} runs with recorded metrics across "
             f"`checkpoints/` and `checkpoint-archive/`.\n",
             "**Rankings are within a task only.** The four tasks have "
             "different targets, shapes and normalisations, so their L2 columns "
             "are not the same quantity and must not be compared across "
             "sections.\n",
             "An architecture marked with a trailing `?` was read off the "
             "directory name because the run has no `run_metadata.json` -- the "
             "oldest archive runs predate metadata recording. `unrecorded` "
             "means not even the name says.\n",
             "3-D runs mostly trained under a *relative* L2 (marked `(rel)` in "
             "the loss column) while logging absolute `val_l2`; both are shown. "
             "Grid variant for 3-D runs comes from the directory name -- the "
             "metadata never recorded which cube cache was used.\n"]

    for task in tasks:
        group = [r for r in ranked if r["task"] == task]
        key = ("val_mse_norm" if task.startswith("z_re") else "best_val_l2")
        have = [r for r in group if finite(r.get(key))]
        rest = [r for r in group if not finite(r.get(key))]
        have.sort(key=lambda r: r[key])
        cols = COLS.get(task, THREE_D)
        lines.append(f"\n## {task}  ({len(group)} runs)\n")
        hdr = "| run | architecture | train loss | ep |" + \
              "".join(f" {lab} |" for _, lab in cols)
        lines.append(hdr)
        lines.append("|---|---|---|---:|" + "---:|" * len(cols))
        print(f"=== {task}  ({len(group)} runs, ranked by "
              f"{'val MSE' if task.startswith('z_re') else 'best val L2'}) ===")
        w = max((len(r["run"]) for r in group), default=10)
        print(f"{'run':<{w}} {'architecture':<26} {'train loss':<26} " +
              " ".join(f"{lab:>13}" for _, lab in cols[:4]))
        for r in have + rest:
            cells = []
            for k, _ in cols:
                v = r.get(k)
                cells.append(f"{v:.5f}" if finite(v) else "--")
            flag = "" if r["complete"] else " *(no final ckpt)*"
            lines.append(f"| `{r['run']}`{flag} | {r['architecture']} | "
                         f"{r['train_loss']} | {r.get('last_epoch', '--')} |" +
                         "".join(f" {c} |" for c in cells))
            print(f"{r['run']:<{w}} {r['architecture']:<26} "
                  f"{r['train_loss'][:25]:<26} " +
                  " ".join(f"{c:>13}" for c in cells[:4]))
        print()

    Path(OUT_MD).write_text("\n".join(lines) + "\n")
    keys = sorted({k for r in recs for k in r})
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(recs)
    print(f"wrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
