"""Backfill run_metadata.json for z_re runs from their SLURM logs.

Runs launched before fno_zre.py recorded the full architecture (and before
it recorded metadata at all) are indistinguishable in the dashboard: the
sweep variables lived only in the job log's "Model:" line.  Logs are
transient; checkpoint dirs are not.  This reconstructs the metadata into
the checkpoint dir so the runs stay self-describing.

Dry run by default; pass --write to apply:
    python -m util.backfill_zre_metadata [--write]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CKPT_RE = re.compile(r"Checkpoint dir:\s*(\S+)|checkpoints/(checkpoints_zre\S*)")
MODEL_RE = re.compile(r"Model:\s*(\S+)\s+(.*?)\s*->\s*([\d,]+)\s*parameters")
LOSS_RE = re.compile(r"Loss:\s*([\d.]+)\*(abs|rel)L2\s*\+\s*([\d.]+)\*(abs|rel)H1")
LR_RE = re.compile(r"Batch size:\s*(\d+),\s*LR:\s*([\d.e+-]+),\s*epochs:\s*(\d+)")
KIND_RE = re.compile(r"Model kind:\s*(\w+)")
CLIP_RE = re.compile(r"Gradient clipping:\s*max_norm=([\d.]+)")
TARGET_RE = re.compile(r"Target kind:\s*(\w+)")
FEATURES_RE = re.compile(r"Input features:\s*(\w+)")

# "window=(16, 16) local-modes=(6, 6) global-modes=(16, 16) widths=16/32/64
#  rank=16 siren=64x1 ff=16@128 sigmoid-output"
DESC_PATTERNS = {
    "localfno_window": r"window=\(([\d, ]+)\)",
    "localfno_modes": r"local-modes=\(([\d, ]+)\)",
    "localfno_global_modes": r"global-modes=\(([\d, ]+)\)",
    "n_modes": r"modes=\(([\d, ]+)\)",
    "localfno_spectral_rank": r"rank=(\d+)",
    "ufno_width": r"width=(\d+)",
    "hidden_channels": r"hidden=(\d+)",
    "n_layers": r"layers=(\d+)",
}


def parse_log(path: Path) -> dict | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    m = MODEL_RE.search(text)
    if not m:
        return None
    desc = m.group(2)
    mc: dict = {}
    kind = KIND_RE.search(text)
    if kind:
        mc["kind"] = kind.group(1)
    else:
        guess = {"LocalSirenFNO2d": "localsirenfno", "LocalWNO2d": "localwno",
                 "LocalFNO2d": "localfno",
                 "U-FNO2d": "ufno", "SirenFNO2d": "sirenfno",
                 "FNO2d": "fno"}.get(m.group(1))
        if guess:
            mc["kind"] = guess
    widths = re.search(r"widths=(\d+)/", desc)
    if widths:
        mc["localfno_base_width"] = int(widths.group(1))
    siren = re.search(r"siren=(\d+)x(\d+)", desc)
    if siren:
        mc["siren_hidden_dim"] = int(siren.group(1))
        mc["siren_n_hidden"] = int(siren.group(2))
    ff = re.search(r"ff=(\d+)@([\d.]+)", desc)
    if ff:
        mc["siren_feature_dim"] = int(ff.group(1))
        mc["siren_ff_sigma"] = float(ff.group(2))
    if "sigmoid-output" in desc:
        mc["output_sigmoid"] = True
    elif "linear-output" in desc:
        mc["output_sigmoid"] = False
    for key, pattern in DESC_PATTERNS.items():
        hit = re.search(pattern, desc)
        if not hit:
            continue
        raw = hit.group(1)
        mc[key] = ([int(v) for v in raw.split(",")] if "," in raw
                   else int(raw))
    tr: dict = {}
    lr = LR_RE.search(text)
    if lr:
        tr["batch_size"] = int(lr.group(1))
        tr["learning_rate"] = float(lr.group(2))
        tr["epochs"] = int(lr.group(3))
    loss = LOSS_RE.search(text)
    if loss:
        tr["loss_weights"] = {"l2": float(loss.group(1)),
                              "h1": float(loss.group(3))}
        tr["loss_relative"] = loss.group(2) == "rel"
    clip = CLIP_RE.search(text)
    tr["grad_clip_norm"] = float(clip.group(1)) if clip else 0.0
    tk = TARGET_RE.search(text)
    if tk:
        tr["target_kind"] = tk.group(1)
    ft = FEATURES_RE.search(text)
    if ft:
        tr["input_features"] = ft.group(1)
    dirs = re.findall(r"(checkpoints/checkpoints_zre\S*?)/", text) or \
           re.findall(r"saved training state to (\S+)", text)
    return {"model_config": mc, "training": tr, "params": m.group(3),
            "dirs": dirs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    parsed: dict[str, dict] = {}
    # log basenames vary by job (fno-zre-*, localsirenfno-zre-*, ...), so
    # select by content instead of filename.
    for log in sorted((ROOT / "logs").glob("*.out")):
        try:
            head = log.read_text(errors="replace")[:4000]
        except OSError:
            continue
        if "checkpoints_zre" not in head and "Model kind:" not in head:
            continue
        info = parse_log(log)
        if not info or not info["dirs"]:
            continue
        for d in set(info["dirs"]):
            parsed.setdefault(d, info)          # first log wins per dir

    for rel, info in sorted(parsed.items()):
        target = ROOT / rel
        if not target.is_dir():
            continue
        meta_path = target / "run_metadata.json"
        existing = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        merged = dict(existing)
        merged.setdefault("task", "zre")
        added = [k for k in info["model_config"]
                 if k not in existing.get("model_config", {})]
        # never overwrite values the training script itself recorded
        mc = dict(info["model_config"]); mc.update(existing.get("model_config", {}))
        tr = dict(info["training"]); tr.update(existing.get("training", {}))
        merged["model_config"], merged["training"] = mc, tr
        merged["metadata_backfilled_from_log"] = True
        # SIREN omega was never printed to the log; for the historical sweep
        # the directory-name suffix (..._om60) is the only surviving record.
        # Mark it so an inferred value is never mistaken for a logged one.
        if "siren_omega" not in mc:
            om = re.search(r"om(\d+)", target.name)
            if om:
                mc["siren_omega"] = float(om.group(1))
                merged["siren_omega_inferred_from_dirname"] = True
                added.append("siren_omega(inferred)")
        if not added and existing:
            continue
        print(f"{'WRITE' if args.write else 'DRY  '} {rel}: +{added}")
        if args.write:
            meta_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
