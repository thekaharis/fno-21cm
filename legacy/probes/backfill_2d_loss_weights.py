"""Repair loss_weights for 2-D runs trained before all terms were recorded.

fno_21cm.py recorded only l2/h1/bce/swd/highk, so runs using wall, h1semi or
expwall stored all-zero weights and cannot say which loss produced them. The
job log's "Loss:" line is the surviving record; logs are transient, checkpoint
dirs are not.

Dry run by default; pass --write to apply.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Longest alternative first: "H1" would otherwise match the prefix of
# "H1semi" and mislabel the gradient-only run as an H1 run.
TERM = re.compile(r"([\d.]+)\*(?:abs)?(H1semi|expwall|highK|wall|BCE|SWD|L2|H1)",
                  re.IGNORECASE)
KEY = {"l2": "l2", "h1": "h1", "bce": "bce", "swd": "swd", "highk": "highk",
       "wall": "wall", "h1semi": "h1semi", "expwall": "expwall"}


def parse_log(path: Path) -> tuple[dict, str] | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"^Loss: (\d.*)$", text, re.M)
    d = re.search(r"^\[Rank 0\]: saved training state to (\S+)", text, re.M)
    if not m or not d:
        return None
    weights = {KEY[name.lower()]: float(w) for w, name in TERM.findall(m.group(1))}
    return weights, d.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    found: dict[str, dict] = {}
    for log in sorted((ROOT / "logs").glob("*.out")):
        got = parse_log(log)
        if got:
            found.setdefault(got[1], got[0])       # first log wins per dir
    for rel, weights in sorted(found.items()):
        meta_path = ROOT / rel / "run_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        current = meta.get("training", {}).get("loss_weights", {})
        if any(current.values()):
            continue                               # already self-describing
        meta.setdefault("training", {})["loss_weights"] = weights
        meta["loss_weights_backfilled_from_log"] = True
        active = {k: v for k, v in weights.items() if v}
        print(f"{'WRITE' if args.write else 'DRY  '} {rel}: {active}")
        if args.write:
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
