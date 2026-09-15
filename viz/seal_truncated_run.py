"""Make a wall-clock-truncated run readable by the eval tools.

The 2-D eval tools gate on `final_report.json` + `final_model_state_dict.pt`,
which only a run that reaches its epoch budget writes. A run killed by the Slurm
wall limit leaves only periodic weights, so every downstream analysis skips it --
`viz.ps_coherence_highk` does so *silently*, reporting success while scoring
only the other model.

This copies a run into a snapshot directory with those two artifacts present:

* `final_model_state_dict.pt` is a copy of the periodic checkpoint. It holds the
  weights of the epoch named in `manifest.pt`, which is NOT necessarily the
  run's best epoch -- both are recorded in the report.
* `final_report.json` carries the metrics actually logged at that epoch in
  `metrics.jsonl`, under their real names. Nothing is recomputed or aliased:
  in particular `val_l2` (neuralop absolute Lp) is not renamed to `val_rmse`,
  so a tool that needs RMSE fails loudly instead of reading a wrong number.

Provenance fields mark the snapshot as reconstructed so it is never mistaken
for a completed run.

  python -m viz.seal_truncated_run --run checkpoints/... --out checkpoints/..._eval
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    manifest = torch.load(args.run / "manifest.pt", map_location="cpu", weights_only=True)
    ckpt_epoch = int(manifest.get("epoch", -1))
    weights = args.run / manifest.get("model", "model_state_dict.pt")
    if not weights.is_file():
        raise SystemExit(f"no weights at {weights}")

    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()
            if line.strip()]
    by_epoch = {int(r["epoch"]): r for r in rows if r.get("epoch") is not None}
    at = by_epoch.get(ckpt_epoch)
    if at is None:
        raise SystemExit(f"metrics.jsonl has no row for checkpoint epoch {ckpt_epoch}")
    scored = [(r["val_l2"], int(r["epoch"])) for r in rows if r.get("val_l2") is not None]
    best_val, best_epoch = min(scored)
    last_epoch = max(by_epoch)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("run_metadata.json", "metrics.jsonl", "manifest.pt"):
        shutil.copy2(args.run / name, args.out / name)
    shutil.copy2(weights, args.out / "model_state_dict.pt")
    shutil.copy2(weights, args.out / "final_model_state_dict.pt")

    report = {k: v for k, v in at.items() if isinstance(v, (int, float))}
    report.update({
        "reconstructed": True,
        "reconstructed_reason": "run truncated by the Slurm wall limit; no final_report.json was written",
        "source_run": str(args.run),
        "checkpoint_epoch": ckpt_epoch,
        "last_logged_epoch": last_epoch,
        "best_val_l2": best_val,
        "best_val_l2_epoch": best_epoch,
        "weights_are_best_epoch": ckpt_epoch == best_epoch,
        "note": ("metrics are the logged values at checkpoint_epoch, not recomputed; "
                 "val_l2 is neuralop absolute Lp and is deliberately not aliased to val_rmse"),
    })
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2))

    # The loaders reject artifacts older than run_metadata.json as possibly stale.
    for name in ("final_report.json", "final_model_state_dict.pt", "model_state_dict.pt"):
        (args.out / name).touch()

    print(f"sealed {args.run} -> {args.out}")
    print(f"  weights epoch {ckpt_epoch} (val_l2 {at.get('val_l2'):.4f}); "
          f"best was {best_val:.4f}@{best_epoch}; last logged {last_epoch}")
    if ckpt_epoch != best_epoch:
        print(f"  NOTE: these are NOT the best-epoch weights "
              f"(+{at.get('val_l2') - best_val:.4f} val_l2)")


if __name__ == "__main__":
    main()
