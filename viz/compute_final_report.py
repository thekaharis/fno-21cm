"""Compute a genuine final_report.json for a wall-truncated run.

`viz.seal_truncated_run` makes the artifacts *exist*, but the 2-D eval tools
also validate that the report contains the metric keys a completed run would
have (val_rmse, test_gradient_rmse, ...). Those are RMSE-family metrics that the
per-epoch log does not carry -- `val_l2` is neuralop's absolute Lp and is not
the same quantity, so aliasing it would put a wrong number under a right name.

This runs the trainer's own `final_report` over the run's recorded split, so the
numbers are computed, not reconstructed. Only the weights differ from a
completed run: they are the last periodic checkpoint, named in the report.

  python -m viz.compute_final_report --run checkpoints/..._eval_snapshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.dataset_3d import ParameterNormalization
from dataset.slices import SliceCache, Subset
from fno_xhi2d import final_report
from viz.ps_coherence_highk import rebuild
from modeling import load_checkpoint
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    meta = json.loads((args.run / "run_metadata.json").read_text())
    cache = SliceCache(meta["dataset"]["cache_file"],
                       input_features=meta["input_features"]["name"],
                       parameter_normalization=ParameterNormalization.from_dict(
                           meta["parameter_normalization"])
                       if meta.get("parameter_normalization") else None)
    cone = np.asarray(cache.cone_id)
    split = meta["split"]
    sets = {}
    for name, key in (("val", "val_cone_ids"), ("test", "test_cone_ids")):
        ids = split.get(key)
        if ids is None:
            raise SystemExit(f"run metadata has no {key}; cannot reproduce the split")
        idx = np.flatnonzero(np.isin(cone, np.asarray(ids, dtype=np.int64)))
        sets[name] = Subset(cache, idx.tolist())
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False)
               for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = rebuild(meta)
    manifest = torch.load(args.run / "manifest.pt", map_location="cpu", weights_only=True)
    load_checkpoint(model, args.run / "final_model_state_dict.pt")
    model = model.to(device)

    report = final_report(model, loaders, device)
    # Keys the tools require but that only a train-split pass would produce.
    for k in list(report):
        if k.startswith("val_"):
            report.setdefault("train_" + k[4:], float("nan"))
    existing = json.loads((args.run / "final_report.json").read_text()) \
        if (args.run / "final_report.json").is_file() else {}
    report.update({k: v for k, v in existing.items() if not isinstance(v, (int, float))})
    report.update({
        "computed_by": "viz.compute_final_report",
        "checkpoint_epoch": int(manifest.get("epoch", -1)),
        "weights_note": "last periodic checkpoint of a wall-truncated run, not the best epoch",
    })
    (args.run / "final_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.run/'final_report.json'}")
    for k in ("val_rmse", "test_rmse", "test_gradient_rmse", "test_mean_xhi_mae",
              "test_high_k_power_ratio", "test_high_k_cross_correlation"):
        print(f"  {k:32s} {report.get(k)}")


if __name__ == "__main__":
    main()
