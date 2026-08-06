#!/usr/bin/env python3
"""Representative-slice viz for an IN-PROGRESS 2-D x_HI run.

viz.visualize_xhi2d_representative requires final_model_state_dict.pt and
final_report.json, which only exist once training finishes. This is the same
rendering path pointed at the periodic model_state_dict.pt instead, for a
qualitative look at a run that is still training. Labeled "(epoch N, live)"
in the figure and filenames so it can never be mistaken for a final result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from viz.compare_xhi2d_models import Run, build_model, representative_indices
from viz.visualize_xhi2d_representative import DEFAULT_QUANTILES, render_slices
from modeling import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=DEFAULT_QUANTILES)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()

    metadata_path = args.run / "run_metadata.json"
    checkpoint_path = args.run / "model_state_dict.pt"
    if not metadata_path.is_file():
        raise SystemExit(f"no run_metadata.json in {args.run}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"no model_state_dict.pt in {args.run} (not started?)")

    metadata = json.loads(metadata_path.read_text())
    # The periodic checkpoint lags the metrics: the trainer saves every
    # `save_every` epochs, so metrics.jsonl is typically several epochs ahead
    # of what is actually in model_state_dict.pt. manifest.pt records the
    # epoch the weights belong to, which is the only honest label here.
    epoch = None
    manifest_path = args.run / "manifest.pt"
    if manifest_path.is_file():
        try:
            epoch = torch.load(
                manifest_path, map_location="cpu", weights_only=False
            ).get("epoch")
        except Exception:  # noqa: BLE001 - fall back to the metrics tail
            epoch = None
    latest_metric_epoch = None
    metrics_path = args.run / "metrics.jsonl"
    if metrics_path.is_file():
        lines = [l for l in metrics_path.read_text().splitlines() if l.strip()]
        if lines:
            latest_metric_epoch = json.loads(lines[-1]).get("epoch")
    if epoch is None:
        epoch = latest_metric_epoch
    tag = f"epoch {epoch}" if epoch is not None else "unknown epoch"
    behind = (
        "" if latest_metric_epoch is None or latest_metric_epoch == epoch
        else f"; training has since reached epoch {latest_metric_epoch}"
    )
    print(f"[live] {args.run.name}: visualizing checkpoint at {tag} "
          f"(training not yet complete{behind})")

    run = Run(label=f"{args.run.name} ({tag}, live)", path=args.run,
             metadata=metadata, report={})

    model = build_model(metadata["model_config"])
    result = load_checkpoint(model, checkpoint_path)
    if result.missing or result.unexpected or result.matched != result.total:
        raise RuntimeError(
            f"incomplete checkpoint load: matched={result.matched}/"
            f"{result.total}, missing={result.missing}, "
            f"unexpected={result.unexpected}"
        )
    model = model.to(args.device).eval()

    normalization = metadata.get("parameter_normalization")
    cache = SliceCache(
        metadata["dataset"]["cache_file"],
        input_features=metadata["input_features"]["name"],
        parameter_normalization=(
            ParameterNormalization.from_dict(normalization)
            if normalization is not None else None
        ),
    )
    test_cones = np.asarray(metadata["split"]["test_cone_ids"], dtype=np.int64)
    test_indices = np.flatnonzero(np.isin(cache.cone_id, test_cones))
    indices = representative_indices(cache, test_indices, args.quantiles)
    samples = [cache[index] for index in indices]
    inputs = torch.stack([sample["x"] for sample in samples])
    truth = torch.stack([sample["y"] for sample in samples]).numpy()[:, 0]

    with torch.inference_mode():
        prediction = model(inputs.to(args.device)).cpu().numpy()[:, 0]
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    slice_info = [
        {
            "quantile": float(quantile),
            "global_index": int(index),
            "cone_id": int(cache.cone_id[index]),
            "z": float(cache.z[index]),
            "xhi_mean": float(cache.xHI_mean[index]),
            "rmse": float(np.sqrt(np.mean(
                (prediction[row] - truth[row]) ** 2
            ))),
            "mae": float(np.mean(np.abs(prediction[row] - truth[row]))),
        }
        for row, (quantile, index) in enumerate(zip(args.quantiles, indices))
    ]

    output = args.output or Path(
        f"figures/{args.run.name}_representative_LIVE_ep{epoch}"
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "representative_z_slices.png"
    render_slices(truth, prediction, slice_info, run.label, figure_path)
    (output / "selected_slices.json").write_text(
        json.dumps(slice_info, indent=2) + "\n"
    )
    (output / "run_summary.json").write_text(json.dumps({
        "run": str(args.run.resolve()),
        "model_kind": run.kind,
        "epoch": epoch,
        "checkpoint_epoch_source": "manifest.pt",
        "live": True,
        "note": "checkpoint from an in-progress run, not the final model",
        "figure": str(figure_path),
    }, indent=2) + "\n")
    print(f"Live representative slice visualization: {figure_path}")


if __name__ == "__main__":
    main()
