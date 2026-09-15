"""Boundary-band edge metrics for 2-D x_HI runs.

`slurm/edge_metrics_eval.sbatch` is hardwired to the 3-D cube cache, so it
cannot score a 2-D slice model. This driver reuses the same engine --
`viz.boundary_band_diagnostic` in its 2-D transverse `mode="slice"`, which is
the primary diagnostic there anyway -- against the 2-D slice cache.

Slices of one cone are stacked along axis 2 and passed as a pseudo-cube; in
slice mode each plane is treated independently, so this is exactly the intended
per-slice transverse geometry and no LOS spacing is implied.

  python -m viz.edge_metrics_xhi2d --runs lap_p4=checkpoints/... fno_fno=... \
      --out figures/2d_xhi/eval/lap_p4_edge
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset.dataset_3d import ParameterNormalization
from dataset.slices import SliceCache
from viz.boundary_band_diagnostic import (
    BandConfig, BoundaryBandAccumulator, front_width, plot_overlay, write_csv,
)
from viz.ps_coherence_highk import load_run, predict_all


def parse_run(text):
    label, _, path = text.partition("=")
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=CHECKPOINT_DIR")
    return label, Path(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=parse_run, required=True)
    ap.add_argument("--n-slices", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("figures/2d_xhi/eval/edge_metrics"))
    args = ap.parse_args(argv)

    runs = [r for r in (load_run(label, path) for label, path in args.runs) if r]
    if not runs:
        raise SystemExit("no loadable runs")

    ref = runs[0].metadata
    cache = SliceCache(ref["dataset"]["cache_file"],
                       input_features=ref["input_features"]["name"],
                       parameter_normalization=ParameterNormalization.from_dict(
                           ref["parameter_normalization"])
                       if ref.get("parameter_normalization") else None)
    # Split by CONE, as the 2-D pipeline records it: selecting rows directly
    # would leak correlated slices of one lightcone across splits.
    test_cones = np.asarray(ref["split"]["test_cone_ids"], dtype=np.int64)
    test_idx = np.flatnonzero(np.isin(cache.cone_id, test_cones))
    if args.n_slices < len(test_idx):
        test_idx = test_idx[np.linspace(0, len(test_idx) - 1, args.n_slices).astype(int)]
    items = [cache[int(i)] for i in test_idx]
    cone_of = np.asarray(cache.cone_id)[test_idx]
    inputs = torch.stack([it["x"] for it in items])
    truth = np.stack([it["y"].numpy()[0] for it in items])
    print(f"[edge] {len(items)} test slices from {len(set(cone_of.tolist()))} cones, "
          f"shape {truth.shape[-2:]}")

    # 1.428571 Mpc transverse pixel (200 Mpc / 140 cells).
    pix = 200.0 / 140.0
    cfg = BandConfig(mode="slice", threshold=args.threshold, dx_mpc=pix, dy_mpc=pix)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows_by_cone = defaultdict(list)
    for r, c in enumerate(cone_of):
        rows_by_cone[int(c)].append(r)

    results, accums = {}, {}
    for run in runs:
        pred = predict_all(run, inputs, device)
        acc = BoundaryBandAccumulator(cfg=cfg)
        for cone, rows in rows_by_cone.items():
            # (Nx, Ny, n_slices_of_this_cone): slice mode treats axis 2 planes
            # independently, so stacking is safe and implies no LOS geometry.
            p = np.stack([pred[i] for i in rows], axis=-1)
            t = np.stack([truth[i] for i in rows], axis=-1)
            acc.add_cone(cone, p, t)
        prof = acc.profile()
        summary = acc.band_summary()
        fw = front_width(np.asarray(prof["d_mpc"]), np.asarray(prof["mean_truth"]))
        fw_pred = front_width(np.asarray(prof["d_mpc"]), np.asarray(prof["mean_pred"]))
        summary["front_width_truth_mpc"] = fw
        summary["front_width_pred_mpc"] = fw_pred
        # plot_overlay/write_csv expect the profile nested under "profile",
        # not flattened into the summary.
        results[run.label] = {"profile": prof, **summary}
        accums[run.label] = acc
        print(f"  {run.label:12s} front width pred {fw_pred:7.3f} Mpc "
              f"(truth {fw:6.3f})  total sq err {acc.total_sq_err:.4e}")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_overlay(results, cfg, args.out / "edge_band_overlay.png")
    write_csv(results, accums, args.out / "edge_band_metrics.csv")
    (args.out / "summary.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
         for k, v in results.items()}, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
