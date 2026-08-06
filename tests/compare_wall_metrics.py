"""Compare runs on wall placement and sharpness, not just L2.

The wall/expwall/highk terms did not exist when the baseline was trained, so its
metrics.jsonl has nothing to compare against.  This evaluates every run on the
same held-out slices with the same metrics.

L2 is reported but is not the criterion here: the L2-free runs are not
optimising it, and the question is whether their walls land in the right place
and are as sharp as the truth's.

Run: sbatch slurm/compare_wall_metrics.sbatch
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import torch

from losses import ExponentialWallDistance, WallPlacementLoss
from util import field_metrics as fm
from util.slice_eval import gather

N = 512
RUNS = [
    ("baseline pure-L2", "checkpoints/xhi2d_whno_glob_lr3e4"),
    ("h1semi (grad only)", "checkpoints/xhi2d_whno_glob_h1semi"),
    ("expwall scale=16", "checkpoints/xhi2d_whno_glob_expwall16"),
    ("expwall scale=8", "checkpoints/xhi2d_whno_glob_expwall8"),
    ("expwall scale=4", "checkpoints/xhi2d_whno_glob_expwall4"),
]


def main() -> None:
    wall = WallPlacementLoss(cap=32)
    expw = ExponentialWallDistance(scale=8.0, cap=32)
    rows, truth_ref = [], None
    for name, run in RUNS:
        got = None
        for ck in ("final_model_state_dict.pt", "model_state_dict.pt"):
            try:
                got = gather(run, split="test", max_slices=N, checkpoint=ck)
                break
            except Exception:                                   # noqa: BLE001
                continue
        if got is None:
            print(f"{name:<22} (unavailable)")
            continue
        p, t = got.pred.unsqueeze(1), got.truth.unsqueeze(1)
        truth_ref = got.truth
        rows.append((name,
                     float(fm.rmse(got.pred, got.truth)),
                     float(wall(p, t)), float(expw(p, t)),
                     float(fm.width_px(got.pred).mean()),
                     float(fm.blur_frac(got.pred).mean()),
                     float(fm.edge_density(got.pred).mean())))

    print(f"{len(truth_ref)} held-out test slices\n")
    print(f"{'run':<22} {'RMSE':>8} {'wall':>9} {'expwall':>9} "
          f"{'width':>7} {'blur':>7} {'edge_d':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r[0]:<22} {r[1]:8.5f} {r[2]:9.5f} {r[3]:9.5f} "
              f"{r[4]:7.3f} {r[5]:7.4f} {r[6]:8.4f}")
    print("-" * 76)
    print(f"{'TRUTH':<22} {'--':>8} {'--':>9} {'--':>9} "
          f"{float(fm.width_px(truth_ref).mean()):7.3f} "
          f"{float(fm.blur_frac(truth_ref).mean()):7.4f} "
          f"{float(fm.edge_density(truth_ref).mean()):8.4f}")
    print("\nwidth/blur/edge_d closest to TRUTH is the sharpness winner;")
    print("wall and expwall are placement (lower is better).")


if __name__ == "__main__":
    main()
