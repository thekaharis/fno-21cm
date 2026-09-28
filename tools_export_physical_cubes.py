"""Export native-window multi-field predictions for viz.physical_evaluation.

The physical-observable suite (main branch, viz/physical_evaluation.py) reads a
manifest of per-cone npz files (``pred``, ``truth``, ``density``) sharing ONE
redshift grid. Native lightcones each have their own LOS grid, so every cone is
sampled on a common grid by taking, for each grid redshift, the NEAREST native
slice -- never interpolating, so every exported slice keeps its native
transverse structure. The default 512 points over 5.001-24.97 are coarser than
the native spacing everywhere, so no native slice is used twice.

``pred``/``truth`` are x_HI (what the suite evaluates). The model's own
brightness temperature and its target are exported as ``tb_pred``/``tb_truth``
(mK) for evaluations beyond the suite's saturated-spin dT_b. ``cone_id`` is
the true 21cmFAST sample id; ``omega_m`` the cone's sampled OMm.

  python tools_export_physical_cubes.py --checkpoint RUN/best.pt --tag NAME \\
      --out-dir /pfs/.../data/eval_cubes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import fno_multifield as fm
from dataset.lightcone_params import PARAM_NAMES
from dataset.los_windows import LOSWindowConfig, predict_native_cone

SAMPLE_IDS = "experiments/los_windows/preparation_multifield_2000_sample_ids.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--n-z", type=int, default=512)
    ap.add_argument("--z-min", type=float, default=5.001)
    ap.add_argument("--z-max", type=float, default=24.97)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = fm.choose_device(args.device)
    checkpoint, model, dataset, rows = fm.restore(args.checkpoint, device)
    meta = checkpoint["metadata"]
    window = LOSWindowConfig(**meta["sampling"])
    targets = list(dataset.mapping.targets)
    ix, it = targets.index("neutral_fraction"), targets.index("brightness_temp")
    ids = {r["row"]: r["sample_id"] for r in json.loads(Path(SAMPLE_IDS).read_text())["rows"]}
    z_grid = np.linspace(args.z_min, args.z_max, args.n_z)
    out = args.out_dir/args.tag
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    split_rows = rows[args.split][:args.max_cones]
    try:
        for k, row in enumerate(split_rows):
            z = np.asarray(dataset.redshifts[row])
            if z[0] > args.z_min + 1e-3 or z[-1] < args.z_max - 1e-3:
                raise ValueError(f"row {row} does not cover the export grid")
            pick = np.abs(z[None, :] - z_grid[:, None]).argmin(axis=1)
            if len(np.unique(pick)) != len(pick):
                raise ValueError("export grid finer than the native spacing (repeated slices)")
            pred = predict_native_cone(model, dataset, row, window, device)[..., pick].numpy()
            norm = dataset.normalization
            phys = {name: pred[i]*norm[name]["scale"] + norm[name]["offset"] for i, name in enumerate(targets)}
            truth = dataset.read_fields(row, names=["neutral_fraction", "brightness_temp", "density"])
            sid = int(ids[int(row)])
            path = out/f"cone_{sid:06d}.npz"
            np.savez(path,
                     pred=phys["neutral_fraction"].astype(np.float32),
                     truth=truth["neutral_fraction"][..., pick].astype(np.float32),
                     density=truth["density"][..., pick].astype(np.float32),
                     tb_pred=phys["brightness_temp"].astype(np.float32),
                     tb_truth=truth["brightness_temp"][..., pick].astype(np.float32),
                     z_native=z[pick])
            omm = float(dataset.params[row][PARAM_NAMES.index("OMm")])
            entries.append({"npz": str(path.resolve()), "cone_id": sid, "row": int(row), "omega_m": omm})
            print(f"[{k+1}/{len(split_rows)}] sample {sid} (row {row}): max |z_native - z_grid| "
                  f"{np.abs(z[pick]-z_grid).max():.4f}", flush=True)
    finally:
        dataset.close()
    manifest = {"z_grid": z_grid.tolist(), "models": {args.tag: entries},
                "checkpoint": str(Path(args.checkpoint).resolve()), "epoch": checkpoint["epoch"],
                "split": args.split, "selection": "nearest native slice per grid redshift",
                "fields": {"pred/truth": "neutral_fraction", "tb_pred/tb_truth": "brightness_temp [mK]",
                           "density": "overdensity (input)"}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(entries)} cones and {out/'manifest.json'}")


if __name__ == "__main__":
    main()
