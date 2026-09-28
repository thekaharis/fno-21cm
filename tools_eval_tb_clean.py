"""Evaluate a multi-field checkpoint against the dv/dr-clipped brightness_temp.

Models trained on the raw targets are scored on the raw targets, where a few
divergent velocity-caustic voxels (up to ~1e6 mK) dominate T_b's squared
error. To compare them fairly with runs trained on the clipped targets
(tools_clean_brightness_temp.py), this re-scores any multi-field checkpoint on
the cleaned preparation:

  * inputs come from the cleaned dataset -- density, velocity, x_HI and their
    normalization are identical to the raw preparation, and the same history
    emulators are installed from the checkpoint metadata, so the model sees
    exactly what it was trained on;
  * targets are the cleaned brightness_temp;
  * the model's T_b output, normalized with the statistics it was trained on,
    is converted to the cleaned preparation's normalization (an affine map via
    physical mK), so metrics are in the same units for every model.

  python tools_eval_tb_clean.py --checkpoint RUN/best.pt --split test --out X.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import fno_multifield as fm
from dataset.global_history import HistoryEmulator
from dataset.los_windows import LOSWindowConfig

CLEAN = "experiments/los_windows/preparation_multifield_2000_tbclean.json"


class Renormalize(nn.Module):
    """Map output channels from the training normalization to another one."""
    def __init__(self, model, targets, trained, evaluated):
        super().__init__()
        self.model = model
        scale = [trained[n]["scale"]/evaluated[n]["scale"] for n in targets]
        shift = [(trained[n]["offset"]-evaluated[n]["offset"])/evaluated[n]["scale"] for n in targets]
        self.register_buffer("scale", torch.tensor(scale).view(1, -1, 1, 1, 1))
        self.register_buffer("shift", torch.tensor(shift).view(1, -1, 1, 1, 1))

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)*self.scale + self.shift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clean-preparation", default=CLEAN)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--spectral-bins", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    device = fm.choose_device(args.device)
    checkpoint, model, trained_ds, _ = fm.restore(args.checkpoint, device)
    trained_ds.close()
    meta = checkpoint["metadata"]
    clean_prep = json.loads(Path(args.clean_preparation).read_text())
    if clean_prep["split"] != meta["preparation"]["split"]:
        raise ValueError("cleaned preparation has a different split")
    dataset, rows, _ = fm.prepared_dataset(clean_prep, trained_ds.mapping)
    history = meta.get("training", {}).get("history_emulator")
    for entry in (history if isinstance(history, list) else [history] if history else []):
        dataset.install_history(HistoryEmulator(entry["path"], entry["sha256"]))
    if list(dataset.channel_names) != list(meta["input_channels"]):
        raise ValueError("input channels differ from the trained model's")
    for name in dataset.mapping.inputs:
        if dataset.normalization[name] != meta["preparation"]["normalization"][name]:
            raise ValueError(f"input normalization differs for {name}")
    wrapped = Renormalize(model, dataset.mapping.targets, meta["preparation"]["normalization"],
                          dataset.normalization).to(device).eval()
    sampling = meta.get("sampling", {"mode": "full"})
    window_config = None if sampling["mode"] == "full" else LOSWindowConfig(**sampling)
    try:
        values = fm.evaluate_rows(wrapped, dataset, rows[args.split], device, 1, 0,
                                  args.spectral_bins, window_config)
    finally:
        dataset.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
                               "split": args.split, "targets": "brightness_temp dv/dr-clipped",
                               "preparation": args.clean_preparation, "fields": values}, indent=1))
    print(json.dumps({k: {m: v[m] for m in ("mse", "rmse", "normalized_mse", "pearson_r")}
                      for k, v in values.items()}, indent=1))


if __name__ == "__main__":
    main()
