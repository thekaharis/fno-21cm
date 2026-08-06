"""Measure raw magnitudes of each loss term on real predictions."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np
from neuralop import H1Loss, LpLoss
from legacy.xhi2d.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from losses import (AbsoluteLoss, BinaryCrossEntropyTerm, HighKPowerRatio,
                    SlicedWassersteinEdges)
from legacy.xhi2d.compare_xhi2d_models import build_model
from modeling import load_checkpoint

run = "checkpoints/xhi2d_whno_glob_lr3e4"
meta = json.load(open(f"{run}/run_metadata.json"))
model = build_model(meta["model_config"])
load_checkpoint(model, f"{run}/final_model_state_dict.pt")
model = model.cuda().eval()

norm = meta.get("parameter_normalization")
cache = SliceCache(meta["dataset"]["cache_file"],
                   input_features=meta["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
test_cones = np.asarray(meta["split"]["test_cone_ids"], dtype=np.int64)
idx = np.flatnonzero(np.isin(cache.cone_id, test_cones))[:64]
samples = [cache[int(i)] for i in idx]
x = torch.stack([s["x"] for s in samples]).cuda()
y = torch.stack([s["y"] for s in samples]).cuda()
with torch.inference_mode():
    pred = model(x)

terms = {
    "l2": AbsoluteLoss(LpLoss(d=2, p=2)),
    "h1": AbsoluteLoss(H1Loss(d=2)),
    "bce": BinaryCrossEntropyTerm(),
    "swd": SlicedWassersteinEdges(n_directions=48, seed=0),
    "highk": HighKPowerRatio(k_min=0.2),
}
print(f"{'term':8s} {'raw value':>12s}   weight for 10% / 25% of an L2=1.0 budget")
vals = {}
for name, fn in terms.items():
    v = float(fn(pred, y))
    vals[name] = v
    print(f"{name:8s} {v:12.5f}", end="")
    if name != "l2":
        l2 = vals["l2"]
        print(f"   {0.10*l2/max(v,1e-12):>9.4f} / {0.25*l2/max(v,1e-12):.4f}")
    else:
        print("   (reference)")
