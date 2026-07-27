"""Verify gradients flow through the new edge terms into model parameters."""
import sys, json, os
sys.path.insert(0, "/pfs/10/work/hd_id260-fno_training/fno-21cm")
os.chdir("/pfs/10/work/hd_id260-fno_training/fno-21cm")
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np, torch
from dataset.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from losses import HighKPowerRatio, SlicedWassersteinEdges
from viz.compare_xhi2d_models import build_model

run = "checkpoints/xhi2d_whno_glob_lr3e4"
meta = json.load(open(f"{run}/run_metadata.json"))
model = build_model(meta["model_config"]).cuda().train()
norm = meta.get("parameter_normalization")
cache = SliceCache(meta["dataset"]["cache_file"],
                   input_features=meta["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
s = [cache[int(i)] for i in range(8)]
x = torch.stack([t["x"] for t in s]).cuda()
y = torch.stack([t["y"] for t in s]).cuda()

for name, fn in (("swd", SlicedWassersteinEdges(n_directions=48, seed=0)),
                 ("highk", HighKPowerRatio(k_min=0.2))):
    model.zero_grad()
    loss = fn(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total = sum(float(g.abs().sum()) for g in grads)
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = sum(1 for g in grads if float(g.abs().sum()) > 0)
    print(f"{name:6s} loss={float(loss):.6f}  grad_sum={total:.4e}  "
          f"all_finite={finite}  params_with_nonzero_grad={nonzero}/{len(grads)}")
