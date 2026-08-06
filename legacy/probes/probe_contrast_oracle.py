"""Ceiling test for a per-sample contrast map.

Compares, on held-out slices:
  identity            -- the raw model output
  best GLOBAL (th,ta) -- one pair shared by every slice
  ORACLE per-slice    -- the best pair for each slice, fitted against truth

The oracle is unattainable (it peeks at the target), so it upper-bounds any
learned per-sample head. If the oracle barely beats identity, a head cannot
help and is not worth building.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np, torch
from dataset.slices import SliceCache
from dataset.dataset_3d import ParameterNormalization
from viz.compare_xhi2d_models import build_model
from modeling import load_checkpoint
from contrast import apply_contrast

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
N = 256
meta = json.load(open(f"{RUN}/run_metadata.json"))
model = build_model(meta["model_config"])
load_checkpoint(model, f"{RUN}/final_model_state_dict.pt")
model = model.cuda().eval()
norm = meta.get("parameter_normalization")
cache = SliceCache(meta["dataset"]["cache_file"],
                   input_features=meta["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
tc = np.asarray(meta["split"]["test_cone_ids"], dtype=np.int64)
idx = np.flatnonzero(np.isin(cache.cone_id, tc))
idx = np.random.default_rng(0).choice(idx, size=min(N, len(idx)), replace=False)
P, T = [], []
with torch.inference_mode():
    for s in range(0, len(idx), 32):
        ch = [cache[int(i)] for i in idx[s:s+32]]
        P.append(model(torch.stack([c["x"] for c in ch]).cuda()))
        T.append(torch.stack([c["y"] for c in ch]).cuda())
pred = torch.cat(P)[:, 0]; truth = torch.cat(T)[:, 0]
print(f"{len(pred)} held-out test slices\n")

thetas = torch.tensor(np.geomspace(0.03, 5.0, 24), device="cuda", dtype=pred.dtype)
taus = torch.tensor(np.linspace(0.15, 0.85, 15), device="cuda", dtype=pred.dtype)

# (n_theta, n_tau, n_slices) per-slice MSE over the whole grid
mse = torch.empty(len(thetas), len(taus), len(pred), device="cuda")
with torch.inference_mode():
    for i, th in enumerate(thetas):
        for j, ta in enumerate(taus):
            g = apply_contrast(pred, th, ta)
            mse[i, j] = ((g - truth) ** 2).mean(dim=(-2, -1))

ident = float(((pred - truth) ** 2).mean().sqrt())
# best single (theta, tau) for all slices
gmean = mse.mean(dim=2)
gi, gj = divmod(int(gmean.argmin()), gmean.shape[1])
glob = float(gmean[gi, gj].sqrt())
# per-slice oracle
per = mse.reshape(-1, len(pred)).min(dim=0)
oracle = float(per.values.mean().sqrt())
flat_arg = per.indices.cpu().numpy()
best_th = thetas.cpu().numpy()[flat_arg // len(taus)]
best_ta = taus.cpu().numpy()[flat_arg % len(taus)]

print(f"{'variant':28s} {'RMSE':>9} {'vs identity':>12}")
print(f"{'identity (raw)':28s} {ident:9.5f} {'--':>12}")
print(f"{'best GLOBAL theta,tau':28s} {glob:9.5f} {100*(glob-ident)/ident:+11.2f}%")
print(f"{'ORACLE per-slice':28s} {oracle:9.5f} {100*(oracle-ident)/ident:+11.2f}%")
print(f"\nbest global: theta={float(thetas[gi]):.3f} tau={float(taus[gj]):.3f}")
print(f"oracle theta: median={np.median(best_th):.3f} "
      f"10-90%={np.percentile(best_th,10):.3f}-{np.percentile(best_th,90):.3f}")
print(f"oracle tau  : median={np.median(best_ta):.3f} "
      f"10-90%={np.percentile(best_ta,10):.3f}-{np.percentile(best_ta,90):.3f}")
frac = float((best_th < 4.0).mean())
print(f"slices where oracle wants real sharpening (theta<4): {100*frac:.0f}%")
