"""Post-hoc contrast-map theta sweep on a trained 2-D x_HI model.

g_theta(x) = 1/2 + 1/2 * tanh((x-1/2)/theta) / tanh(1/(2*theta))

Maps [0,1]->[0,1] fixing 0, 1/2 and 1; theta->0 is a hard step at 1/2 and
theta->inf recovers the identity, so theta is a transition-width knob
(slope at 1/2 is ~1/(2*theta)).

The question this answers: is the model's softness *under-confidence* we can
simply undo (RMSE improves at some theta < inf), or is it calibrated hedging
against genuine positional uncertainty (RMSE only degrades)?
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np, torch
from legacy.xhi2d.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from legacy.xhi2d.compare_xhi2d_models import build_model
from modeling import load_checkpoint

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
N_SLICES = 256

def contrast(x, theta):
    if theta is None:
        return x
    return 0.5 + 0.5*np.tanh((x-0.5)/theta)/np.tanh(0.5/theta)

def sharpness(field):
    blur = ((field > 0.1) & (field < 0.9)).mean()
    dx = np.roll(field, -1, -2) - field
    dy = np.roll(field, -1, -1) - field
    grad = np.sqrt(dx**2 + dy**2)
    width = (((field > 0.1) & (field < 0.9)).sum()
             / max(grad.sum(), 1e-9))
    return blur, np.percentile(grad.reshape(grad.shape[0], -1), 99.9, axis=1).mean(), width

meta = json.load(open(f"{RUN}/run_metadata.json"))
model = build_model(meta["model_config"])
load_checkpoint(model, f"{RUN}/final_model_state_dict.pt")
model = model.cuda().eval()
norm = meta.get("parameter_normalization")
cache = SliceCache(meta["dataset"]["cache_file"],
                   input_features=meta["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
test_cones = np.asarray(meta["split"]["test_cone_ids"], dtype=np.int64)
idx = np.flatnonzero(np.isin(cache.cone_id, test_cones))
rng = np.random.default_rng(0)
idx = rng.choice(idx, size=min(N_SLICES, len(idx)), replace=False)

preds, truths = [], []
with torch.inference_mode():
    for start in range(0, len(idx), 32):
        chunk = [cache[int(i)] for i in idx[start:start+32]]
        x = torch.stack([c["x"] for c in chunk]).cuda()
        preds.append(model(x).cpu().numpy()[:, 0])
        truths.append(torch.stack([c["y"] for c in chunk]).numpy()[:, 0])
pred = np.concatenate(preds); truth = np.concatenate(truths)
print(f"{len(pred)} held-out test slices from {RUN}\n")

tb, tp, tw = sharpness(truth)
print(f"{'theta':>8} {'RMSE':>9} {'vs base':>8} {'blur_frac':>10} "
      f"{'peak_grad':>10} {'width_px':>9}")
print(f"{'TRUTH':>8} {'--':>9} {'--':>8} {tb:10.4f} {tp:10.4f} {tw:9.3f}")
base = float(np.sqrt(np.mean((pred - truth)**2)))
for theta in (None, 4.0, 2.0, 1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05):
    p = contrast(pred, theta)
    rmse = float(np.sqrt(np.mean((p - truth)**2)))
    b, pk, w = sharpness(p)
    label = "identity" if theta is None else f"{theta:.2f}"
    print(f"{label:>8} {rmse:9.5f} {100*(rmse-base)/base:+7.1f}% "
          f"{b:10.4f} {pk:10.4f} {w:9.3f}")
