"""Render truth / raw prediction / contrast-mapped predictions side by side.

g_theta(x) = 1/2 + 1/2 * tanh((x-1/2)/theta) / tanh(1/(2*theta))
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from legacy.xhi2d.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from legacy.xhi2d.compare_xhi2d_models import build_model, representative_indices
from modeling import load_checkpoint

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
THETAS = [None, 0.6, 0.4, 0.3, 0.2]
QUANTILES = [0.2, 0.45, 0.7, 0.9]

def contrast(x, theta):
    if theta is None:
        return x
    return 0.5 + 0.5*np.tanh((x-0.5)/theta)/np.tanh(0.5/theta)

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
tidx = np.flatnonzero(np.isin(cache.cone_id, test_cones))
idx = representative_indices(cache, tidx, QUANTILES)
samples = [cache[i] for i in idx]
x = torch.stack([s["x"] for s in samples]).cuda()
truth = torch.stack([s["y"] for s in samples]).numpy()[:, 0]
with torch.inference_mode():
    pred = model(x).cpu().numpy()[:, 0]

nrow, ncol = len(idx), len(THETAS) + 1
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6*ncol, 2.75*nrow),
                         constrained_layout=True, squeeze=False)
for r in range(nrow):
    panels = [("Truth", truth[r], None)]
    for th in THETAS:
        panels.append((("raw prediction" if th is None else f"$\\theta$={th}"),
                       contrast(pred[r], th), th))
    for c, (title, img, th) in enumerate(panels):
        ax = axes[r][c]
        ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=11)
        if c > 0:
            rmse = float(np.sqrt(np.mean((img - truth[r])**2)))
            blur = float(((img > 0.1) & (img < 0.9)).mean())
            ax.text(0.02, 0.98, f"RMSE={rmse:.3f}\nblur={blur:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
        else:
            blur = float(((img > 0.1) & (img < 0.9)).mean())
            ax.text(0.02, 0.98, f"blur={blur:.3f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7.5, color="white",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    axes[r][0].set_ylabel(
        f"z={cache.z[idx[r]]:.2f}\n$\\bar{{x}}_{{HI}}$={cache.xHI_mean[idx[r]]:.2f}",
        fontsize=9)
fig.suptitle("Contrast map on trained whno_glob predictions "
             "(theta -> 0 = hard step at 1/2)", fontsize=13)
out = "figures/contrast_map_examples"
os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/contrast_map_examples.png", dpi=155)
print(f"wrote {out}/contrast_map_examples.png")
