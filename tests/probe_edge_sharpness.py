"""Objective edge-sharpness metrics for 2-D x_HI runs on identical slices.

blur_frac : fraction of pixels in the intermediate band 0.1 < x_HI < 0.9.
            A sharp two-phase field has few; blurring inflates it.
grad_ratio: mean |grad pred| / mean |grad truth|. <1 means under-sharp.
"""
import sys, json, os
sys.path.insert(0, "/pfs/10/work/hd_id260-fno_training/fno-21cm")
os.chdir("/pfs/10/work/hd_id260-fno_training/fno-21cm")
from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
import numpy as np, torch
from dataset.dataset import SliceCache
from dataset.dataset_3d import ParameterNormalization
from viz.compare_xhi2d_models import build_model, representative_indices
from viz.visualize_xhi2d_representative import DEFAULT_QUANTILES
from modeling import load_checkpoint

RUNS = [
    ("L2 baseline (final ep29)", "checkpoints/xhi2d_whno_glob_lr3e4",
     "final_model_state_dict.pt"),
    ("L2+SWD+highK (ep14)", "checkpoints/xhi2d_whno_glob_edge",
     "model_state_dict.pt"),
    ("L2+SWD      (ep14)", "checkpoints/xhi2d_whno_glob_swd",
     "model_state_dict.pt"),
    ("L2+highK    (ep14)", "checkpoints/xhi2d_whno_glob_highk",
     "model_state_dict.pt"),
]

def metrics(field):
    """Sharpness metrics that survive the total-variation invariance.

    Mean |grad| is useless here: blurring a monotonic step spreads the same
    total variation over more pixels, so it barely moves. What distinguishes
    a sharp edge is the *peak* gradient it reaches, and how few pixels the
    transition occupies per unit of edge.
    """
    blur = ((field > 0.1) & (field < 0.9)).mean(axis=(-2, -1))
    dx = np.roll(field, -1, -2) - field
    dy = np.roll(field, -1, -1) - field
    grad = np.sqrt(dx**2 + dy**2)
    n = grad.shape[0]
    flat = grad.reshape(n, -1)
    # p99.9 of |grad|: the steepness actually attained at boundaries.
    peak = np.percentile(flat, 99.9, axis=1)
    # Transition width proxy: intermediate area divided by total variation,
    # i.e. how many pixels of ramp per unit of edge crossed.
    tv = flat.sum(axis=1)
    width = (((field > 0.1) & (field < 0.9)).sum(axis=(-2, -1))
             / np.maximum(tv, 1e-9))
    return blur, peak, width

meta0 = json.load(open(f"{RUNS[0][1]}/run_metadata.json"))
norm = meta0.get("parameter_normalization")
cache = SliceCache(meta0["dataset"]["cache_file"],
                   input_features=meta0["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
test_cones = np.asarray(meta0["split"]["test_cone_ids"], dtype=np.int64)
test_idx = np.flatnonzero(np.isin(cache.cone_id, test_cones))
idx = representative_indices(cache, test_idx, DEFAULT_QUANTILES)
samples = [cache[i] for i in idx]
x = torch.stack([s["x"] for s in samples]).cuda()
truth = torch.stack([s["y"] for s in samples]).numpy()[:, 0]
tb, tp, tw = metrics(truth)
print(f"{'model':26s} {'blur_frac':>10s} {'peak_grad':>10s} {'width_px':>9s}")
print(f"{'TRUTH (target)':26s} {tb.mean():10.4f} {tp.mean():10.4f} "
      f"{tw.mean():9.3f}")
for label, d, ckpt in RUNS:
    meta = json.load(open(f"{d}/run_metadata.json"))
    model = build_model(meta["model_config"])
    load_checkpoint(model, f"{d}/{ckpt}")
    model = model.cuda().eval()
    with torch.inference_mode():
        pred = model(x).cpu().numpy()[:, 0]
    pb, pp, pw = metrics(pred)
    print(f"{label:26s} {pb.mean():10.4f} {pp.mean():10.4f} "
          f"{pw.mean():9.3f}")
    del model; torch.cuda.empty_cache()
