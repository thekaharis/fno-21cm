"""Is the per-slice contrast-map headroom real, or selection bias?

The oracle picks the best of ~400 (theta,tau) pairs per slice using that
slice's own target, so it is guaranteed to find something. Here the pair is
fitted on a random HALF of each slice's pixels and scored on the other half.
Real headroom survives; noise-fitting does not.

Also asks whether the fitted parameters are predictable from pooled
prediction statistics at all -- if not, no head can reach them.
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
from contrast import apply_contrast, prediction_statistics

RUN = "checkpoints/2d_xhi/fno_whno/xhi2d_whno_glob_lr3e4"
N = 600
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
        P.append(model(torch.stack([c["x"] for c in ch]).cuda())[:, :1])
        T.append(torch.stack([c["y"] for c in ch]).cuda()[:, :1])
pred = torch.cat(P); truth = torch.cat(T)
n = len(pred)
print(f"{n} held-out test slices\n")

thetas = torch.tensor(np.geomspace(0.03, 5.0, 24), device="cuda", dtype=pred.dtype)
taus = torch.tensor(np.linspace(0.10, 0.90, 17), device="cuda", dtype=pred.dtype)

# random half-mask of pixels per slice: fit on A, score on B
g = torch.Generator(device="cuda").manual_seed(0)
maskA = (torch.rand(pred.shape, generator=g, device="cuda") < 0.5)
maskB = ~maskA

def masked_mse(a, b, m):
    d = ((a - b) ** 2) * m
    return d.flatten(1).sum(1) / m.flatten(1).sum(1)

seA = torch.empty(len(thetas), len(taus), n, device="cuda")
seB = torch.empty_like(seA)
with torch.inference_mode():
    for i, t1 in enumerate(thetas):
        for j, t2 in enumerate(taus):
            gg = apply_contrast(pred, t1, t2)
            seA[i, j] = masked_mse(gg, truth, maskA)
            seB[i, j] = masked_mse(gg, truth, maskB)

ident_B = float(masked_mse(pred, truth, maskB).mean().sqrt())
# in-sample oracle scored on the half it was fitted on (biased)
oracle_A_on_A = float(seA.reshape(-1, n).min(0).values.mean().sqrt())
# honest: argmin on A, score on B
arg = seA.reshape(-1, n).argmin(0)
held = seB.reshape(-1, n)[arg, torch.arange(n, device="cuda")]
oracle_A_on_B = float(held.mean().sqrt())
glob = float(seB.mean(dim=2).min().sqrt())

print(f"{'variant (scored on held-out pixel half)':44s} {'RMSE':>9} {'vs identity':>12}")
print(f"{'identity':44s} {ident_B:9.5f} {'--':>12}")
print(f"{'best global (theta,tau)':44s} {glob:9.5f} {100*(glob-ident_B)/ident_B:+11.2f}%")
print(f"{'per-slice fit on half A -> scored on half B':44s} "
      f"{oracle_A_on_B:9.5f} {100*(oracle_A_on_B-ident_B)/ident_B:+11.2f}%")
print(f"\n[for reference, the biased number: per-slice fit and scored on the")
print(f" SAME pixels = {oracle_A_on_A:.5f}, {100*(oracle_A_on_A-ident_B)/ident_B:+.2f}% ]")

# are the fitted parameters predictable from pooled prediction stats?
best_th = thetas[(arg // len(taus))].cpu().numpy()
best_ta = taus[(arg % len(taus))].cpu().numpy()
stats = prediction_statistics(pred[:, 0]).cpu().numpy()
X = np.concatenate([stats, np.ones((n, 1))], axis=1)
print()
for name, y in (("theta", np.log(best_th)), ("tau", best_ta)):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"linear predictability of fitted {name:5s} from pooled stats: R2={r2:+.3f}")
