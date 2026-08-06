"""What is needed for a per-sample contrast head to work?

(1) Decompose the 5.3% headroom: per-slice theta alone, tau alone, or both.
(2) Test how predictable the fitted parameters are from progressively richer
    features -- pooled stats, + value histogram, + the conditioning inputs
    (redshift and cosmological parameters), linear vs small MLP.

All parameter fits use half the pixels of each slice and are scored on the
other half, so nothing here is inflated by selection bias.
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
from contrast import apply_contrast, prediction_statistics

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
N = 1200
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
P, T, X = [], [], []
with torch.inference_mode():
    for s in range(0, len(idx), 32):
        ch = [cache[int(i)] for i in idx[s:s+32]]
        xin = torch.stack([c["x"] for c in ch]).cuda()
        P.append(model(xin)[:, :1]); T.append(torch.stack([c["y"] for c in ch]).cuda()[:, :1])
        # conditioning channels are spatially constant -> one scalar each
        X.append(xin[:, 1:].mean(dim=(-2, -1)))
pred, truth, cond = torch.cat(P), torch.cat(T), torch.cat(X)
n = len(pred); del model; torch.cuda.empty_cache()
print(f"{n} held-out test slices; {cond.shape[1]} conditioning scalars\n")

thetas = torch.tensor(np.geomspace(0.03, 5.0, 24), device="cuda", dtype=pred.dtype)
taus = torch.tensor(np.linspace(0.10, 0.90, 17), device="cuda", dtype=pred.dtype)
g = torch.Generator(device="cuda").manual_seed(0)
mA = torch.rand(pred.shape, generator=g, device="cuda") < 0.5
mB = ~mA
def mse(a, b, m):
    return (((a - b) ** 2) * m).flatten(1).sum(1) / m.flatten(1).sum(1)

seA = torch.empty(len(thetas), len(taus), n, device="cuda"); seB = torch.empty_like(seA)
with torch.inference_mode():
    for i, t1 in enumerate(thetas):
        for j, t2 in enumerate(taus):
            gg = apply_contrast(pred, t1, t2)
            seA[i, j] = mse(gg, truth, mA); seB[i, j] = mse(gg, truth, mB)
ident = float(mse(pred, truth, mB).mean().sqrt())
ar = torch.arange(n, device="cuda")

def held(argflat, table):
    return float(table.reshape(-1, n)[argflat, ar].mean().sqrt())

print(f"{'headroom decomposition (fit half A, score half B)':48s} {'RMSE':>9} {'gain':>8}")
print(f"{'identity':48s} {ident:9.5f} {'--':>8}")
# tau fixed at 0.5 (nearest grid point), per-slice theta
j_half = int(np.argmin(np.abs(taus.cpu().numpy() - 0.5)))
a_th = seA[:, j_half, :].argmin(0) * len(taus) + j_half
v = held(a_th, seB); print(f"{'per-slice THETA only (tau=0.5)':48s} {v:9.5f} {100*(v-ident)/ident:+7.2f}%")
# theta fixed at global best, per-slice tau
i_g, j_g = divmod(int(seB.mean(2).argmin()), len(taus))
a_ta = i_g * len(taus) + seA[i_g, :, :].argmin(0)
v = held(a_ta, seB); print(f"{'per-slice TAU only (theta=global best)':48s} {v:9.5f} {100*(v-ident)/ident:+7.2f}%")
a_both = seA.reshape(-1, n).argmin(0)
v = held(a_both, seB); print(f"{'per-slice BOTH':48s} {v:9.5f} {100*(v-ident)/ident:+7.2f}%")

# --- predictability of the fitted parameters -----------------------------
best_th = np.log(thetas.cpu().numpy()[(a_both // len(taus)).cpu().numpy()])
best_ta = taus.cpu().numpy()[(a_both % len(taus)).cpu().numpy()]
stats = prediction_statistics(pred[:, 0]).cpu().numpy()
p = pred[:, 0].flatten(1)
hist = torch.stack([((p >= lo) & (p < lo + 1/16)).float().mean(1)
                    for lo in np.arange(0, 1, 1/16)], 1).cpu().numpy()
condn = cond.cpu().numpy()
sets = {
    "pooled stats (6)": stats,
    "+ value histogram (16)": np.concatenate([stats, hist], 1),
    "+ conditioning (z, params)": np.concatenate([stats, hist, condn], 1),
}
rng = np.random.default_rng(1); perm = rng.permutation(n); cut = int(0.7 * n)
tr, te = perm[:cut], perm[cut:]
print(f"\n{'features':30s} {'target':6s} {'linear R2':>10s} {'MLP R2':>8s}   (held-out 30%)")
def torch_mlp_r2(Ftr, ytr, Fte, yte, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Ftr.shape[1], 64), torch.nn.GELU(),
        torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).cuda()
    o = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.tensor(Ftr, dtype=torch.float32, device="cuda")
    Yt = torch.tensor(ytr, dtype=torch.float32, device="cuda")[:, None]
    Xe = torch.tensor(Fte, dtype=torch.float32, device="cuda")
    for _ in range(1500):
        o.zero_grad(); ((net(Xt) - Yt) ** 2).mean().backward(); o.step()
    with torch.inference_mode():
        pm = net(Xe).cpu().numpy()[:, 0]
    return 1 - ((yte - pm) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()

for name, F in sets.items():
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Ftr, Fte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    for tname, y in (("theta", best_th), ("tau", best_ta)):
        A = np.concatenate([Ftr, np.ones((len(tr), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pl = np.concatenate([Fte, np.ones((len(te), 1))], 1) @ coef
        r2l = 1 - ((y[te]-pl)**2).sum() / ((y[te]-y[te].mean())**2).sum()
        r2m = torch_mlp_r2(Ftr, y[tr], Fte, y[te])
        print(f"{name:30s} {tname:6s} {r2l:10.3f} {r2m:8.3f}")

print(f"\njoint fitted params: theta median={np.exp(np.median(best_th)):.3f} "
      f"10-90%={np.exp(np.percentile(best_th,10)):.3f}-{np.exp(np.percentile(best_th,90)):.3f}")
print(f"                     tau   median={np.median(best_ta):.3f} "
      f"10-90%={np.percentile(best_ta,10):.3f}-{np.percentile(best_ta,90):.3f}")
print(f"corr(log theta, tau) = {np.corrcoef(best_th, best_ta)[0,1]:+.3f}")
