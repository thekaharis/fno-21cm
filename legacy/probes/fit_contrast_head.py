"""Fit a per-sample ContrastHead on top of a FROZEN trained model.

The base model never changes, so any gain is attributable to the head, and
the fit costs minutes: raw predictions are computed once and cached, then
only the ~1k-parameter head is trained on them.

Reports identity / best-global / learned-head / oracle on the same held-out
slices, so the learned head can be placed against its own ceiling.
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
from contrast import ContrastHead, apply_contrast

RUN = "checkpoints/xhi2d_whno_glob_lr3e4"
N_TRAIN, N_EVAL, EPOCHS = 3000, 600, 400
torch.manual_seed(0)

meta = json.load(open(f"{RUN}/run_metadata.json"))
model = build_model(meta["model_config"])
load_checkpoint(model, f"{RUN}/final_model_state_dict.pt")
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
norm = meta.get("parameter_normalization")
cache = SliceCache(meta["dataset"]["cache_file"],
                   input_features=meta["input_features"]["name"],
                   parameter_normalization=(ParameterNormalization.from_dict(norm)
                                            if norm else None))
split = meta["split"]
rng = np.random.default_rng(0)

def collect(cone_key, n):
    cones = np.asarray(split[cone_key], dtype=np.int64)
    idx = np.flatnonzero(np.isin(cache.cone_id, cones))
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    P, T = [], []
    with torch.inference_mode():
        for s in range(0, len(idx), 32):
            ch = [cache[int(i)] for i in idx[s:s+32]]
            P.append(model(torch.stack([c["x"] for c in ch]).cuda())[:, :1].cpu())
            T.append(torch.stack([c["y"] for c in ch])[:, :1])
    return torch.cat(P), torch.cat(T)

tr_p, tr_t = collect("train_cone_ids", N_TRAIN)
te_p, te_t = collect("test_cone_ids", N_EVAL)
print(f"cached: train {len(tr_p)} slices, test {len(te_p)} slices")
del model; torch.cuda.empty_cache()

head = ContrastHead().cuda()
opt = torch.optim.Adam(head.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tr_p_g, tr_t_g = tr_p.cuda(), tr_t.cuda()
bs = 256
for ep in range(EPOCHS):
    perm = torch.randperm(len(tr_p_g), device="cuda")
    for s in range(0, len(perm), bs):
        b = perm[s:s+bs]
        opt.zero_grad()
        loss = ((head.apply(tr_p_g[b]) - tr_t_g[b]) ** 2).mean()
        loss.backward(); opt.step()
    sched.step()
    if ep % 100 == 0 or ep == EPOCHS - 1:
        print(f"  epoch {ep:3d}  train MSE {float(loss):.6f}")

head.eval()
te_p_g, te_t_g = te_p.cuda(), te_t.cuda()
with torch.inference_mode():
    ident = float(((te_p_g - te_t_g) ** 2).mean().sqrt())
    learned = float(((head.apply(te_p_g) - te_t_g) ** 2).mean().sqrt())
    th, ta = head(te_p_g)
    # best global on the SAME slices, and the per-slice oracle
    thetas = torch.tensor(np.geomspace(0.03, 5.0, 24), device="cuda", dtype=te_p_g.dtype)
    taus = torch.tensor(np.linspace(0.10, 0.90, 17), device="cuda", dtype=te_p_g.dtype)
    mse = torch.empty(len(thetas), len(taus), len(te_p_g), device="cuda")
    for i, t1 in enumerate(thetas):
        for j, t2 in enumerate(taus):
            mse[i, j] = ((apply_contrast(te_p_g, t1, t2) - te_t_g) ** 2).mean(dim=(1, 2, 3))
    glob = float(mse.mean(dim=2).min().sqrt())
    oracle = float(mse.reshape(-1, len(te_p_g)).min(dim=0).values.mean().sqrt())

print(f"\n{'variant':26s} {'test RMSE':>10} {'vs identity':>12}")
for name, v in (("identity (raw)", ident), ("best global (th,tau)", glob),
                ("LEARNED per-sample head", learned), ("oracle per-slice", oracle)):
    d = "--" if name.startswith("identity") else f"{100*(v-ident)/ident:+.2f}%"
    print(f"{name:26s} {v:10.5f} {d:>12}")
captured = (ident - learned) / max(ident - oracle, 1e-12)
print(f"\nhead captures {100*captured:.0f}% of the oracle's headroom")
print(f"predicted theta: median={float(th.median()):.3f} "
      f"range={float(th.min()):.3f}-{float(th.max()):.3f}")
print(f"predicted tau  : median={float(ta.median()):.3f} "
      f"range={float(ta.min()):.3f}-{float(ta.max()):.3f}")
torch.save(head.state_dict(), f"{RUN}/contrast_head.pt")
print(f"saved {RUN}/contrast_head.pt")
