"""Global reionization history emulator: parameters -> mean x_HI(z).

The field models get reionization *timing* almost entirely from the 11
parameters (density carries little of it), and they regress to the typical
history at the edges of the prior: cone 27 (z_mid 15.8, 99.8th percentile of
the training cones) is reionized dz ~ 0.6-1.3 too late by every windowed run.
Timing is a 1-D function per simulation, and every raw file stores it densely
(lightcone/global_quantities/neutral_fraction at 101 shared node redshifts),
for all ~6600 simulations -- not only the 2000-cone field subset.

Steps (each cached in --out):
  collect   read (params, history) of every raw simulation
  train     ensemble of monotone MLPs; the history is sigmoid(b + cumsum
            softplus(d)) over increasing z, so it can only rise with z
  evaluate  history RMSE and midpoint error on the held-out cones, and a
            comparison with the histories implied by the saved field
            predictions (mean over each transverse slice) on the same cones
  relevel   post-hoc test of value: shift every predicted slice in logit space
            so its mean equals the emulator's x_HI(z); MSE before/after, with
            the true slice mean as an oracle bound

Held out from training: the validation and test cones of BOTH the x_HI study
(preparation_xhi_2000.json, row == sample id) and the multi-field study
(preparation_multifield_2000.json, via its sample-id map), so the emulator
can feed either field model without leakage.

  python tools_global_history_emulator.py --workers 16

--quantity brightness_temp builds the global T_b history emulator instead
(unconstrained MLP ensemble on standardized mK; files histories_brightness_temp.npz
and emulator_brightness_temp.pt; channel T_b_global_emulated in units of 100 mK).
Its evaluate step reports RMSE in mK and absorption-trough depth/position errors;
relevel applies to x_HI only.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.global_history import HistoryEmulator, MonotoneHistory, PlainHistory

RAW = Path("/pfs/10/work/hd_id260-fno_training/data/data")
OUT = Path("experiments/global_history")
PRED_DIRS = {"contiguous_ep18": Path("experiments/los_windows/predictions/contiguous_ep18"),
             "coarse_context_ep16": Path("experiments/los_windows/predictions/coarse_context_ep16")}
Z_RANGE = (5.0, 25.0)           # lightcone coverage
CHANNELS = {"neutral_fraction": ("x_HI_global_emulated", 1.0),
            "brightness_temp": ("T_b_global_emulated", 100.0)}     # name, channel scale


def suffix(quantity):
    return "" if quantity == "neutral_fraction" else f"_{quantity}"


def read_one(path, quantity="neutral_fraction"):
    try:
        with h5py.File(path, "r") as h:
            sid = int(h.attrs["sample_id"])
            z = h["lightcone/node_redshifts"][:]
            x = h[f"lightcone/global_quantities/{quantity}"][:]
            theta = h["params/values"][:]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["params/names"][:]]
        if not (np.isfinite(x).all() and np.isfinite(theta).all()):
            return None
        return sid, z, x, theta, names
    except (OSError, KeyError):
        return None


def collect(out, workers, quantity="neutral_fraction"):
    paths = sorted(RAW.glob("21cmfast_11d_sample*.h5"))
    with Pool(workers) as pool:
        rows = [r for r in pool.map(partial(read_one, quantity=quantity), paths, chunksize=16)
                if r is not None]
    z = rows[0][1]
    if not all(np.array_equal(r[1], z) for r in rows):
        raise ValueError("node redshifts differ between simulations")
    order = np.argsort(z)                                   # increasing z
    np.savez(out/f"histories{suffix(quantity)}.npz", sample_id=np.array([r[0] for r in rows]),
             z=z[order], x=np.stack([r[2][order] for r in rows]),
             theta=np.stack([r[3] for r in rows]), names=np.array(rows[0][4]))
    print(f"collected {len(rows)}/{len(paths)} simulations")


def holdout_ids():
    xhi = json.load(open("experiments/los_windows/preparation_xhi_2000.json"))["split"]
    mf_map = json.load(open("experiments/los_windows/preparation_multifield_2000_sample_ids.json"))["rows"]
    mf = {r["sample_id"] for r in mf_map if r["split"] in ("val", "test")}
    return {"xhi_val": set(xhi["val"]), "xhi_test": set(xhi["test"]),
            "all": set(xhi["val"]) | set(xhi["test"]) | mf}


def train(out, members, epochs, seed0=0, quantity="neutral_fraction"):
    d = np.load(out/f"histories{suffix(quantity)}.npz")
    monotone = quantity == "neutral_fraction"
    held = holdout_ids()["all"]
    sid = d["sample_id"]
    pool_idx = np.flatnonzero(~np.isin(sid, list(held)))
    rng = np.random.default_rng(1234)
    rng.shuffle(pool_idx)
    n_val = len(pool_idx)//10
    val_idx, train_idx = pool_idx[:n_val], pool_idx[n_val:]
    mu, sd = d["theta"][train_idx].mean(0), d["theta"][train_idx].std(0)
    th = torch.tensor((d["theta"]-mu)/sd, dtype=torch.float32)
    offset, scale = (0.0, 1.0) if monotone else (float(d["x"][train_idx].mean()), float(d["x"][train_idx].std()))
    x = torch.tensor((d["x"]-offset)/scale, dtype=torch.float32)
    states = []
    for m in range(members):
        torch.manual_seed(seed0+m)
        model = (MonotoneHistory if monotone else PlainHistory)(th.shape[1], x.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = np.inf, None
        tr = torch.as_tensor(train_idx)
        for ep in range(epochs):
            model.train()
            perm = tr[torch.randperm(len(tr))]
            for b in perm.split(256):
                loss = ((model(th[b])-x[b])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if ep % 10 == 9 or ep == epochs-1:
                model.eval()
                with torch.no_grad():
                    v = float(((model(th[val_idx])-x[val_idx])**2).mean())
                if v < best:
                    best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        print(f"member {m}: best val MSE {best:.3e}", flush=True)
        states.append(best_state)
    channel, channel_scale = CHANNELS[quantity]
    torch.save({"states": states, "theta_mean": mu, "theta_std": sd, "z": d["z"],
                "names": list(d["names"]), "train_ids": sid[train_idx].tolist(),
                "val_ids": sid[val_idx].tolist(), "hidden": 256, "quantity": quantity,
                "kind": "monotone" if monotone else "plain", "channel": channel,
                "output_offset": offset, "output_scale": scale, "channel_scale": channel_scale},
               out/f"emulator{suffix(quantity)}.pt")
    print(f"train {len(train_idx)}  val {len(val_idx)}  held out {len(held)}")


def load_emulator(out, quantity="neutral_fraction"):
    emu = HistoryEmulator(out/f"emulator{suffix(quantity)}.pt")

    def predict(theta):                                            # (members, n, nodes)
        return emu.ensemble(theta, emu.names)
    return predict, emu.z


def crossing(z, x, level=0.5):
    """Lowest z at which the (rising) history reaches ``level``."""
    i = np.flatnonzero((x[:-1] < level) & (x[1:] >= level))
    if not len(i):
        return np.nan
    i = i[0]
    return z[i] + (level-x[i])*(z[i+1]-z[i])/(x[i+1]-x[i])


def evaluate(out):
    d = np.load(out/"histories.npz")
    predict, z = load_emulator(out)
    held = holdout_ids()
    sid = list(d["sample_id"])
    band = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
    report = {}
    for split in ("xhi_val", "xhi_test"):
        idx = [sid.index(s) for s in sorted(held[split]) if s in sid]
        ens = predict(d["theta"][idx]); mean = ens.mean(0)
        truth = d["x"][idx]
        dz = np.array([crossing(z, mean[i]) - crossing(z, truth[i]) for i in range(len(idx))])
        report[split] = {"cones": len(idx),
                         "history_rmse_5_25": float(np.sqrt(((mean-truth)[:, band]**2).mean())),
                         "midpoint_abs_dz_median": float(np.nanmedian(np.abs(dz))),
                         "midpoint_abs_dz_p90": float(np.nanpercentile(np.abs(dz), 90)),
                         "midpoint_cones": int(np.isfinite(dz).sum())}
    # versus the field networks on the saved prediction cones
    comparison = {}
    for name, pdir in PRED_DIRS.items():
        for f in sorted(pdir.glob("cone_*.h5")):
            cone = int(f.stem.split("_")[1])
            with h5py.File(f) as h:
                zc = h["target_z"][:]
                t = h["target/neutral_fraction"][::2, ::2, :].mean((0, 1))
                p = h["prediction/neutral_fraction"][::2, ::2, :].mean((0, 1))
            ens = predict(d["theta"][sid.index(cone)])[:, 0]
            e = np.interp(zc, z, ens.mean(0))
            spread = np.interp(zc, z, ens.std(0))
            active = (t > 0.02) & (t < 0.98)
            row = comparison.setdefault(cone, {})
            row["z_mid_true"] = float(crossing(zc, t))
            row["z_mid_emulator"] = float(crossing(zc, e))
            row[f"z_mid_{name}"] = float(crossing(zc, p))
            # Floor for any history model: the true BOX history versus the
            # mean of each thin lightcone slice (slice sampling + lightcone
            # interpolation between nodes).
            g = np.interp(zc, z, d["x"][sid.index(cone)])
            row["z_mid_box_history"] = float(crossing(zc, g))
            if active.any():
                row["history_rmse_box_vs_slice"] = float(np.sqrt(((g-t)[active]**2).mean()))
                row["history_rmse_emulator"] = float(np.sqrt(((e-t)[active]**2).mean()))
                row[f"history_rmse_{name}"] = float(np.sqrt(((p-t)[active]**2).mean()))
                row["emulator_spread"] = float(spread[active].mean())
    report["cones"] = comparison
    (out/"evaluation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "cones"}, indent=1))
    keys = ["z_mid_true", "z_mid_box_history", "z_mid_emulator", "z_mid_contiguous_ep18", "z_mid_coarse_context_ep16",
            "history_rmse_box_vs_slice", "history_rmse_emulator", "history_rmse_contiguous_ep18", "history_rmse_coarse_context_ep16"]
    print("cone  " + "  ".join(k.replace("history_rmse", "rmse").replace("z_mid", "zmid")[:22] for k in keys))
    for cone, r in sorted(comparison.items()):
        print(f"{cone:5d} " + "  ".join(f"{r.get(k, np.nan):22.3f}" for k in keys))
    plot(d, predict, z, comparison, out)


def evaluate_tb(out):
    """Global T_b emulator on held-out cones: RMSE (mK) and absorption trough."""
    q = "brightness_temp"
    d = np.load(out/f"histories{suffix(q)}.npz")
    predict, z = load_emulator(out, q)
    held = holdout_ids()
    mf_test = {r["sample_id"] for r in json.load(open(
        "experiments/los_windows/preparation_multifield_2000_sample_ids.json"))["rows"] if r["split"] == "test"}
    sid = list(d["sample_id"])
    band = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
    report = {}
    for split, ids in (("xhi_test", held["xhi_test"]), ("mf_test", mf_test)):
        idx = [sid.index(s) for s in sorted(ids) if s in sid]
        mean = predict(d["theta"][idx]).mean(0)[:, band]
        truth = d["x"][idx][:, band]
        zb = z[band]
        depth_t, depth_p = truth.min(1), mean.min(1)
        report[split] = {"cones": len(idx),
                         "rmse_mK": float(np.sqrt(((mean-truth)**2).mean())),
                         "truth_rms_mK": float(np.sqrt((truth**2).mean())),
                         "trough_depth_abs_err_median_mK": float(np.median(np.abs(depth_p-depth_t))),
                         "trough_depth_rel_err_median": float(np.median(np.abs(depth_p-depth_t)/np.abs(depth_t))),
                         "trough_z_abs_err_median": float(np.median(np.abs(zb[mean.argmin(1)]-zb[truth.argmin(1)])))}
    cones = {}
    for c in (27, 969, 1872, 403, 762, 1275, 1031):
        i = sid.index(c)
        t, p = d["x"][i][band], predict(d["theta"][i:i+1]).mean(0)[0][band]
        cones[c] = {"trough_truth_mK": float(t.min()), "trough_emulator_mK": float(p.min()),
                    "trough_z_truth": float(z[band][t.argmin()]), "trough_z_emulator": float(z[band][p.argmin()]),
                    "rmse_mK": float(np.sqrt(((p-t)**2).mean()))}
    report["cones"] = cones
    (out/"evaluation_brightness_temp.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, c in zip(axes.flat, list(cones)+[None]):
        if c is None:
            ax.axis("off"); continue
        i = sid.index(c); ens = predict(d["theta"][i:i+1])[:, 0]
        ax.plot(z, d["x"][i], "k", lw=2, label="truth (box history)")
        ax.fill_between(z, ens.min(0), ens.max(0), color="C0", alpha=0.3)
        ax.plot(z, ens.mean(0), "C0", lw=1.3, label="emulator")
        ax.set_xlim(*Z_RANGE); ax.set_title(f"sample {c}", fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlabel("z"); ax.set_ylabel("mean T_b [mK]")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"tb_histories_emulator.png", dpi=120)


def relevel_slices(pred, target_mean, iters=40):
    """Shift each LOS slice's logit so its mean equals target_mean (bisection)."""
    p = np.clip(pred, 1e-5, 1-1e-5).astype(np.float64)
    logit = np.log(p/(1-p))
    target = np.clip(target_mean, 1e-5, 1-1e-5)
    lo, hi = np.full(p.shape[-1], -30.0), np.full(p.shape[-1], 30.0)
    for _ in range(iters):
        mid = (lo+hi)/2
        m = (1/(1+np.exp(-(logit+mid)))).mean((0, 1))
        lo, hi = np.where(m < target, mid, lo), np.where(m < target, hi, mid)
    return 1/(1+np.exp(-(logit+(lo+hi)/2)))


def relevel(out):
    d = np.load(out/"histories.npz")
    predict, z = load_emulator(out)
    sid = list(d["sample_id"])
    rows = []
    for name, pdir in PRED_DIRS.items():
        for f in sorted(pdir.glob("cone_*.h5")):
            cone = int(f.stem.split("_")[1])
            with h5py.File(f) as h:
                zc = h["target_z"][:]
                t = h["target/neutral_fraction"][::2, ::2, :].astype(np.float64)
                p = h["prediction/neutral_fraction"][::2, ::2, :].astype(np.float64)
            e = np.interp(zc, z, predict(d["theta"][sid.index(cone)])[:, 0].mean(0))
            r = {"run": name, "cone": cone, "mse": float(((p-t)**2).mean()),
                 "mse_relevel_emulator": float(((relevel_slices(p, e)-t)**2).mean()),
                 "mse_relevel_oracle": float(((relevel_slices(p, t.mean((0, 1)))-t)**2).mean())}
            rows.append(r)
            print(f"{name:20s} cone {cone:5d}  mse {r['mse']:.5f}  -> emulator {r['mse_relevel_emulator']:.5f}"
                  f"  (oracle slice mean {r['mse_relevel_oracle']:.5f})", flush=True)
    for name in PRED_DIRS:
        sel = [r for r in rows if r["run"] == name]
        print(f"{name}: mean mse {np.mean([r['mse'] for r in sel]):.5f} -> emulator "
              f"{np.mean([r['mse_relevel_emulator'] for r in sel]):.5f}, oracle "
              f"{np.mean([r['mse_relevel_oracle'] for r in sel]):.5f}")
    (out/"relevel.json").write_text(json.dumps(rows, indent=1))


def plot(d, predict, z, comparison, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cones = sorted(c for c, r in comparison.items() if np.isfinite(r["z_mid_true"]))
    ncol = 4
    nrow = int(np.ceil(len(cones)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), squeeze=False)
    sid = list(d["sample_id"])
    for ax, cone in zip(axes.flat, cones):
        with h5py.File(PRED_DIRS["contiguous_ep18"]/f"cone_{cone}.h5") as h:
            zc = h["target_z"][:]
            t = h["target/neutral_fraction"][::2, ::2, :].mean((0, 1))
            p = h["prediction/neutral_fraction"][::2, ::2, :].mean((0, 1))
        ens = predict(d["theta"][sid.index(cone)])[:, 0]
        ax.plot(zc, t, "k", lw=2, label="truth (slice mean)")
        ax.plot(zc, p, "C3", lw=1.3, label="field model contiguous_ep18")
        ax.fill_between(z, ens.min(0), ens.max(0), color="C0", alpha=0.3)
        ax.plot(z, ens.mean(0), "C0", lw=1.3, label="history emulator")
        zm = comparison[cone]["z_mid_true"]
        ax.set_xlim(max(5, zm-4), min(25, zm+4))
        ax.set_title(f"cone {cone}", fontsize=10)
        ax.set_xlabel("z"); ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(cones):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"histories_vs_field_model.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["collect", "train", "evaluate", "relevel"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--quantity", choices=tuple(CHANNELS), default="neutral_fraction")
    args = ap.parse_args()
    if args.quantity != "neutral_fraction" and "relevel" in args.steps:
        args.steps = [s for s in args.steps if s != "relevel"]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.workers))
    for step in args.steps:
        q = args.quantity
        {"collect": lambda: collect(args.out, args.workers, q),
         "train": lambda: train(args.out, args.members, args.epochs, quantity=q),
         "evaluate": lambda: evaluate(args.out) if q == "neutral_fraction" else evaluate_tb(args.out),
         "relevel": lambda: relevel(args.out)}[step]()


if __name__ == "__main__":
    main()
