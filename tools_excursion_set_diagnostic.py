"""How much of the x_HI morphology does a sharp-k excursion set explain?

21cmFAST (hii_filter = sharp-k) ionizes a cell when the density smoothed with a
sharp-k filter of radius R exceeds a barrier at ANY R. If that structure is
right, a differentiable excursion-set layer (filter bank + learned barrier
B(R, z, theta)) is a strong inductive bias for x_HI. This script tests the
structure before any training, against the saved network predictions of the
contiguous_ep18 run on the same test cones and the same cells.

Geometry. Each lightcone repeats one periodic 200 Mpc box along the LOS (140
cells, density correlation 0.9965 at that lag), and the transverse planes are
periodic. Any 140 consecutive LOS cells therefore form a (cyclically shifted)
full box, so a periodic 3-D FFT over such a chunk reproduces the coeval
filtering up to growth across the chunk (dz ~ 0.6 at z ~ 7).

Per evaluation slab (28 LOS cells, dz ~ 0.1) with 0.05 < <x_HI> < 0.95:

  local        threshold on the unsmoothed density
  single R     threshold on delta_R, best R
  EPS barrier  ionized iff max_R [delta_R + b s_R] > a, s_R = sqrt(s2_min - s2_R)
               (the extended Press-Schechter barrier shape; 2 numbers per slab)
  free barrier ionized iff delta_R > B_R for any R (one number per R per slab),
               fitted by exact coordinate descent on MSE
  network      contiguous_ep18 prediction (NOT fitted per slab)

Two metrics:
  IoU@f   ionized-region IoU with every method thresholded to the TRUE ionized
          fraction f -- morphology only; the global level is given to all.
  R2      1 - MSE/Var within the slab. Excursion-set methods predict 0 in
          ionized cells and the neutral-cell mean elsewhere, with the barrier
          fitted on the truth: an oracle for the barrier, not for the shape.
          The network's R2 has no oracle input.

  python tools_excursion_set_diagnostic.py --workers 12
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

MIRROR = "/pfs/10/work/hd_id260-fno_training/data/native_mirror_4f/21cmfast_11d_sample{:06d}.h5"
PRED_DIR = Path("experiments/los_windows/predictions/contiguous_ep18")
OUT = Path("experiments/excursion_set")
PERIOD = 140          # LOS cells per tiled box
SLAB = 28             # evaluation slab length (40 Mpc)
# Mpc; 21cmFAST's R_min ~ 0.62 cell. Capped at L/(2 pi) = 31.8 Mpc: beyond it
# the sharp-k cutoff falls below the fundamental mode and delta_R is a constant
# (plus round-off, which a fitted threshold would happily exploit).
RADII = np.geomspace(0.9, 31.0, 16)
B_GRID = np.linspace(0.0, 2.0, 21)
SWEEPS = 3


def sharp_k_bank(chunk, cell):
    """delta_R for every radius; keeps |k| R <= 1 (21cmFAST sharp-k)."""
    field = torch.from_numpy(chunk.astype(np.float64))
    spec = torch.fft.rfftn(field)
    n = chunk.shape
    kx = 2*np.pi*torch.fft.fftfreq(n[0], d=cell, dtype=torch.float64)
    ky = 2*np.pi*torch.fft.fftfreq(n[1], d=cell, dtype=torch.float64)
    kz = 2*np.pi*torch.fft.rfftfreq(n[2], d=cell, dtype=torch.float64)
    k = torch.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    return [torch.fft.irfftn(spec*(k*R <= 1.0), s=n).numpy() for R in RADII]


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 1.0


def top_fraction(score, f):
    """Mask of the fraction f of cells with the highest score."""
    k = int(round(f*score.size))
    mask = np.zeros(score.size, bool)
    if k:
        # exactly k cells; ties broken by position rather than all admitted
        mask[np.argpartition(-score.ravel(), k-1)[:k]] = True
    return mask.reshape(score.shape)


def binary_r2(x, ionized):
    neutral = ~ionized
    c = x[neutral].mean() if neutral.any() else 0.0
    pred = np.where(ionized, 0.0, c)
    return 1.0 - float(np.mean((pred-x)**2)/x.var())


def best_threshold(score, x, fixed):
    """Exact MSE-optimal threshold t for ionized = fixed | score > t.

    Prediction is 0 in ionized cells and the neutral mean c elsewhere, so
    MSE = const - S_n^2/n_n over the neutral set; maximize S_n^2/n_n.
    """
    free = ~fixed
    s, xv = score[free], x[free]
    order = np.argsort(s, kind="stable")
    s, xv = s[order], xv[order]
    csum = np.concatenate(([0.0], np.cumsum(xv, dtype=np.float64)))
    cnt = np.arange(len(s)+1)
    # neutral = the first j sorted free cells (score <= s[j-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(cnt > 0, csum**2/cnt, 0.0)
    # only cut between distinct scores
    valid = np.ones(len(s)+1, bool)
    valid[1:-1] = s[1:] > s[:-1]
    j = int(np.argmax(np.where(valid, gain, -np.inf)))
    if j == 0:
        return -np.inf
    if j == len(s):
        return np.inf
    return 0.5*(s[j-1]+s[j])


def fit_free_barrier(bank, x, barrier):
    barrier = barrier.copy()
    for _ in range(SWEEPS):
        for i in range(len(bank)):
            others = np.zeros(x.shape, bool)
            for j, d in enumerate(bank):
                if j != i:
                    others |= d > barrier[j]
            barrier[i] = best_threshold(bank[i], x, others)
    ionized = np.zeros(x.shape, bool)
    first = np.full(x.shape, -1)
    # 21cmFAST walks from the largest radius down; attribute each cell to the
    # largest R at which it crosses.
    for i in range(len(bank)-1, -1, -1):
        hit = (bank[i] > barrier[i]) & ~ionized
        first[hit] = i
        ionized |= hit
    counts = np.bincount(first[first >= 0], minlength=len(bank))
    return barrier, ionized, counts


def analyze_cone(cone_id):
    torch.set_num_threads(2)
    with h5py.File(MIRROR.format(cone_id)) as h, h5py.File(PRED_DIR/f"cone_{cone_id}.h5") as p:
        z = h["lightcone/lightcone_redshifts"][:]
        cell = float(np.diff(h["lightcone/lightcone_distances"][:2])[0])
        xhi_mean = h["lightcone/neutral_fraction"][::4, ::4, :].mean((0, 1))
        n = len(z)
        rows = []
        examples = []
        for s0 in range(0, n-SLAB+1, SLAB):
            if not 0.05 < xhi_mean[s0:s0+SLAB].mean() < 0.95:
                continue
            c0 = min(max(s0+SLAB//2-PERIOD//2, 0), n-PERIOD)
            dens = h["lightcone/density"][:, :, c0:c0+PERIOD]
            x = h["lightcone/neutral_fraction"][:, :, s0:s0+SLAB].astype(np.float64)
            net = p["prediction/neutral_fraction"][:, :, s0:s0+SLAB].astype(np.float64)
            core = slice(s0-c0, s0-c0+SLAB)
            bank_full = sharp_k_bank(dens, cell)
            var = np.array([b.var() for b in bank_full])
            bank = [b[:, :, core] for b in bank_full]
            local = dens[:, :, core].astype(np.float64)
            truth = x < 0.5
            f = float(truth.mean())
            row = {"cone": cone_id, "slab_start": s0, "z": float(z[s0:s0+SLAB].mean()),
                   "xhi_mean": float(x.mean()), "ionized_fraction": f, "var": float(x.var())}
            row["network_r2"] = 1.0 - float(np.mean((net-x)**2)/x.var())
            row["network_mse"] = float(np.mean((net-x)**2))
            row["network_iou"] = iou(top_fraction(-net, f), truth)
            row["local_iou"] = iou(top_fraction(local, f), truth)
            single = [iou(top_fraction(b, f), truth) for b in bank]
            row["single_iou"] = max(single)
            row["single_R"] = float(RADII[int(np.argmax(single))])
            s_r = np.sqrt(np.clip(var[0]-var, 0, None))
            best = (-1.0, None, None)
            for b in B_GRID:
                margin = np.max(np.stack([d + b*s for d, s in zip(bank, s_r)]), axis=0)
                mask = top_fraction(margin, f)
                score = iou(mask, truth)
                if score > best[0]:
                    best = (score, b, margin)
            row["eps_iou"], row["eps_b"] = best[0], float(best[1])
            a = best_threshold(best[2], x, np.zeros(x.shape, bool))
            eps_mask = best[2] > a
            row["eps_a"] = float(a)
            row["eps_r2"] = binary_r2(x, eps_mask)
            barrier, free_mask, counts = fit_free_barrier(bank, x, a - best[1]*s_r)
            row["free_r2"] = binary_r2(x, free_mask)
            row["free_iou"] = iou(free_mask, truth)
            row["free_barrier"] = [float(v) for v in barrier]
            row["free_counts"] = [int(v) for v in counts]
            rows.append(row)
            if abs(f-0.5) < 0.2:
                mid = SLAB//2
                examples.append((abs(f-0.5), {
                    "cone": cone_id, "z": row["z"], "f": f,
                    "truth": x[:, :, mid].astype(np.float32),
                    "network": net[:, :, mid].astype(np.float32),
                    "eps": (~eps_mask[:, :, mid]).astype(np.float32),
                    "local": (~top_fraction(local, f)[:, :, mid]).astype(np.float32),
                    "density": local[:, :, mid].astype(np.float32)}))
        example = min(examples, key=lambda e: e[0])[1] if examples else None
    print(f"cone {cone_id}: {len(rows)} slabs", flush=True)
    return rows, example


def summarize(rows):
    out = {}
    bins = {"late (x_HI 0.05-0.35)": (0.05, 0.35), "mid (0.35-0.65)": (0.35, 0.65),
            "early (0.65-0.95)": (0.65, 0.95), "all": (0.0, 1.0)}
    for name, (lo, hi) in bins.items():
        sel = [r for r in rows if lo < r["xhi_mean"] <= hi]
        if not sel:
            continue
        w = np.array([r["var"] for r in sel])
        entry = {"slabs": len(sel)}
        for m in ("network", "local", "single", "eps", "free"):
            entry[f"{m}_iou"] = float(np.mean([r[f"{m}_iou"] for r in sel]))
        for m in ("network", "eps", "free"):
            entry[f"{m}_r2_pooled"] = float(1 - np.sum([(1-r[f"{m}_r2"])*r["var"] for r in sel])/w.sum())
        out[name] = entry
    counts = np.sum([r["free_counts"] for r in rows], axis=0)
    out["ionizing_radius_share"] = {f"{R:.1f}": float(c/counts.sum()) for R, c in zip(RADII, counts)}
    out["eps_b_median"] = float(np.median([r["eps_b"] for r in rows]))
    return out


def plot(rows, examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, :2])
    xm = np.array([r["xhi_mean"] for r in rows])
    order = np.argsort(xm)
    style = {"network": ("k", "network (contiguous_ep18)"), "free": ("C3", "free barrier (oracle)"),
             "eps": ("C0", "EPS barrier (2 per slab)"), "single": ("C2", "best single R"),
             "local": ("C7", "unsmoothed density")}
    edges = np.linspace(0.05, 0.95, 10)
    for m, (c, lab) in style.items():
        v = np.array([r[f"{m}_iou"] for r in rows])
        ax.scatter(xm, v, s=6, color=c, alpha=0.25)
        idx = np.digitize(xm, edges)
        cen = [(xm[idx == i].mean(), v[idx == i].mean()) for i in range(1, len(edges)) if (idx == i).any()]
        ax.plot(*zip(*cen), color=c, lw=2, label=lab)
    ax.set_xlabel("slab mean x_HI"); ax.set_ylabel("ionized-region IoU at true ionized fraction")
    ax.set_title("Morphology: every method given the true ionized fraction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    cones = sorted({r["cone"] for r in rows})
    for c in cones:
        sel = [r for r in rows if r["cone"] == c]
        ax.plot([r["z"] for r in sel], [r["eps_a"] for r in sel], lw=1)
    ax.set_xlabel("z"); ax.set_ylabel("fitted EPS barrier offset a")
    ax.set_title("Barrier vs z (one line per cone)"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    counts = np.sum([r["free_counts"] for r in rows], axis=0)
    ax.bar(range(len(RADII)), counts/counts.sum())
    ax.set_xticks(range(len(RADII))[::2]); ax.set_xticklabels([f"{R:.1f}" for R in RADII[::2]], rotation=45)
    ax.set_xlabel("R [Mpc] (largest radius crossing the barrier)"); ax.set_ylabel("share of ionized cells")
    ax.set_title("Which scales ionize (free barrier)")

    if examples:
        e = examples[0]
        for i, (key, title) in enumerate([("truth", "truth x_HI"), ("network", "network"),
                                          ("eps", "EPS excursion set"), ("local", "unsmoothed density")]):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(e[key].T, origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{title}\ncone {e['cone']}, z={e['z']:.2f}, ionized {e['f']:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cones", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    cones = args.cones or sorted(int(p.stem.split("_")[1]) for p in PRED_DIR.glob("cone_*.h5"))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(analyze_cone, cones)
    else:
        results = [analyze_cone(c) for c in cones]
    rows = [r for rs, _ in results for r in rs]
    examples = [e for _, e in results if e is not None]
    summary = {"cones": cones, "radii_mpc": [float(R) for R in RADII], "slab_cells": SLAB,
               "chunk_cells": PERIOD, "summary": summarize(rows)}
    (args.out/"slabs.json").write_text(json.dumps(rows, indent=1))
    (args.out/"summary.json").write_text(json.dumps(summary, indent=1))
    plot(rows, examples, args.out/"excursion_set_diagnostic.png")
    print(json.dumps(summary["summary"], indent=1))


if __name__ == "__main__":
    main()
