"""Detailed per-field diagnostics for a multi-field lightcone prediction.

Companion to viz/visualize_3d_detailed.py, but for models with more than one
target. Each target field gets its own page, because the fields live on
different scales and fail in different ways: x_HI is bounded and morphological,
brightness temperature is unbounded, signed and heavy tailed.

Per field, one page of:
  row 1  truth / prediction / signed error, as an x-z lightcone slice
  row 2  transverse maps at three redshifts (truth over prediction)
  row 3  error vs redshift, parity (hedging) and the error distribution
  row 4  transverse power spectrum ratio and cross-correlation r(k)

Transverse spectra only: the 256-point LOS grid discards most of the
line-of-sight small-scale power (experiments/multifield/pilot_report.md), so a
full 3-D spectrum would largely measure the cache.

  python -m viz.multifield_detailed --prediction experiments/.../cone_541.h5 \
      --out-dir figures/3d_xhi/mf_cnn_fno
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from viz._zaxis import edges, transverse_edges

BOX_MPC = 200.0


def transverse_spectrum(a, b, n_bins=12):
    """Ring-averaged P(k_perp) of a and b, plus their cross-correlation."""
    nx, ny = a.shape[:2]
    kx = np.fft.fftfreq(nx, d=BOX_MPC / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=BOX_MPC / ny) * 2 * np.pi
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = np.logspace(np.log10(kk[kk > 0].min()), np.log10(kk.max() * 0.7), n_bins + 1)
    idx = np.digitize(kk.ravel(), edges) - 1
    fa = np.fft.fft2(a - a.mean(axis=(0, 1)), axes=(0, 1))
    fb = np.fft.fft2(b - b.mean(axis=(0, 1)), axes=(0, 1))
    pa = (np.abs(fa) ** 2).mean(axis=2).ravel()
    pb = (np.abs(fb) ** 2).mean(axis=2).ravel()
    px = (fa * np.conj(fb)).mean(axis=2).real.ravel()
    centers, ratio, coh = [], [], []
    for i in range(n_bins):
        m = idx == i
        if m.sum() < 4:
            continue
        A, B, X = pa[m].mean(), pb[m].mean(), px[m].mean()
        centers.append(np.sqrt(edges[i] * edges[i + 1]))
        ratio.append(B / A if A > 0 else np.nan)
        coh.append(X / np.sqrt(A * B) if A > 0 and B > 0 else np.nan)
    return np.array(centers), np.array(ratio), np.array(coh)


def page(field, truth, pred, z, cone_id, units, out_path, epoch_note=""):
    err = pred - truth
    nx, ny, nz = truth.shape
    # Choose the transverse row where the field is most active, so the maps are
    # not three near-identical frames of quiescent high-redshift gas.
    activity = truth.std(axis=(0, 1))
    zc = int(np.argmax(activity))
    # Three slices spread across the band where the field actually varies.
    # Offsetting from the peak by a fixed fraction of the LOS could land on a
    # fully ionized (empty) slice, or collapse to fewer than three near an edge.
    active = np.flatnonzero(activity > 0.05 * activity.max()) if activity.max() > 0 \
        else np.arange(nz)
    lo, hi = int(active[0]), int(active[-1])
    picks = sorted({int(round(lo + f * (hi - lo))) for f in (0.2, 0.5, 0.8)})

    diverging = field != "neutral_fraction"
    cmap = "viridis" if not diverging else "magma"
    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.34, wspace=0.24,
                          height_ratios=[1.0, 1.0, 0.85, 0.85])

    # ---- row 1: lightcone slice through the middle of the box ----------------
    mid = nx // 2
    lo, hi = np.percentile(truth[mid], [1, 99])
    xe, ze = transverse_edges(ny, BOX_MPC), edges(z)
    for col, (name, data) in enumerate((("truth", truth[mid]), ("prediction", pred[mid]))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=lo, vmax=hi,
                           shading="flat", rasterized=True)
        ax.set_title(f"{name}", fontsize=11)
        ax.set_xlabel("x [Mpc]"); ax.set_ylabel("redshift" if col == 0 else "")
        fig.colorbar(im, ax=ax, label=units if col == 1 else "")
    ax = fig.add_subplot(gs[0, 2])
    lim = np.percentile(np.abs(err[mid]), 99) or 1.0
    im = ax.pcolormesh(xe, ze, err[mid].T, cmap="RdBu_r",
                       norm=TwoSlopeNorm(0.0, -lim, lim), shading="flat", rasterized=True)
    ax.set_title("prediction - truth", fontsize=11)
    ax.set_xlabel("x [Mpc]")
    fig.colorbar(im, ax=ax, label=units)

    # ---- row 2: transverse maps at three redshifts ---------------------------
    for col, j in enumerate(picks):
        ax = fig.add_subplot(gs[1, col])
        # Join along x (axis 0) so the transposed image is [truth | prediction]
        # side by side. Joining along y put truth in the bottom half and the
        # prediction in the top half, each squashed, under a "truth | prediction"
        # label -- every row-2 panel before 2026-09-21 was drawn that way.
        pair = np.concatenate([truth[:, :, j], pred[:, :, j]], axis=0)
        v0, v1 = np.percentile(truth[:, :, j], [1, 99])
        im = ax.imshow(pair.T, origin="lower", cmap=cmap, vmin=v0, vmax=v1,
                       extent=[0, 2 * BOX_MPC, 0, BOX_MPC])
        ax.axvline(BOX_MPC, color="w", lw=1.2)
        ax.set_title(f"z = {z[j]:.2f}   truth | prediction", fontsize=10)
        ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]" if col == 0 else "")
        fig.colorbar(im, ax=ax, fraction=0.025)

    # ---- row 3: error vs z, parity, error distribution -----------------------
    ax = fig.add_subplot(gs[2, 0])
    rmse = np.sqrt((err ** 2).mean(axis=(0, 1)))
    bias = err.mean(axis=(0, 1))
    ax.plot(z, rmse, color="#c0392b", label="RMSE")
    ax.plot(z, bias, color="#2c6fbb", label="mean bias")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("redshift"); ax.set_ylabel(units)
    ax.set_title("error vs redshift", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[2, 1])
    sub = (slice(None, None, 3), slice(None, None, 3), slice(None, None, 2))
    t, p = truth[sub].ravel(), pred[sub].ravel()
    ax.hexbin(t, p, gridsize=70, bins="log", cmap="inferno", mincnt=1)
    span = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(span, span, "w--", lw=1.0)
    ax.set_xlabel(f"truth [{units}]"); ax.set_ylabel(f"prediction [{units}]")
    # Regression slope < 1 is the hedging signature: predictions pulled to the mean.
    # Closed form in float64. np.polyfit's default rcond is len(x) * eps(dtype):
    # with float32 and millions of voxels that reaches ~0.3 and silently
    # truncates the fit whenever the truth sits near one value (a 0.965 slope
    # came out as 0.497 on a mostly neutral cone).
    t64, p64 = t.astype(np.float64), p.astype(np.float64)
    var_t = t64.var()
    slope = float(((t64 - t64.mean()) * (p64 - p64.mean())).mean() / var_t) if var_t > 0 else float("nan")
    ax.set_title(f"parity — slope {slope:.3f}", fontsize=10)

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(err[sub].ravel(), bins=120, color="#555", log=True)
    ax.axvline(0, color="r", lw=0.8)
    ax.set_xlabel(f"error [{units}]"); ax.set_ylabel("voxels")
    ax.set_title(f"error distribution — RMSE {np.sqrt((err**2).mean()):.4g}", fontsize=10)

    # ---- row 4: transverse power spectrum ------------------------------------
    band = slice(max(0, zc - 24), min(nz, zc + 24))
    k, ratio, coh = transverse_spectrum(truth[:, :, band], pred[:, :, band])
    ax = fig.add_subplot(gs[3, 0])
    ax.semilogx(k, ratio, "o-", color="#2c6fbb")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]"); ax.set_ylabel(r"$P_{\rm pred}/P_{\rm truth}$")
    ax.set_title("transverse power ratio", fontsize=10); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogx(k, coh, "o-", color="#c0392b")
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]"); ax.set_ylabel(r"$r(k_\perp)$")
    ax.set_title("cross-correlation", fontsize=10); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 2]); ax.axis("off")
    good = coh[np.isfinite(coh)]
    kc = k[np.isfinite(coh)]
    below = kc[good < 0.9]
    lines = [
        f"cone {cone_id}   field: {field}",
        f"units: {units}",
        f"RMSE            {np.sqrt((err**2).mean()):.5g}",
        f"MAE             {np.abs(err).mean():.5g}",
        f"mean bias       {err.mean():+.5g}",
        f"Pearson r       {np.corrcoef(t, p)[0,1]:.5f}",
        f"parity slope    {slope:.4f}",
        f"truth std       {truth.std():.5g}",
        f"pred  std       {pred.std():.5g}",
        f"std ratio       {pred.std()/ (truth.std() or 1):.4f}",
        f"r(k) < 0.9 from {below[0]:.3f} Mpc^-1" if len(below) else "r(k) >= 0.9 throughout",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10.5)

    fig.suptitle(f"{field} — cone {cone_id}{epoch_note}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    return dict(field=field, cone=cone_id, rmse=float(np.sqrt((err**2).mean())),
                bias=float(err.mean()), r=float(np.corrcoef(t, p)[0, 1]),
                slope=float(slope), std_ratio=float(pred.std()/(truth.std() or 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import h5py
    summary = []
    for path in args.prediction:
        with h5py.File(path, "r") as f:
            z = f["target_z"][:]
            cone = int(f.attrs["cone_id"])
            for field in f["target"]:
                truth = f[f"target/{field}"][:].astype(np.float32)
                pred = f[f"prediction/{field}"][:].astype(np.float32)
                units = f[f"target/{field}"].attrs.get("units", "")
                out = args.out_dir / f"cone{cone}_{field}.png"
                summary.append(page(field, truth, pred, z, cone, units, out, args.note))
                print(f"wrote {out}")
    rows = args.out_dir / "summary.json"
    rows.write_text(json.dumps(summary, indent=2))
    print(f"wrote {rows}")


if __name__ == "__main__":
    main()
