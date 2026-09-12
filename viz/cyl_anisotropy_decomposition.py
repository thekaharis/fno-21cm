#!/usr/bin/env python
"""Split each model's cylindrical P(k) error into k-parallel (LOS) and
k-perpendicular (transverse) components, and decompose the U-FNO's advantage
over the other 3-D matrix models along those two axes.

Reads the cylindrical spectra saved by the matrix `ps` eval suite
(figures/shared/eval/final_eval/matrix/ps/ps_results.npz), which stores
cyl_ratio_med and cyl_r_med as (n_zbin, n_kpar, n_kperp) medians over 200 cones.

Two error channels are used:
  amplitude   |log10(P_pred / P_truth)|   -- how much power is where it should be
  decoherence 1 - r(k)                    -- whether it is the *same* structure

For each (model, reference) pair the advantage is err_ref - err_model (positive =
model better). A two-way ANOVA-style decomposition splits the variance of that
advantage map into a k_par-only part, a k_perp-only part and a residual, which
answers "is the gain a LOS gain or a broadband gain?".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_NPZ = Path("figures/shared/eval/final_eval/matrix/ps/ps_results.npz")
DEFAULT_OUT = Path("figures/shared/eval/final_eval/matrix/ps")


def load_models(npz_path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    models: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        model, _, field = key.rpartition("/")
        models.setdefault(model, {})[field] = data[key]
    return models


def error_channels(entry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """(n_zbin, n_kpar, n_kperp) error maps, lower is better."""
    ratio = np.asarray(entry["cyl_ratio_med"], dtype=float)
    coh = np.asarray(entry["cyl_r_med"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.abs(np.log10(np.where(ratio > 0, ratio, np.nan)))
    return {"amplitude": amp, "decoherence": 1.0 - coh}


def two_way_decomposition(adv: np.ndarray) -> dict[str, float]:
    """Variance share of an advantage map attributable to k_par vs k_perp.

    adv is (n_kpar, n_kperp). Additive model adv ~ mu + a(kpar) + b(kperp);
    the three shares sum to 1 because the main effects are orthogonal by
    construction on a complete grid.
    """
    mu = np.nanmean(adv)
    a = np.nanmean(adv, axis=1) - mu           # k_par main effect
    b = np.nanmean(adv, axis=0) - mu           # k_perp main effect
    resid = adv - (mu + a[:, None] + b[None, :])
    ss_par = adv.shape[1] * np.nansum(a**2)
    ss_perp = adv.shape[0] * np.nansum(b**2)
    ss_res = np.nansum(resid**2)
    total = ss_par + ss_perp + ss_res
    if total <= 0:
        return {"par_share": np.nan, "perp_share": np.nan, "resid_share": np.nan}
    return {
        "par_share": ss_par / total,
        "perp_share": ss_perp / total,
        "resid_share": ss_res / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="ufno_plain_gnorm",
                    help="model whose advantage is decomposed")
    ap.add_argument("--zbin", type=int, default=1,
                    help="index into cyl_z_centers used for the 2-D maps")
    args = ap.parse_args()

    models = load_models(args.npz)
    if args.model not in models:
        raise SystemExit(f"{args.model!r} not in {sorted(models)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_entry = models[args.model]
    kperp = np.asarray(ref_entry["k_centers"], dtype=float)
    kpar_edges = np.asarray(ref_entry["cyl_kpar_edges"], dtype=float)
    zc = np.asarray(ref_entry["cyl_z_centers"], dtype=float)
    kpar = 0.5 * (kpar_edges[:, :-1] + kpar_edges[:, 1:])

    errs = {name: error_channels(entry) for name, entry in models.items()}
    others = [m for m in sorted(models) if m != args.model]

    rows = []
    for channel in ("amplitude", "decoherence"):
        me = errs[args.model][channel]
        for other in others:
            oe = errs[other][channel]
            for zi in range(me.shape[0]):
                adv = oe[zi] - me[zi]
                dec = two_way_decomposition(adv)
                rows.append({
                    "channel": channel,
                    "model": args.model,
                    "reference": other,
                    "z": zc[zi],
                    "mean_advantage": float(np.nanmean(adv)),
                    "adv_lowest_kpar": float(np.nanmean(adv[0])),
                    "adv_highest_kpar": float(np.nanmean(adv[-1])),
                    "adv_lowest_kperp": float(np.nanmean(adv[:, 0])),
                    "adv_highest_kperp": float(np.nanmean(adv[:, -1])),
                    **{k: float(v) for k, v in dec.items()},
                })

    csv_path = args.out_dir / "cyl_anisotropy_decomposition.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[f]:.6g}" if isinstance(row[f], float)
                              else str(row[f]) for f in fields) + "\n")

    # ---- figure: marginals + one 2-D advantage map -------------------------
    zi = args.zbin
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ri, channel in enumerate(("amplitude", "decoherence")):
        me = errs[args.model][channel][zi]
        ax_par, ax_perp, ax_map = axes[ri]
        for ci, name in enumerate([args.model] + others):
            e = errs[name][channel][zi]
            style = dict(lw=2.6, color="k", zorder=5) if name == args.model else \
                    dict(lw=1.3, color=colors[ci % 10], alpha=0.85)
            ax_par.plot(kpar[zi], np.nanmean(e, axis=1), label=name, **style)
            ax_perp.plot(kperp, np.nanmean(e, axis=0), label=name, **style)
        for ax, xlab in ((ax_par, r"$k_\parallel$  [$h\,$Mpc$^{-1}$]"),
                         (ax_perp, r"$k_\perp$  [$h\,$Mpc$^{-1}$]")):
            ax.set_xscale("log")
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.25)
        ylab = (r"$|\log_{10} P_{\rm pred}/P_{\rm truth}|$" if channel == "amplitude"
                else r"$1-r(k)$")
        ax_par.set_ylabel(f"{channel}\n{ylab}")
        ax_par.set_title(f"marginal over $k_\\perp$", fontsize=10)
        ax_perp.set_title(f"marginal over $k_\\parallel$", fontsize=10)

        # advantage of the focus model over the worst reference in this channel
        worst = max(others, key=lambda n: np.nanmean(errs[n][channel][zi]))
        adv = errs[worst][channel][zi] - me
        vmax = np.nanmax(np.abs(adv)) or 1.0
        im = ax_map.pcolormesh(kperp, kpar[zi], adv, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, shading="nearest")
        ax_map.set_xscale("log")
        ax_map.set_yscale("log")
        ax_map.set_xlabel(r"$k_\perp$")
        ax_map.set_ylabel(r"$k_\parallel$")
        dec = two_way_decomposition(adv)
        ax_map.set_title(
            f"advantage over {worst}\n"
            f"$k_\\parallel$ {100*dec['par_share']:.0f}%  "
            f"$k_\\perp$ {100*dec['perp_share']:.0f}%  "
            f"resid {100*dec['resid_share']:.0f}%", fontsize=9)
        fig.colorbar(im, ax=ax_map, label="error reduction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(
        f"Cylindrical error split, 3-D $x_{{HI}}$ matrix  —  z = {zc[zi]:g}, "
        f"{int(ref_entry['n_cones'])} cones", fontsize=13)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    fig_path = args.out_dir / "cyl_anisotropy_decomposition.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
