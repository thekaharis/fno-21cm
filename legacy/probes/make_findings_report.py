"""Generate FINDINGS-<date>.md from the run artifacts.

Tables are computed from final_report.json / metrics.jsonl / run_metadata.json
so the report cannot drift from the data; the prose is maintained here.

    python -m util.make_findings_report [--out FINDINGS-2026-07-21.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Truth variances, measured over the held-out test splits (see report text).
ZRE_VAR = {"dense_total": 0.013871, "dense_struct": 0.000146,
           "mask_total": 0.015789, "mask_struct": 0.000288}
X_HI_VAR = {"total": 0.09322, "struct": 0.01237}


def param_counts() -> dict[str, str]:
    """Model parameter counts, scraped from each run's job log."""
    out: dict[str, str] = {}
    for log in ROOT.glob("logs/*.out"):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"Model: .*?-> ([\d,]+) parameters", text)
        if not m:
            continue
        for d in set(re.findall(r"checkpoints/(checkpoints_\S+?)/", text)):
            out.setdefault(d, m.group(1))
    return out


def zre_rows():
    params = param_counts()
    rows = []
    for d in sorted(glob.glob(str(ROOT / "checkpoints/zre/*/checkpoints_zre*"))):
        rep = Path(d) / "final_report.json"
        if not rep.exists():
            continue
        r = json.loads(rep.read_text())
        try:
            meta = json.loads((Path(d) / "run_metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        mc, tr = meta.get("model_config", {}), meta.get("training", {})
        lw = tr.get("loss_weights", {})
        loss = ("L2-only" if lw.get("l2") and not lw.get("h1")
                else "L2+H1" if lw.get("h1") else "?")
        rows.append({
            "name": os.path.basename(d).replace("checkpoints_zre_", "") or "zre",
            "kind": mc.get("kind", "?"),
            "width": (mc.get("localfno_base_width") or mc.get("ufno_width")
                      or mc.get("hidden_channels")),
            "omega": mc.get("siren_omega"),
            "lr": tr.get("learning_rate"),
            "epochs": tr.get("epochs"),
            "loss": loss,
            "rmse": r["test_rmse_z"],
            "mrmse": r["test_rmse_masked_z"],
            "ratio": r["test_rmse_z"] / r["train_rmse_z"],
            "r2_struct": 1 - r["test_mse_norm"] / ZRE_VAR["dense_struct"],
            "r2_total": 1 - r["test_mse_norm"] / ZRE_VAR["dense_total"],
            "params": params.get(os.path.basename(d), ""),
        })
    return sorted(rows, key=lambda x: x["rmse"])


def xhi2d_rows():
    """2-D x_HI slice runs (checkpoints/2d_xhi/*/xhi2d_*), from final_report.json."""
    params = param_counts()
    rows = []
    for d in sorted(glob.glob(str(ROOT / "checkpoints/2d_xhi/*/xhi2d_*"))):
        rep = Path(d) / "final_report.json"
        if not rep.exists():
            continue
        r = json.loads(rep.read_text())
        try:
            meta = json.loads((Path(d) / "run_metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        mc, tr = meta.get("model_config", {}), meta.get("training", {})
        lw = tr.get("loss_weights", {})
        loss = ("L2-only" if lw.get("l2") and not lw.get("h1")
                else f"L2+{lw['h1']:g}H1" if lw.get("h1") else "?")
        kind = mc.get("kind", "?")
        if kind == "localop":
            ordering = (mc.get("local_operator_kwargs") or {}).get("ordering") \
                or (mc.get("global_operator_kwargs") or {}).get("ordering")
            kind = (f"localop({mc.get('local_operator','?')}/"
                   f"{mc.get('global_operator','?')})"
                   + (f" [{ordering}]" if ordering not in (None, "sequency")
                      else ""))
        rows.append({
            "name": os.path.basename(d).replace("xhi2d_", ""),
            "kind": kind,
            "width": (mc.get("localfno_base_width") or mc.get("ufno_width")),
            "rank": mc.get("localfno_spectral_rank"),
            "lr": tr.get("learning_rate"),
            "loss": loss,
            "val_rmse": r.get("val_rmse"),
            "test_rmse": r.get("test_rmse"),
            "ratio": (r["val_rmse"] / r["train_rmse"]
                      if r.get("train_rmse") else None),
            "params": params.get(os.path.basename(d), ""),
        })
    return sorted(rows, key=lambda x: (x["val_rmse"] is None, x["val_rmse"]))


def three_d_rows():
    rows = []
    patterns = ["checkpoints/3d_xhi/*/checkpoints_3d_*", "checkpoints_3d_*",
                "checkpoint-archive/checkpoints_3d_*",
                "checkpoint-archive/checkpoints_ufno*"]
    for pattern in patterns:
        for d in glob.glob(str(ROOT / pattern)):
            mfile = Path(d) / "metrics.jsonl"
            if not mfile.is_file():
                continue
            ev = [json.loads(l) for l in mfile.read_text().splitlines()
                  if l.strip()]
            ev = [r for r in ev if "val_l2" in r and "test_l2" in r]
            if not ev:
                continue
            best = min(ev, key=lambda r: r["val_l2"])
            name = os.path.basename(d)
            if "warped" in name:          # different dataset, not comparable
                continue
            mse = best["test_l2"] ** 2
            rows.append({
                "name": ("archive/" if "checkpoint-archive" in d else "") + name,
                "rmse": best["test_l2"],
                "epoch": best["epoch"],
                "n_epochs": len(ev),
                "r2_total": 1 - mse / X_HI_VAR["total"],
                "r2_struct": 1 - mse / X_HI_VAR["struct"],
            })
    return sorted(rows, key=lambda x: x["rmse"])


def fmt(v, spec=""):
    if v is None:
        return "–"
    return format(v, spec) if spec else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"FINDINGS-{date.today():%Y-%m-%d}.md")
    args = ap.parse_args()

    zre, td, x2 = zre_rows(), three_d_rows(), xhi2d_rows()
    L = []
    A = L.append

    A(f"# fno-21cm — findings ({date.today():%Y-%m-%d})\n")
    A("Auto-generated by `util/make_findings_report.py`; tables are computed "
      "from `final_report.json` / `metrics.jsonl` / `run_metadata.json` in the "
      "checkpoint directories. Prose is maintained in that script.\n")

    A("## Headline results\n")
    A("1. **Pure L2 beats L2+H1 on the z_re task by ~35%** for every "
      "architecture tested — and improves the H1 metric itself, so the H1 "
      "term was not buying gradient fidelity.")
    A("2. **SIREN-generated spectral weights transform LocalFNO**: on z_re it "
      "goes from worst (RMSE 0.305) to second best (0.123), and the "
      "train→test gap collapses from ~3.0x to ~1.2x.")
    A("3. **Nominal loss weights are misleading.** With absolute-mode losses "
      "at 0.5/0.5, the H1 term supplies **99.4%** of the 3-D training loss "
      "(raw H1 ≈ 180x raw L2); in relative mode it is ~88%. Every archived "
      "\"L2+H1\" 3-D run is effectively H1-dominated.")
    A("4. **Model capacity (width) is the strongest hyperparameter** on z_re; "
      "spectral bandwidth knobs (more local modes, bigger windows) do not "
      "help there, consistent with the mode-weight diagnostics.")
    A("5. **Structured-transform operators (wavelet, Walsh-Hadamard) are "
      "competitive with Fourier on 2-D x_HI, and the best model so far "
      "swaps in Walsh-Hadamard for the GLOBAL bottleneck only** "
      "(`localop(fourier/hadamard)`, val RMSE 0.1453 vs LocalFNO's 0.1487; "
      "beats U-FNO by ~9%). Putting the same operator in the *local* windowed "
      "slot instead consistently underperforms LocalFNO — across both the "
      "wavelet and Hadamard families, the global whole-field bottleneck is "
      "where a non-Fourier operator pays off, not the local branches. "
      "LocalWNO's earlier large apparent lead over 3-D baselines was mostly "
      "a z-interpolation handicap on those baselines, not the operator "
      "itself; its genuine edge is stage-localized (strong in the "
      "neutral/sparse-bubble regime).\n")

    A("## 1. z_re task (2-D: density lightcone -> z_re(x, y) map)\n")
    A("Test-set metrics, 660 held-out cones. `RMSE(z)` is pooled over all "
      "pixels, in redshift units. `masked` restricts to pixels whose "
      "transition falls inside the cone. `R2_struct` is measured against the "
      "**within-cone** variance (spatial pattern of the front); `R2_total` "
      "uses total variance, which is dominated by cone-to-cone timing and so "
      "saturates near 1 for every model.\n")
    A("| run | kind | width | ω | lr | epochs | loss | RMSE(z) | masked | "
      "R2_struct | test/train | params |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in zre:
        A(f"| {r['name']} | {r['kind']} | {fmt(r['width'])} | "
          f"{fmt(r['omega'], '.0f') if r['omega'] else ('30' if 'siren' in r['kind'] else '–')} | "
          f"{fmt(r['lr'], '.0e')} | {fmt(r['epochs'])} | {r['loss']} | "
          f"{r['rmse']:.4f} | {r['mrmse']:.4f} | {r['r2_struct']:.3f} | "
          f"{r['ratio']:.2f}x | {r['params'] or '–'} |")
    A("")
    A("### Sweep conclusions (Local-SirenFNO, 14 runs)\n")
    A("- **Width saturates at 32.** bw16 -> bw24 -> bw32 improves "
      "monotonically; bw48 is consistently *worse* than bw32 at matched "
      "settings, despite 300-epoch schedules.")
    A("- **Higher LR wins at higher width**, but only over a full schedule: "
      "at epoch ~50 the lr 1e-4 run led, yet the final ordering is "
      "3e-4 > 2e-4 > 1e-4 at bw32. Mid-run LR comparisons are unreliable.")
    A("- **ω=60 helps at low width, not at high**: -6.5% at bw16, -2.4% at "
      "bw32/lr2e-4, neutral at bw48. ω=15 is clearly worse than the ω=30 "
      "default.")
    A("- **Bandwidth knobs hurt**: local modes 6->8 (`m88`) and window "
      "16->32 with modes 12 (`win32m1212`) both land below the baseline. "
      "The z_re target has 7-11x *less* small-scale transverse power than "
      "x_HI slices, so this negative result should **not** be transferred to "
      "the 3-D task.")
    A("- **Best config: bw32, ω=30, lr 3e-4, pure L2, 300 epochs** -> "
      "RMSE(z) 0.1233, beating the U-FNO baseline by 18% with ~54x fewer "
      "parameters. The plain FNO still leads at 0.1168 (a 5.6% gap).\n")
    A("### Failure modes found\n")
    A("- **SirenFNO NaN (2-D).** Mostly-zero target -> sigmoid saturation "
      "(100% of outputs at the rails by batch ~40) -> on fully-clamped cones "
      "prediction and target are both exactly 0 -> the L2 norm is exactly 0 "
      "-> `PowBackward0` (the sqrt) returns NaN -> weights poisoned. "
      "Loss-agnostic (pure L2 fails too); gradient clipping cannot help "
      "because NaN clips to NaN. Fixes: decouple the DC mode from the "
      "SIREN-generated weights, and use a **linear output head** "
      "(`SIREN_OUTPUT_SIGMOID=0`, now the default).")
    A("- **Relative-mode L1+H2 run froze.** Loss 2e11 at epoch 0, then "
      "constant 6.562121 from epoch 2 onwards: relative normalisation "
      "divides by a near-zero target norm, blows up the weights, saturates, "
      "and gradients die. Use absolute mode for L1/H2 on this target.\n")

    A("## 2. 2-D x_HI slice task (density slice -> x_HI slice)\n")
    A("All models trained and evaluated on the SAME 2-D task, split, and "
      "resolution (native slice prediction, no z-interpolation), so this is "
      "the clean architecture comparison. `RMSE` in x_HI units; `val/train` "
      "is the generalization gap. 30 epochs each.\n")
    A("| run | kind | width | rank | lr | loss | val RMSE | test RMSE | "
      "val/train | params |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in x2:
        A(f"| {r['name']} | {r['kind']} | {fmt(r['width'])} | "
          f"{fmt(r['rank'])} | {fmt(r['lr'], '.0e')} | {r['loss']} | "
          f"{fmt(r['val_rmse'], '.4f')} | {fmt(r['test_rmse'], '.4f')} | "
          f"{fmt(r['ratio'], '.2f')}x | {r['params'] or '–'} |")
    A("")
    A("- **Best model: Walsh-Hadamard in the GLOBAL slot only** "
      "(`localop(fourier/hadamard)`, val RMSE **0.1453**, test **0.1502**) — "
      "the first run to beat plain LocalFNO (0.1487) on this task. "
      "`localop(hadamard/hadamard)` (both slots) is a close second (0.1501); "
      "every *local*-slot-Hadamard variant (`localop(hadamard/fourier)` and "
      "its width/mode/ordering variants) instead sits **below** LocalFNO, "
      "0.151-0.153.")
    A("- **The local/global split matters more than which basis.** Six "
      "Walsh-Hadamard configs were run varying which U-Net slot gets the "
      "operator (local, global, or both), local mode count (6 vs 12), width "
      "(32 vs 48), and coefficient ordering (sequency vs natural, i.e. "
      "smoothest-first truncation vs Kronecker/bit-reversed order). Sequency "
      "vs natural ordering made only a small difference (0.1520 vs 0.1533 at "
      "matched settings); moving the operator from local-only to "
      "global-only changed the result by ~5%, more than any other knob in "
      "the sweep. Combined with LocalWNO (wavelet-**local**, Fourier-global) "
      "landing close to LocalFNO rather than clearly ahead, the pattern "
      "across both structured-transform families is that **the global "
      "whole-field bottleneck is where a well-chosen non-Fourier operator "
      "pays off on this task**, not the windowed local branches.")
    A("- **U-FNO generalizes worst** (val/train 1.19x vs ~1.08x for the local "
      "models), consistent with the z_re and 3-D tasks.")
    A("- **Caveat vs the earlier 2-D-vs-3-D figures**: those compared the 2-D "
      "LocalWNO against 3-D U-FNO/LocalFNO whose predictions were "
      "z-interpolated onto each slice, a handicap that inflated WNO's lead. "
      "Out-of-sample there WNO still led (test-cone RMSE 0.214 vs U-FNO "
      "0.241), but the same-task numbers above are the fairer comparison. "
      "WNO's robust advantage is stage-localized: strongest where the field "
      "is mostly neutral with sparse compact bubbles (wavelet sparsity), "
      "weakest / negative in the nearly-fully-ionized regime.\n")

    A("## 3. 3-D task (x_HI lightcone cubes)\n")
    A("`RMSE` is the mean per-cone RMSE in x_HI units (= `val_l2`/`test_l2` "
      "with absolute LpLoss and unit measure). Test-set variance: total "
      f"{X_HI_VAR['total']:.5f} (std {X_HI_VAR['total']**0.5:.3f}), "
      f"within-slice {X_HI_VAR['struct']:.5f} "
      f"(std {X_HI_VAR['struct']**0.5:.3f}) — i.e. **87% of the variance is "
      "the global reionization history**, which is why `R2_total` saturates "
      "and `R2_struct` is the meaningful column.\n")
    A("| run | best test RMSE | @epoch | epochs | R2_total | R2_struct |")
    A("|---|---|---|---|---|---|")
    for r in td:
        A(f"| {r['name']} | {r['rmse']:.4f} | {r['epoch']} | {r['n_epochs']} | "
          f"{r['r2_total']:.4f} | {r['r2_struct']:.4f} |")
    A("")
    A("Runs on the warped256 dataset are excluded: different target grid, "
      "not comparable.\n")
    A("### Diagnostics\n")
    A("- **Mode-weight profiles (3-D LocalFNO).** Edge/peak weight ratio is "
      "0.03-0.20 on the x and y axes but **0.60-0.78 on the LOS (z) axis** "
      "in the bottleneck and decoders. The binding bandwidth constraint in "
      "3-D is the LOS direction — which the 2-D z_re task cannot inform. "
      "`LOCALFNO_MODES_Z` (12, cap 17) and the bottleneck z modes (16, cap "
      "33) both have headroom.")
    A("- **Power spectra.** Both U-FNO and LocalFNO lose transverse power "
      "above k_perp ~ 1 Mpc^-1 (the front-smoothing signature). LocalFNO "
      "additionally *gains* spurious low-k power at early epochs, growing "
      "with redshift — consistent with a window-grid artifact. Figures: "
      "`figures/3d_xhi/eval/ps_out/localfno-vs-ufno-contrastz7/`.")
    A("- **Boundary band.** Truth front width is 3.6 Mpc; both models "
      "predict 12-14 Mpc, i.e. fronts ~3.5x too smooth, despite the "
      "H1-dominated loss.\n")

    A("## 4. Infrastructure changes\n")
    A("- **z_re input cache** (`dataset/zre_input_cache.py`, "
      "`slurm/build_zre_inputs.sbatch`): per-cone density slices + params in "
      "one HDF5. Startup drops from ~2 h (re-reading 6600 raw lightcones) to "
      "minutes; verified byte-identical to raw reads.")
    A("- **`metrics.jsonl` + `run_metadata.json` for z_re runs** "
      "(`ZreLoggingTrainer`), so z_re runs appear on the dashboard.")
    A("- **Dashboard**: per-task pages (21cm 3-D / z_re), richer run labels, "
      "and a config comparison table that highlights only the parameters "
      "that differ across the selected runs.")
    A("- **Metadata backfill** (`util/backfill_zre_metadata.py`): "
      "reconstructs architecture fields for historical z_re runs from job "
      "logs into the checkpoint dirs. ω was never logged, so for the "
      "historical sweep it is inferred from the directory name and flagged "
      "`siren_omega_inferred_from_dirname`.")
    A("- **`fno_zre.py`** now records the full architecture per model kind, "
      "supports `RESUME_DIR`, and applies `GRAD_CLIP_NORM` (default 1.0 for "
      "SIREN kinds).\n")

    A("## 5. Open questions\n")
    A("- Does pure L2 also win in 3-D? Being measured now "
      "(`checkpoints_3d_localsirenfno` vs `..._l2only`).")
    A("- Does the z_re width finding transfer? bw32/bw48 with ω=60 are "
      "running on the 3-D task.")
    A("- A genuinely **balanced** L2+H1 has never been tested in 3-D: "
      "queued as `..._bw32om60_relbal` (relative mode, weights 0.5/0.07).")
    A("- LOS bandwidth (`LOCALFNO_MODES_Z`) is the untested 3-D-specific "
      "lever indicated by the mode-weight diagnostic.")
    A("- LocalFNO (non-SIREN) overfitting: augmentation (random transverse "
      "rolls + D4 symmetries) and AdamW remain untried.")
    A("- Does global-only Walsh-Hadamard (`localop(fourier/hadamard)`, the "
      "new best 2-D x_HI model) transfer to 3-D and to z_re? Untested; the "
      "2-D-vs-3-D lesson above says a 2-D win is not a reliable predictor.")

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(zre)} z_re runs, {len(td)} 3-D runs, "
          f"{out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
