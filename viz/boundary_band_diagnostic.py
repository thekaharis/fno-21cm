#!/usr/bin/env python3
"""Boundary-band diagnostic for x_HI lightcone predictions.

Motivation
----------
Whole-volume L2/H1 on the density -> x_HI map is dominated by trivially-correct
saturated regions (fully neutral at high z, fully ionized at low z).  The error
that matters lives in a thin shell around the ionization fronts (bubble walls),
and it is mostly a *gradient* error (diffuse walls) -- which is exactly why the
U-FNO win showed up in H1, not L2.

This module localises error by *distance to the nearest ionization front* so you
can (a) confirm the error is front-dominated, (b) measure how sharp each model's
fronts are, and (c) discriminate models that look identical on whole-volume L2.

It is the real-space complement to P(k)/r(k): a diffuse front is suppressed
high-k power and low cross-correlation at high k, so the two diagnostics should
agree.  If SirenFNO's flat spectrum buys anything physical it should show up here
as a narrower front and lower boundary-band H1 even where whole-volume L2 is flat.

Design choices specific to this lightcone
-----------------------------------------
* Cubes are ``(Nx, Ny, Nz) = (140, 140, 256)``; the LOS axis (z) is uniform in
  *redshift* (z: 5 -> 25), NOT comoving distance, and it mixes geometry with
  cosmic evolution.  So the PRIMARY diagnostic is a 2-D per-slice *transverse*
  signed distance (``mode="slice"``): pure bubble-wall geometry, no LOS spacing
  ambiguity.  A 3-D mode is available as an approximate cross-check (uses a
  single mean LOS cell size; see ``los_cell_mpc``) -- read it with that caveat.
* The signed distance is computed from the TRUE field (threshold 0.5): positive
  into the neutral side, negative into the ionized side, ~0 at the front.
* Distance transforms use physical ``sampling`` in Mpc (transverse cell ~1.43
  Mpc), so anisotropy is handled correctly.
* Only slices that actually contain a front contribute; fully neutral/ionized
  slices have no boundary and are skipped (they would just add trivial zeros).

The core engine (numpy + scipy.ndimage) is torch-free and unit-tested via
``--selftest``.  Torch / project imports are needed only for the cluster
``--checkpoints`` driver and are imported lazily.

Typical use
-----------
Cluster (generates cubes from checkpoints, caches npz, writes figure + CSV)::

    python -m viz.boundary_band_diagnostic --checkpoints \
        fno=checkpoints/checkpoints_3d/best_model_state_dict.pt \
        ufno=checkpoints/checkpoints_3d_ufno/best_model_state_dict.pt \
        sirenfno=checkpoints/checkpoints_3d_sirenfno_m64_stable/best_model_state_dict.pt \
        --n-cones 200 --split test --save-cubes cubes_cache/ --out figures/band_out/

Offline (re-use saved cubes; runs anywhere)::

    python -m viz.boundary_band_diagnostic --manifest band_manifest.json --out figures/band_out/

Self-test (no data needed)::

    python -m viz.boundary_band_diagnostic --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required: conda activate 21cmfast  (or pip install scipy)") from exc


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BOX_MPC = 200.0          # transverse comoving box size (Mpc) -> dx = dy
N_TRANSVERSE = 140       # transverse cells
DEFAULT_DX = BOX_MPC / N_TRANSVERSE   # ~1.4286 Mpc / cell


@dataclass
class BandConfig:
    """Binning / threshold configuration for the diagnostic."""
    threshold: float = 0.5            # x_HI level defining the front
    dx_mpc: float = DEFAULT_DX        # transverse cell size (x)
    dy_mpc: float = DEFAULT_DX        # transverse cell size (y)
    los_cell_mpc: float = 13.0        # mean LOS cell (Mpc), 3-D mode only (approx)
    mode: str = "slice"               # "slice" (2-D transverse) or "3d" (approx)
    d_max_mpc: float = 30.0           # distance range for the profile
    bin_mpc: float = 1.5              # profile bin width (~1 transverse cell)
    bands_mpc: tuple[float, ...] = (2.0, 5.0)  # band half-widths for band metrics
    z_window: tuple[float, float] | None = None  # restrict slices to z in [lo, hi]

    @property
    def bin_edges(self) -> np.ndarray:
        n = int(round(2 * self.d_max_mpc / self.bin_mpc))
        return np.linspace(-self.d_max_mpc, self.d_max_mpc, n + 1)


# --------------------------------------------------------------------------- #
# Core engine (torch-free)
# --------------------------------------------------------------------------- #
def _signed_distance_2d(mask: np.ndarray, sampling: tuple[float, float]) -> np.ndarray:
    """Signed distance to the True/False boundary of a 2-D mask.

    Positive inside the True (neutral) region, negative inside False (ionized),
    in physical units given by ``sampling``.  Returns NaN everywhere if the slice
    has no boundary (all-True or all-False).
    """
    if mask.all() or (~mask).all():
        return np.full(mask.shape, np.nan, dtype=np.float64)
    d_in = distance_transform_edt(mask, sampling=sampling)        # neutral side >0
    d_out = distance_transform_edt(~mask, sampling=sampling)      # ionized side >0
    return d_in - d_out


def _grad_sqdiff_2d(pred: np.ndarray, truth: np.ndarray,
                    dx: float, dy: float) -> np.ndarray:
    """Per-voxel squared in-plane gradient error |grad(pred-truth)|^2 (H1 seminorm)."""
    diff = pred - truth
    gx, gy = np.gradient(diff, dx, dy)
    return gx * gx + gy * gy


@dataclass
class BoundaryBandAccumulator:
    """Aggregates the diagnostic over many cones, memory-safely.

    Stores per-bin pooled sums (mean +/- std reconstructable) and per-cone band
    metrics (for a paired bootstrap across cones).  Does NOT store raw voxels.
    """
    cfg: BandConfig
    # pooled, distance-binned accumulators
    n: np.ndarray = field(init=False)
    sum_sq: np.ndarray = field(init=False)      # sum of (pred-truth)^2
    sum_sq2: np.ndarray = field(init=False)     # sum of (pred-truth)^4  (for std)
    sum_grad: np.ndarray = field(init=False)    # sum of |grad diff|^2
    sum_truth: np.ndarray = field(init=False)   # sum of truth (front profile)
    sum_pred: np.ndarray = field(init=False)    # sum of pred  (front profile)
    total_sq_err: float = 0.0                   # over ALL contributing voxels
    per_cone: list[dict] = field(default_factory=list)

    def __post_init__(self):
        nb = len(self.cfg.bin_edges) - 1
        z = lambda: np.zeros(nb, dtype=np.float64)
        self.n, self.sum_sq, self.sum_sq2 = z(), z(), z()
        self.sum_grad, self.sum_truth, self.sum_pred = z(), z(), z()

    # -- ingestion -------------------------------------------------------- #
    def add_cone(self, cone_id, pred: np.ndarray, truth: np.ndarray):
        cfg = self.cfg
        edges = cfg.bin_edges
        pred = np.asarray(pred, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        assert pred.shape == truth.shape and pred.ndim == 3, "expect (Nx,Ny,Nz) cubes"

        d, sq, grad = self._fields_for_cone(pred, truth)   # all raveled, same length
        truth_f, pred_f = truth.ravel(), pred.ravel()
        valid = np.isfinite(d)
        if not valid.any():
            return  # cone has no fronts in the considered slices
        d, sq, grad = d[valid], sq[valid], grad[valid]
        truth_v, pred_v = truth_f[valid], pred_f[valid]

        # pooled binned sums
        idx = np.digitize(d, edges) - 1
        inb = (idx >= 0) & (idx < len(edges) - 1)
        i, sqi, gri = idx[inb], sq[inb], grad[inb]
        ti = truth_v[inb]
        pi = pred_v[inb]
        np.add.at(self.n, i, 1.0)
        np.add.at(self.sum_sq, i, sqi)
        np.add.at(self.sum_sq2, i, sqi * sqi)
        np.add.at(self.sum_grad, i, gri)
        np.add.at(self.sum_truth, i, ti)
        np.add.at(self.sum_pred, i, pi)
        self.total_sq_err += float(sq.sum())

        # per-cone band metrics (for bootstrap)
        rec = {"cone_id": cone_id, "n_front_vox": int(valid.sum())}
        for w in cfg.bands_mpc:
            m = np.abs(d) <= w
            rec[f"L2_band{w:g}"] = float(np.sqrt(sq[m].mean())) if m.any() else np.nan
            rec[f"H1_band{w:g}"] = float(np.sqrt(grad[m].mean())) if m.any() else np.nan
            rec[f"errfrac_band{w:g}"] = float(sq[m].sum() / sq.sum()) if sq.sum() > 0 else np.nan
        self.per_cone.append(rec)

    def _fields_for_cone(self, pred, truth):
        """Return flattened (signed_distance, sq_err, grad_sq_err) over contributing voxels."""
        cfg = self.cfg
        mask = truth > cfg.threshold
        sq_err = (pred - truth) ** 2

        if cfg.mode == "slice":
            d = np.full(truth.shape, np.nan)
            grad = np.zeros(truth.shape)
            zsel = self._z_indices(truth.shape[2])
            for k in zsel:
                d[:, :, k] = _signed_distance_2d(mask[:, :, k], (cfg.dx_mpc, cfg.dy_mpc))
                grad[:, :, k] = _grad_sqdiff_2d(pred[:, :, k], truth[:, :, k],
                                                cfg.dx_mpc, cfg.dy_mpc)
            return d.ravel(), sq_err.ravel(), grad.ravel()

        elif cfg.mode == "3d":
            sampling = (cfg.dx_mpc, cfg.dy_mpc, cfg.los_cell_mpc)
            if mask.all() or (~mask).all():
                d = np.full(truth.shape, np.nan)
            else:
                d = (distance_transform_edt(mask, sampling=sampling)
                     - distance_transform_edt(~mask, sampling=sampling))
            gx, gy, gz = np.gradient(pred - truth, cfg.dx_mpc, cfg.dy_mpc, cfg.los_cell_mpc)
            grad = gx * gx + gy * gy + gz * gz
            # zero out non-windowed slices
            if cfg.z_window is not None:
                keep = np.zeros(truth.shape[2], dtype=bool)
                keep[self._z_indices(truth.shape[2])] = True
                d[:, :, ~keep] = np.nan
            return d.ravel(), sq_err.ravel(), grad.ravel()
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

    def _z_indices(self, nz: int) -> np.ndarray:
        if self.cfg.z_window is None or self._z_grid is None:
            return np.arange(nz)
        lo, hi = self.cfg.z_window
        return np.where((self._z_grid >= lo) & (self._z_grid <= hi))[0]

    _z_grid: np.ndarray | None = None

    def set_z_grid(self, z_grid: np.ndarray | None):
        self._z_grid = None if z_grid is None else np.asarray(z_grid, dtype=float)

    # -- reduction -------------------------------------------------------- #
    def profile(self) -> dict:
        """Distance-binned mean error, gradient error, and front profile."""
        c = 0.5 * (self.cfg.bin_edges[:-1] + self.cfg.bin_edges[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sq = self.sum_sq / self.n
            var_sq = self.sum_sq2 / self.n - mean_sq ** 2
            mean_grad = self.sum_grad / self.n
            mean_truth = self.sum_truth / self.n
            mean_pred = self.sum_pred / self.n
        return {
            "d_mpc": c,
            "count": self.n,
            "rmse": np.sqrt(mean_sq),
            "rmse_std": np.sqrt(np.clip(var_sq, 0, None)),
            "grad_rms": np.sqrt(mean_grad),
            "mean_truth": mean_truth,
            "mean_pred": mean_pred,
        }

    def band_summary(self) -> dict:
        """Mean over cones of each band metric (the headline numbers)."""
        out = {}
        keys = [k for k in self.per_cone[0] if k not in ("cone_id", "n_front_vox")] \
            if self.per_cone else []
        for k in keys:
            vals = np.array([r[k] for r in self.per_cone], float)
            out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        return out


def front_width(d_mpc: np.ndarray, mean_field: np.ndarray,
                lo: float = 0.1, hi: float = 0.9) -> float:
    """10-90 transition width (Mpc) of a mean x_HI-vs-signed-distance curve.

    The curve rises from ~0 on the ionized side (d<0) to ~1 on the neutral side
    (d>0).  Returns the distance between the ``lo`` and ``hi`` crossings.  A
    blurrier (worse) front gives a larger width.
    """
    m = np.isfinite(mean_field)
    d, f = d_mpc[m], mean_field[m]
    order = np.argsort(d)
    d, f = d[order], f[order]
    if f.min() > lo or f.max() < hi:
        return np.nan

    def crossing(level):
        idx = np.where(np.diff(np.sign(f - level)) != 0)[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        f0, f1, d0, d1 = f[i], f[i + 1], d[i], d[i + 1]
        return d0 + (level - f0) * (d1 - d0) / (f1 - f0) if f1 != f0 else d0

    return crossing(hi) - crossing(lo)


def saturation_calibration(pred: np.ndarray, truth: np.ndarray,
                           tol: float = 1e-3) -> dict:
    """Where the truth is saturated (==0 or ==1), how confident is the model?"""
    neutral = truth >= 1 - tol
    ionized = truth <= tol
    return {
        "pred_where_truth_neutral_mean": float(pred[neutral].mean()) if neutral.any() else np.nan,
        "pred_where_truth_ionized_mean": float(pred[ionized].mean()) if ionized.any() else np.nan,
        "frac_neutral": float(neutral.mean()),
        "frac_ionized": float(ionized.mean()),
    }


def paired_bootstrap(values_a: np.ndarray, values_b: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over cones of (a - b).  Negative mean => a is better.

    ``values_a``/``values_b`` are per-cone band metrics for two models on the
    SAME cones (paired by index).  Returns mean difference and 95% CI.
    """
    a = np.asarray(values_a, float)
    b = np.asarray(values_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(n_boot)])
    return {
        "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "ci95": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "p_a_better": float((means < 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_overlay(results: dict[str, dict], cfg: BandConfig, out_path: Path):
    """results: {model_name: {"profile": profile_dict, "truth_profile": mean_truth}}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # (1) RMSE vs signed distance
    ax = axes[0]
    for i, (name, r) in enumerate(results.items()):
        p = r["profile"]
        ax.plot(p["d_mpc"], p["rmse"], label=name, color=colors[i % len(colors)])
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("signed distance to true front  [Mpc]\n(- ionized   +  neutral)")
    ax.set_ylabel("RMSE per voxel")
    ax.set_title("Error localised at the front")
    ax.legend(fontsize=8)

    # (2) mean x_HI vs signed distance: front sharpness
    ax = axes[1]
    any_truth = next(iter(results.values()))["profile"]
    ax.plot(any_truth["d_mpc"], any_truth["mean_truth"], "k-", lw=2.2, label="truth")
    for i, (name, r) in enumerate(results.items()):
        p = r["profile"]
        w = front_width(p["d_mpc"], p["mean_pred"])
        ax.plot(p["d_mpc"], p["mean_pred"], color=colors[i % len(colors)],
                label=f"{name} (10-90 = {w:.1f} Mpc)")
    wt = front_width(any_truth["d_mpc"], any_truth["mean_truth"])
    ax.axhline(0.1, color="grey", lw=0.6, ls="--"); ax.axhline(0.9, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("signed distance to true front  [Mpc]")
    ax.set_ylabel(r"$\langle x_{\rm HI}\rangle$")
    ax.set_title(f"Front sharpness (truth 10-90 = {wt:.1f} Mpc)")
    ax.legend(fontsize=8)

    # (3) gradient (H1 seminorm) RMS vs signed distance
    ax = axes[2]
    for i, (name, r) in enumerate(results.items()):
        p = r["profile"]
        ax.plot(p["d_mpc"], p["grad_rms"], color=colors[i % len(colors)], label=name)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("signed distance to true front  [Mpc]")
    ax.set_ylabel(r"RMS $|\nabla(\hat{x}-x)|$")
    ax.set_title("Gradient error (where the H1 win lives)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_csv(results: dict[str, dict], accums: dict[str, BoundaryBandAccumulator],
              out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, acc in accums.items():
        bs = acc.band_summary()
        prof = results[name]["profile"]
        row = {"model": name,
               "front_width_pred_mpc": round(front_width(prof["d_mpc"], prof["mean_pred"]), 3),
               "front_width_truth_mpc": round(front_width(prof["d_mpc"], prof["mean_truth"]), 3),
               "n_cones": len(acc.per_cone)}
        for k, (mean, std) in bs.items():
            row[k] = round(mean, 5)
            row[k + "_std"] = round(std, 5)
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    keys = ["model"] + [k for k in keys if k != "model"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_from_cubes(cube_source: dict[str, Callable[[], "iter"]],
                   cfg: BandConfig, out_dir: Path,
                   z_grid: np.ndarray | None = None,
                   reference: str | None = None) -> dict:
    """cube_source: {model_name: callable -> iterator of (cone_id, pred, truth)}."""
    accums: dict[str, BoundaryBandAccumulator] = {}
    results: dict[str, dict] = {}
    for name, source in cube_source.items():
        acc = BoundaryBandAccumulator(cfg)
        acc.set_z_grid(z_grid)
        n = 0
        for cone_id, pred, truth in source():
            acc.add_cone(cone_id, pred, truth)
            n += 1
        print(f"[{name}] accumulated {n} cones ({len(acc.per_cone)} with fronts)")
        accums[name] = acc
        results[name] = {"profile": acc.profile()}

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(results, cfg, out_dir / "boundary_band_overlay.png")
    write_csv(results, accums, out_dir / "boundary_band_metrics.csv")

    # paired bootstrap vs reference model
    if reference and reference in accums:
        ref = accums[reference]
        lines = [f"# paired bootstrap of band metrics vs reference '{reference}'",
                 "# negative mean_diff => this model BETTER than reference\n"]
        ref_by_cone = {r["cone_id"]: r for r in ref.per_cone}
        for name, acc in accums.items():
            if name == reference:
                continue
            for w in cfg.bands_mpc:
                key = f"H1_band{w:g}"
                pairs = [(r[key], ref_by_cone[r["cone_id"]][key])
                         for r in acc.per_cone if r["cone_id"] in ref_by_cone]
                if not pairs:
                    continue
                a, b = zip(*pairs)
                bs = paired_bootstrap(np.array(a), np.array(b))
                lines.append(f"{name} vs {reference}  {key}: "
                             f"mean_diff={bs['mean_diff']:+.4f} "
                             f"CI95=({bs['ci95'][0]:+.4f},{bs['ci95'][1]:+.4f}) "
                             f"P({name} better)={bs['p_a_better']:.3f}  n={bs['n_pairs']}")
        (out_dir / "bootstrap_vs_reference.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    print(f"\nWrote: {out_dir/'boundary_band_overlay.png'}")
    print(f"Wrote: {out_dir/'boundary_band_metrics.csv'}")
    return results


def _manifest_source(model_cones: list[dict]) -> Callable:
    def gen():
        for entry in model_cones:
            data = np.load(entry["npz"])
            yield entry.get("cone_id", entry["npz"]), data["pred"], data["truth"]
    return gen


def run_from_manifest(manifest_path: Path, cfg: BandConfig, out_dir: Path,
                      reference: str | None = None):
    spec = json.loads(Path(manifest_path).read_text())
    z_grid = np.array(spec["z_grid"]) if "z_grid" in spec else None
    sources = {name: _manifest_source(cones)
               for name, cones in spec["models"].items()}
    return run_from_cubes(sources, cfg, out_dir, z_grid=z_grid,
                          reference=reference or spec.get("reference"))


def run_from_checkpoints(checkpoints: dict[str, str], cfg: BandConfig, out_dir: Path,
                         n_cones: int, split: str, save_cubes: Path | None,
                         reference: str | None):
    """Cluster driver: build cubes from checkpoints via the project pipeline.

    Imports torch + the v3 viz/dataset modules lazily so the engine stays
    importable without them.  Run this on the cluster where CUBES_CACHE is set.
    """
    import torch  # noqa: F401
    from dataset.dataset_3d import (
        InputFeatures,
        LightconeCubeCache,
        ParameterNormalization,
        resolve_split,
    )
    from modeling import ModelConfig
    from viz.visualize_3d import load_model, predict_cube
    from util.run_metadata import load_run_metadata

    cache = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))
    first_checkpoint = Path(next(iter(checkpoints.values())))
    first_meta = load_run_metadata(first_checkpoint.parent)
    input_features = InputFeatures(
        first_meta["input_features"]["name"]
        if first_meta and "input_features" in first_meta
        else "density_z_params"
    )
    dataset = LightconeCubeCache(cache, input_features=input_features)
    if first_meta and first_meta.get("parameter_normalization"):
        dataset.set_parameter_normalization(
            ParameterNormalization.from_dict(
                first_meta["parameter_normalization"]
            )
        )
    z_grid = np.asarray(dataset.target_z, dtype=float)

    # resolve split rows once (same seed/order for every model)
    meta = None
    try:
        meta = first_meta
    except Exception:
        pass
    train_idx, val_idx, test_idx, _ = resolve_split(dataset, meta)
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    rows = rows[:n_cones]
    cone_ids = [int(dataset.cone_ids[r]) for r in rows]
    print(f"[checkpoints] {split} split: using {len(rows)} cones")

    def make_source(ckpt_path: str, name: str):
        def gen():
            checkpoint = Path(ckpt_path)
            metadata = load_run_metadata(checkpoint.parent)
            config = (
                ModelConfig.from_dict(metadata["model_config"])
                if metadata and "model_config" in metadata
                else None
            )
            if metadata and "input_features" in metadata:
                expected = metadata["input_features"]["name"]
                if expected != input_features.name:
                    raise ValueError(
                        f"checkpoint {checkpoint} expects input features "
                        f"{expected!r}, but comparison dataset uses "
                        f"{input_features.name!r}"
                    )
            model = load_model(
                in_channels=dataset.in_channels,
                checkpoint=checkpoint,
                model_config=config,
            )
            for r, cid in zip(rows, cone_ids):
                _dens, truth, pred = predict_cube(model, dataset[r])
                if save_cubes is not None:
                    save_cubes.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(save_cubes / f"{name}_cone{cid}.npz",
                                        truth=truth.astype(np.float32),
                                        pred=pred.astype(np.float32))
                yield cid, pred, truth
        return gen

    sources = {name: make_source(path, name) for name, path in checkpoints.items()}
    return run_from_cubes(sources, cfg, out_dir, z_grid=z_grid, reference=reference)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Synthetic verification with a known front and a known blur."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    nx = ny = 140
    nz = 40
    dx = DEFAULT_DX  # ~1.43 Mpc

    # Truth: a sphere of ionized gas (x_HI=0) growing along z inside neutral (x_HI=1).
    yy, xx = np.mgrid[0:nx, 0:ny].astype(float)
    truth = np.ones((nx, ny, nz), dtype=np.float32)
    for k in range(nz):
        radius = 10 + 1.2 * k  # grows with z-index -> fronts at varied positions
        cx, cy = nx / 2, ny / 2
        ionized = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        truth[:, :, k][ionized] = 0.0

    # Prediction: blur the front by a known sigma (in voxels) -> wider wall.
    blur_sigma_vox = 2.5
    pred = np.empty_like(truth)
    for k in range(nz):
        pred[:, :, k] = gaussian_filter(truth[:, :, k], blur_sigma_vox)
    pred = np.clip(pred + 0.01 * rng.standard_normal(pred.shape), 0, 1).astype(np.float32)

    cfg = BandConfig(mode="slice", dx_mpc=dx, dy_mpc=dx, d_max_mpc=25, bin_mpc=dx)
    acc = BoundaryBandAccumulator(cfg)
    acc.add_cone(0, pred, truth)
    prof = acc.profile()

    # Checks
    ok = True
    # 1) error peaks near the front (|d| small), not in saturated regions
    near = np.abs(prof["d_mpc"]) <= 3
    far = np.abs(prof["d_mpc"]) >= 15
    peak_near = np.nanmax(prof["rmse"][near])
    peak_far = np.nanmax(prof["rmse"][far])
    c1 = peak_near > 5 * peak_far
    ok &= c1
    print(f"[1] error concentrated at front: near_peak={peak_near:.3f} "
          f">> far_peak={peak_far:.4f}  -> {'PASS' if c1 else 'FAIL'}")

    # 2) predicted front is wider than truth, by ~ the blur scale
    wt = front_width(prof["d_mpc"], prof["mean_truth"])
    wp = front_width(prof["d_mpc"], prof["mean_pred"])
    c2 = np.isfinite(wt) and np.isfinite(wp) and wp > wt
    ok &= c2
    print(f"[2] predicted front wider than truth: truth={wt:.2f} Mpc, "
          f"pred={wp:.2f} Mpc  -> {'PASS' if c2 else 'FAIL'}")

    # 3) predicted width is in the right ballpark for the blur (~ +/- a few Mpc)
    expected_extra = 2 * blur_sigma_vox * dx * 0.8  # rough 10-90 broadening scale
    c3 = 0.3 * expected_extra < (wp - wt) < 4 * expected_extra
    ok &= c3
    print(f"[3] broadening ~ blur scale: dW={wp-wt:.2f} Mpc, "
          f"expected ~{expected_extra:.2f} Mpc  -> {'PASS' if c3 else 'FAIL'}")

    # 4) band metrics + error fraction sane
    bs = acc.band_summary()
    frac2 = bs[f"errfrac_band2"][0]
    c4 = 0.0 < frac2 <= 1.0 and np.isfinite(bs["H1_band2"][0])
    ok &= c4
    print(f"[4] band metrics finite; errfrac(|d|<=2Mpc)={frac2:.2f}  "
          f"-> {'PASS' if c4 else 'FAIL'}")

    # 5) saturation calibration: blurred sphere edge bleeds, interiors near correct
    cal = saturation_calibration(pred, truth)
    c5 = cal["pred_where_truth_neutral_mean"] > 0.8 and cal["pred_where_truth_ionized_mean"] < 0.2
    ok &= c5
    print(f"[5] saturation calibration: neutral->{cal['pred_where_truth_neutral_mean']:.2f}, "
          f"ionized->{cal['pred_where_truth_ionized_mean']:.2f}  -> {'PASS' if c5 else 'FAIL'}")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--selftest", action="store_true", help="run synthetic verification")
    src.add_argument("--manifest", type=Path, help="JSON manifest of saved npz cubes")
    src.add_argument("--checkpoints", nargs="+", metavar="name=path",
                     help="cluster mode: build cubes from checkpoints")
    ap.add_argument("--out", type=Path, default=Path("figures/band_out"))
    ap.add_argument("--mode", choices=["slice", "3d"], default="slice")
    ap.add_argument("--n-cones", type=int, default=200)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--save-cubes", type=Path, default=None)
    ap.add_argument("--reference", default="ufno",
                    help="model name to bootstrap others against")
    ap.add_argument("--dx-mpc", type=float, default=DEFAULT_DX)
    ap.add_argument("--d-max-mpc", type=float, default=30.0)
    ap.add_argument("--bin-mpc", type=float, default=1.5)
    ap.add_argument("--z-window", type=float, nargs=2, default=None,
                    metavar=("ZLO", "ZHI"), help="restrict to slices with z in [lo,hi]")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    cfg = BandConfig(dx_mpc=args.dx_mpc, dy_mpc=args.dx_mpc, mode=args.mode,
                     d_max_mpc=args.d_max_mpc, bin_mpc=args.bin_mpc,
                     z_window=tuple(args.z_window) if args.z_window else None)

    if args.manifest:
        run_from_manifest(args.manifest, cfg, args.out, reference=args.reference)
    else:
        run_from_checkpoints(_parse_kv(args.checkpoints), cfg, args.out,
                             n_cones=args.n_cones, split=args.split,
                             save_cubes=args.save_cubes, reference=args.reference)


if __name__ == "__main__":
    main()
