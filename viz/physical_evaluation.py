#!/usr/bin/env python3
"""Physical-observable evaluation of emulated x_HI lightcones.

Voxel MSE says how far a prediction is from the truth, not whether the
difference matters physically. This evaluator measures the prediction in the
quantities a reionization audience uses, and scales every error by a physical
reference:

1. **Reionization history.** Global x_HI(z), the Thomson optical depth tau_e,
   the x_HI = 0.25 / 0.5 / 0.75 crossing redshifts and the duration
   ``delta_z = z(0.75) - z(0.25)``. Errors are reported in units of the Planck
   2018 uncertainties (sigma_tau = 0.0073, sigma_zre = 0.73).
   Also the **sample-variance significance** of the transverse power error:
   per slice, ``(P_pred - P_true) / (P_true sqrt(2 / N_modes))``, i.e. the
   emulator bias in units of the Gaussian sample-variance error of a single
   200 cMpc slice. ``1 / significance^2`` is the number of independent slices
   an observer must average before the bias equals their statistical error.
2. **One-point statistics and the 21-cm observable.** Stage-resolved PDFs of
   x_HI (with the partially-ionized fraction 0.1 < x_HI < 0.9, which an
   MSE-blurred field inflates) and of dT_b; per-slice dT_b variance and
   skewness versus the global neutral fraction; and the transverse 21-cm power
   ``Delta^2_21(k_perp) = k^2 P_2D(k) / 2 pi`` in mK^2, per stage and at
   fixed k versus x_HI. dT_b uses the saturated-spin-temperature limit without
   velocities, with the *same* density for truth and prediction, so it
   isolates the emulator's x_HI error (see ``util.physical_stats``).
3. **EFT bias diagnostics.** With ``u = x_HI - <x_HI>`` per slice and the
   input density delta, the transfer function ``T(k) = P_u,delta / P_delta``,
   the stochastic term ``P_eps = P_uu - P_u,delta^2 / P_delta`` and the
   correlation ``r_u,delta``. Per cone and stage, ``T`` is fitted as
   ``b_1 + b_k2 k^2`` and ``P_eps`` as ``P_eps,0 + P_eps,2 k^2`` for
   ``k <= eft_k_max``; truth and prediction coefficients are compared cone by
   cone, the difference scaled by the truth fit's error. A conditional-mean
   (MSE-trained) emulator typically reproduces T(k) but under-predicts P_eps.

Lightcone caveats
-----------------
* All spectra are **2-D transverse spectra of individual LOS slices**; the
  LOS axis mixes geometry with evolution and the 256-cell grid discards most
  LOS structure (``experiments/multifield/pilot_report.md``). A thin-slice
  spectrum integrates over k_par, so ``T(k_perp)`` corresponds to 3-D
  wavenumbers of roughly 1-2 k_perp. The bias coefficients are therefore a
  like-for-like emulator statistic, not literal 3-D EFT measurements; delta
  is the evolved (Eulerian) density the network receives as input.
* Stage = the truth slice's mean x_HI, used for both fields (paired).
* Gaussian mode counting treats slices within a stage as independent, which
  under-estimates fit errors; compare truth/pred coefficient *differences*
  rather than reading the errors as absolute.
* Density-dependent diagnostics (2 and 3 beyond the x_HI PDF) are skipped for
  manifests whose npz files contain no ``density``.

Examples
--------
Cluster checkpoint comparison::

    python -m viz.physical_evaluation --checkpoints \
      ufno=checkpoints/3d_xhi/ufno/checkpoints_3d_ufno/best_model_state_dict.pt \
      --split test --n-cones 200 --out figures/3d_xhi/eval/physical_out

Offline saved-cube comparison (npz with pred, truth and density; manifest
with ``z_grid`` and optional per-cone ``omega_m``)::

    python -m viz.physical_evaluation --manifest cubes/manifest.json \
      --out figures/3d_xhi/eval/physical_out

Synthetic verification::

    python -m viz.physical_evaluation --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from util.physical_stats import (
    HUBBLE_H,
    OMEGA_B,
    OMEGA_M,
    PLANCK_TAU_SIGMA,
    PLANCK_ZRE_SIGMA,
    brightness_temperature_mk,
    fit_even_polynomial,
    histogram_w1,
    history_features,
    slice_moments,
    thomson_optical_depth,
)
from viz.power_spectrum_evaluation import KBinner


BOX_MPC = 200.0
HISTORY_KEYS = ("z_xhi25", "z_xhi50", "z_xhi75", "delta_z")


@dataclass
class PhysicalConfig:
    """Binning, fitting and cosmology configuration."""

    box_mpc: float = BOX_MPC
    n_k_bins: int = 15
    stage_edges: tuple[float, ...] = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)
    active_range: tuple[float, float] = (0.05, 0.95)
    min_stage_slices: int = 2
    xhi_range: tuple[float, float] = (-0.2, 1.2)
    xhi_bins: int = 70
    partial_range: tuple[float, float] = (0.1, 0.9)
    dtb_range_mk: tuple[float, float] = (-5.0, 100.0)
    dtb_bins: int = 105
    # "saturated": dT_b recomputed from x_HI and density (T_S >> T_CMB, no
    # velocities) -- isolates the x_HI error. "npz": a model's own predicted
    # dT_b and its target, read from the manifest npz (tb_pred / tb_truth).
    dtb_source: str = "saturated"
    xbar_bins: int = 20
    k_targets: tuple[float, ...] = (0.1, 0.2, 0.5)
    eft_k_max: float = 0.25
    eft_terms: int = 2
    omega_m: float = OMEGA_M      # fallback when a cone carries no OMm
    omega_b: float = OMEGA_B
    hubble_h: float = HUBBLE_H

    def __post_init__(self):
        if self.box_mpc <= 0 or self.n_k_bins < 2:
            raise ValueError("box_mpc must be positive and n_k_bins >= 2")
        if any(b <= a for a, b in zip(self.stage_edges[:-1], self.stage_edges[1:])):
            raise ValueError("stage_edges must be strictly increasing")
        if self.eft_terms < 1 or self.eft_k_max <= 0:
            raise ValueError("eft_terms must be >= 1 and eft_k_max positive")

    @property
    def n_stages(self) -> int:
        return len(self.stage_edges) - 1

    @property
    def n_rows(self) -> int:
        return self.n_stages + 1

    def stage_labels(self) -> list[str]:
        labels = [
            f"xbar_HI {lo:.2f}-{hi:.2f}"
            for lo, hi in zip(self.stage_edges[:-1], self.stage_edges[1:])
        ]
        labels.append(
            f"active {self.active_range[0]:.2f}-{self.active_range[1]:.2f}"
        )
        return labels

    @property
    def xhi_edges(self) -> np.ndarray:
        return np.linspace(*self.xhi_range, self.xhi_bins + 1)

    @property
    def dtb_edges(self) -> np.ndarray:
        return np.linspace(*self.dtb_range_mk, self.dtb_bins + 1)

    @property
    def xbar_edges(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.xbar_bins + 1)


def stage_masks(xbar: np.ndarray, cfg: PhysicalConfig) -> list[np.ndarray]:
    """Boolean LOS masks per stage plus the trailing "active" pseudo-stage."""
    index = np.digitize(xbar, cfg.stage_edges) - 1
    masks = [index == s for s in range(cfg.n_stages)]
    masks.append((xbar >= cfg.active_range[0]) & (xbar <= cfg.active_range[1]))
    return masks


def _demean(field: np.ndarray) -> np.ndarray:
    return field - field.mean(axis=(0, 1), keepdims=True)


class _quiet_ctx:
    """Silences all-NaN reductions, which are legitimate here."""

    def __enter__(self):
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


def _med_lo_hi(values: np.ndarray, axis: int = 0):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (np.nanmedian(values, axis=axis),
                np.nanpercentile(values, 16, axis=axis),
                np.nanpercentile(values, 84, axis=axis))


# --------------------------------------------------------------------------- #
# Accumulation
# --------------------------------------------------------------------------- #
class PhysicalAccumulator:
    """Reduces each cone to stage- and x_HI-binned statistics (a few kB)."""

    def __init__(self, cfg: PhysicalConfig, z_grid, transverse_shape):
        self.cfg = cfg
        self.z = np.asarray(z_grid, dtype=np.float64)
        self.binner = KBinner(*transverse_shape, cfg.box_mpc, cfg.n_k_bins)
        self.k = self.binner.centers
        self.k_target_index = np.array([
            int(np.argmin(np.abs(np.log(self.k) - np.log(target))))
            for target in cfg.k_targets
        ])
        self.records: list[dict] = []

    # -- per-slice helpers -------------------------------------------------- #
    def _power(self, field: np.ndarray) -> np.ndarray:
        """(Nz, n_k) per-slice binned sums of |FFT|^2 (demeaned per slice)."""
        return self.binner.binned_slice_sums(_demean(field))

    def _cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.binner.binned_cross_sums(_demean(a), _demean(b))

    def _mean_power(self, sums: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.binner.counts > 0, sums / self.binner.counts, np.nan)

    def _significance(self, truth_sums: np.ndarray, pred_sums: np.ndarray) -> np.ndarray:
        """Per-slice power bias in units of single-slice Gaussian sample variance.

        A real 2-D field has counts/2 independent modes per bin, so the
        fractional error of one slice's P(k) is sqrt(2 / counts).
        """
        counts = self.binner.counts
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = truth_sums * np.sqrt(2.0 / counts)
            out = (pred_sums - truth_sums) / sigma
        out[~(truth_sums > 0) | ~np.isfinite(out)] = np.nan
        return out

    def _stage_mean(self, per_slice: np.ndarray, masks) -> np.ndarray:
        rows = np.full((len(masks),) + per_slice.shape[1:], np.nan)
        with _quiet_ctx():
            for row, mask in enumerate(masks):
                if mask.sum() >= self.cfg.min_stage_slices:
                    rows[row] = np.nanmean(per_slice[mask], axis=0)
        return rows

    def _stage_sum(self, per_slice: np.ndarray, masks) -> np.ndarray:
        return np.stack([per_slice[mask].sum(axis=0) for mask in masks])

    def _stage_hist(self, field: np.ndarray, masks, edges) -> np.ndarray:
        rows = []
        for mask in masks:
            values = np.clip(field[:, :, mask].ravel(), edges[0], edges[-1])
            rows.append(np.histogram(values, bins=edges)[0])
        return np.stack(rows).astype(np.float64)

    def _xbar_mean(self, per_slice: np.ndarray, xbar: np.ndarray) -> np.ndarray:
        edges = self.cfg.xbar_edges
        index = np.clip(np.digitize(xbar, edges) - 1, 0, len(edges) - 2)
        rows = np.full((len(edges) - 1,) + per_slice.shape[1:], np.nan)
        with _quiet_ctx():
            for b in range(len(edges) - 1):
                select = index == b
                if select.any():
                    rows[b] = np.nanmean(per_slice[select], axis=0)
        return rows

    # -- main entry --------------------------------------------------------- #
    def add_cone(self, cone_id, pred, truth, density=None, omega_m=None,
                 dtb_pred=None, dtb_truth=None):
        cfg = self.cfg
        pred = np.asarray(pred, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if pred.shape != truth.shape or truth.ndim != 3:
            raise ValueError("pred and truth must be matching (Nx, Ny, Nz) cubes")
        if truth.shape[2] != self.z.size:
            raise ValueError("z_grid length must match the cube LOS dimension")
        if omega_m is None or not np.isfinite(omega_m):
            omega_m = cfg.omega_m
        omega_m = float(omega_m)

        xbar_t = truth.mean(axis=(0, 1))
        xbar_p = pred.mean(axis=(0, 1))
        masks = stage_masks(xbar_t, cfg)
        cosmo = {"omega_m": omega_m, "omega_b": cfg.omega_b, "h": cfg.hubble_h}

        rec = {
            "cone_id": cone_id,
            "omega_m": omega_m,
            "stage_n": np.array([int(m.sum()) for m in masks]),
            "xbar_truth": xbar_t,
            "xbar_pred": xbar_p,
            "tau_truth": thomson_optical_depth(self.z, xbar_t, **cosmo),
            "tau_pred": thomson_optical_depth(self.z, xbar_p, **cosmo),
            "history_truth": history_features(self.z, xbar_t),
            "history_pred": history_features(self.z, xbar_p),
        }

        # ---- x_HI one-point statistics and power significance ------------- #
        rec["xhi_hist_truth"] = self._stage_hist(truth, masks, cfg.xhi_edges)
        rec["xhi_hist_pred"] = self._stage_hist(pred, masks, cfg.xhi_edges)
        lo, hi = cfg.partial_range
        rec["partial_truth"] = np.array([
            np.mean((truth[:, :, m] > lo) & (truth[:, :, m] < hi)) if m.any() else np.nan
            for m in masks
        ])
        rec["partial_pred"] = np.array([
            np.mean((pred[:, :, m] > lo) & (pred[:, :, m] < hi)) if m.any() else np.nan
            for m in masks
        ])
        power_t = self._power(truth)
        power_p = self._power(pred)
        rec["sig_xhi"] = self._stage_mean(self._significance(power_t, power_p), masks)

        rec["has_density"] = density is not None
        if density is not None:
            delta = np.asarray(density, dtype=np.float64)
            if delta.shape != truth.shape:
                raise ValueError("density must match the x_HI cube shape")
            self._add_observable(rec, masks, xbar_t, truth, pred, delta,
                                 cosmo, power_t, power_p, dtb_pred, dtb_truth)
        self.records.append(rec)

    def _add_observable(self, rec, masks, xbar_t, truth, pred, delta, cosmo,
                        power_t, power_p, dtb_pred=None, dtb_truth=None):
        cfg = self.cfg
        if (dtb_pred is None) != (dtb_truth is None):
            raise ValueError("give both dtb_pred and dtb_truth, or neither")
        if dtb_pred is not None:
            dtb_t = np.asarray(dtb_truth, dtype=np.float64)
            dtb_p = np.asarray(dtb_pred, dtype=np.float64)
            if dtb_t.shape != truth.shape or dtb_p.shape != truth.shape:
                raise ValueError("dT_b cubes must match the x_HI cube shape")
        else:
            dtb_t = brightness_temperature_mk(truth, delta, self.z, **cosmo)
            dtb_p = brightness_temperature_mk(pred, delta, self.z, **cosmo)

        rec["dtb_hist_truth"] = self._stage_hist(dtb_t, masks, cfg.dtb_edges)
        rec["dtb_hist_pred"] = self._stage_hist(dtb_p, masks, cfg.dtb_edges)

        dtb_power_t = self._power(dtb_t)
        dtb_power_p = self._power(dtb_p)
        rec["sig_dtb"] = self._stage_mean(
            self._significance(dtb_power_t, dtb_power_p), masks)

        # Stage-mean transverse Delta^2_21(k) in mK^2.
        scale = self.k ** 2 / (2 * np.pi)
        n = np.maximum(rec["stage_n"], 1)[:, None]
        rec["d2_dtb_truth"] = scale * self._mean_power(
            self._stage_sum(dtb_power_t, masks)) / n
        rec["d2_dtb_pred"] = scale * self._mean_power(
            self._stage_sum(dtb_power_p, masks)) / n

        # Per-slice curves binned by the truth's global neutral fraction.
        moments_t = slice_moments(dtb_t)
        moments_p = slice_moments(dtb_p)
        k_index = self.k_target_index
        per_slice_d2_t = scale[k_index] * self._mean_power(dtb_power_t)[:, k_index]
        per_slice_d2_p = scale[k_index] * self._mean_power(dtb_power_p)[:, k_index]
        rec["xbar_var_truth"] = self._xbar_mean(moments_t["variance"], xbar_t)
        rec["xbar_var_pred"] = self._xbar_mean(moments_p["variance"], xbar_t)
        rec["xbar_skew_truth"] = self._xbar_mean(moments_t["skewness"], xbar_t)
        rec["xbar_skew_pred"] = self._xbar_mean(moments_p["skewness"], xbar_t)
        rec["xbar_d2_truth"] = self._xbar_mean(per_slice_d2_t, xbar_t)
        rec["xbar_d2_pred"] = self._xbar_mean(per_slice_d2_p, xbar_t)

        # EFT spectra: u = x_HI - <x_HI>, delta = input density (demeaned).
        rec["s_dd"] = self._stage_sum(self._power(delta), masks)
        rec["s_td"] = self._stage_sum(self._cross(truth, delta), masks)
        rec["s_pd"] = self._stage_sum(self._cross(pred, delta), masks)
        rec["s_tt"] = self._stage_sum(power_t, masks)
        rec["s_pp"] = self._stage_sum(power_p, masks)

    # -- reduction ---------------------------------------------------------- #
    def reduce(self) -> dict:
        cfg = self.cfg
        if not self.records:
            raise ValueError("no cones accumulated")
        recs = self.records
        k = self.k
        out = {
            "k_centers": k,
            "k_edges": self.binner.edges,
            "k_targets": k[self.k_target_index],
            "stage_labels": cfg.stage_labels(),
            "z_grid": self.z,
            "xhi_edges": cfg.xhi_edges,
            "dtb_edges": cfg.dtb_edges,
            "xbar_edges": cfg.xbar_edges,
            "n_cones": len(recs),
            "cone_ids": np.array([str(r["cone_id"]) for r in recs]),
            "omega_m": np.array([r["omega_m"] for r in recs]),
        }
        thin = np.stack([r["stage_n"] < cfg.min_stage_slices for r in recs])

        # ---- history ------------------------------------------------------ #
        out["tau_truth"] = np.array([r["tau_truth"] for r in recs])
        out["tau_pred"] = np.array([r["tau_pred"] for r in recs])
        for key in HISTORY_KEYS:
            out[f"{key}_truth"] = np.array([r["history_truth"][key] for r in recs])
            out[f"{key}_pred"] = np.array([r["history_pred"][key] for r in recs])
        dx = np.stack([r["xbar_pred"] - r["xbar_truth"] for r in recs])
        out["dxbar_med"], out["dxbar_lo"], out["dxbar_hi"] = _med_lo_hi(dx)

        # ---- x_HI one-point + significance -------------------------------- #
        self._reduce_pdf(out, "xhi", cfg.xhi_edges, thin)
        for which in ("truth", "pred"):
            values = np.stack([r[f"partial_{which}"] for r in recs])
            values[thin] = np.nan
            out[f"partial_{which}_all"] = values
            out[f"partial_{which}_med"] = _med_lo_hi(values)[0]
        sig = np.stack([r["sig_xhi"] for r in recs])
        out["sig_xhi_med"], out["sig_xhi_lo"], out["sig_xhi_hi"] = _med_lo_hi(sig)

        has_density = all(r["has_density"] for r in recs)
        if not has_density and any(r["has_density"] for r in recs):
            raise ValueError("density must be provided for all cones or none")
        out["has_density"] = has_density
        if has_density:
            self._reduce_observable(out, thin)
            self._reduce_eft(out, thin)
        return out

    def _reduce_pdf(self, out, name, edges, thin):
        hist_t = np.stack([r[f"{name}_hist_truth"] for r in self.records])
        hist_p = np.stack([r[f"{name}_hist_pred"] for r in self.records])
        widths = np.diff(edges)
        for which, hist in (("truth", hist_t), ("pred", hist_p)):
            pooled = hist.sum(axis=0)
            total = pooled.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{name}_pdf_{which}"] = pooled / (total * widths)
        w1 = np.array([
            [histogram_w1(hist_t[c, s], hist_p[c, s], edges)
             for s in range(hist_t.shape[1])]
            for c in range(hist_t.shape[0])
        ])
        w1[thin] = np.nan
        out[f"{name}_w1_all"] = w1
        out[f"{name}_w1_med"], out[f"{name}_w1_lo"], out[f"{name}_w1_hi"] = _med_lo_hi(w1)

    def _reduce_observable(self, out, thin):
        recs = self.records
        self._reduce_pdf(out, "dtb", self.cfg.dtb_edges, thin)
        sig = np.stack([r["sig_dtb"] for r in recs])
        out["sig_dtb_med"], out["sig_dtb_lo"], out["sig_dtb_hi"] = _med_lo_hi(sig)
        d2_t = np.stack([r["d2_dtb_truth"] for r in recs])
        d2_p = np.stack([r["d2_dtb_pred"] for r in recs])
        d2_t[thin], d2_p[thin] = np.nan, np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(d2_t > 0, d2_p / d2_t, np.nan)
        for key, values in (("d2_dtb_truth", d2_t), ("d2_dtb_pred", d2_p),
                            ("d2_dtb_ratio", ratio)):
            out[f"{key}_med"], out[f"{key}_lo"], out[f"{key}_hi"] = _med_lo_hi(values)
        for key in ("xbar_var", "xbar_skew", "xbar_d2"):
            for which in ("truth", "pred"):
                values = np.stack([r[f"{key}_{which}"] for r in recs])
                (out[f"{key}_{which}_med"], out[f"{key}_{which}_lo"],
                 out[f"{key}_{which}_hi"]) = _med_lo_hi(values)

    def _reduce_eft(self, out, thin):
        cfg = self.cfg
        recs = self.records
        k = self.k
        counts = self.binner.counts
        n_cones, n_rows, n_k = len(recs), cfg.n_rows, k.size
        curves = {name: np.full((n_cones, n_rows, n_k), np.nan)
                  for name in ("T_truth", "T_pred", "peps_truth", "peps_pred",
                               "r_truth", "r_pred")}
        fits = {name: np.full((n_cones, n_rows, cfg.eft_terms), np.nan)
                for name in ("bias_truth", "bias_pred", "bias_sigma_truth",
                             "stoch_truth", "stoch_pred", "stoch_sigma_truth")}

        with np.errstate(invalid="ignore", divide="ignore"):
            for c, rec in enumerate(recs):
                for s in range(n_rows):
                    if thin[c, s]:
                        continue
                    n_modes = counts * rec["stage_n"][s]
                    n_independent = n_modes / 2.0
                    s_dd = rec["s_dd"][s]
                    p_dd = np.where(n_modes > 0, s_dd / n_modes, np.nan)
                    for which, s_ud, s_uu in (("truth", rec["s_td"][s], rec["s_tt"][s]),
                                              ("pred", rec["s_pd"][s], rec["s_pp"][s])):
                        valid = (s_dd > 0) & (n_modes > 0)
                        transfer = np.where(valid, s_ud / s_dd, np.nan)
                        peps = np.where(valid, (s_uu - s_ud ** 2 / s_dd) / n_modes, np.nan)
                        peps = np.where(peps > 0, peps, np.nan)
                        r_ud = np.where(valid & (s_uu > 0),
                                        s_ud / np.sqrt(s_uu * s_dd), np.nan)
                        curves[f"T_{which}"][c, s] = transfer
                        curves[f"peps_{which}"][c, s] = peps
                        curves[f"r_{which}"][c, s] = r_ud

                        var_t = peps / (n_independent * p_dd)
                        bias = fit_even_polynomial(k, transfer, var_t,
                                                   cfg.eft_k_max, cfg.eft_terms)
                        var_p = 2.0 * peps ** 2 / n_independent
                        stoch = fit_even_polynomial(k, peps, var_p,
                                                    cfg.eft_k_max, cfg.eft_terms)
                        fits[f"bias_{which}"][c, s] = bias["coef"]
                        fits[f"stoch_{which}"][c, s] = stoch["coef"]
                        if which == "truth":
                            fits["bias_sigma_truth"][c, s] = bias["sigma"]
                            fits["stoch_sigma_truth"][c, s] = stoch["sigma"]

        for name, values in curves.items():
            out[f"eft_{name}_med"], out[f"eft_{name}_lo"], out[f"eft_{name}_hi"] = \
                _med_lo_hi(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["eft_peps_ratio_med"] = _med_lo_hi(
                curves["peps_pred"] / curves["peps_truth"])[0]
        for name, values in fits.items():
            out[f"eft_{name}"] = values
        with np.errstate(invalid="ignore", divide="ignore"):
            out["eft_bias_pull"] = (fits["bias_pred"] - fits["bias_truth"]) \
                / fits["bias_sigma_truth"]
            out["eft_stoch_pull"] = (fits["stoch_pred"] - fits["stoch_truth"]) \
                / fits["stoch_sigma_truth"]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _colors():
    return _plt().rcParams["axes.prop_cycle"].by_key()["color"]


def _stage_colors(n: int):
    cmap = _plt().get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _band(ax, x, med, lo, hi, color, label, **kwargs):
    ax.plot(x, med, color=color, label=label, **kwargs)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def _one_to_one(ax, *arrays):
    values = np.concatenate([np.ravel(a)[np.isfinite(np.ravel(a))] for a in arrays])
    if values.size:
        lo, hi = float(values.min()), float(values.max())
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=0.8)


def plot_history(results: dict[str, dict], out_path: Path):
    plt = _plt()
    colors = _colors()
    first = next(iter(results.values()))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    ax = axes[0]
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    for i, (name, res) in enumerate(results.items()):
        _band(ax, res["z_grid"], res["dxbar_med"], res["dxbar_lo"],
              res["dxbar_hi"], colors[i % len(colors)], name)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\bar{x}_{\rm HI}^{\rm pred} - \bar{x}_{\rm HI}^{\rm true}$")
    ax.set_title("Global history error (cone median, 16-84%)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axvspan(-1, 1, color="grey", alpha=0.18, lw=0, label=r"$\pm1\sigma_{\rm Planck}$")
    all_pulls = [(res["tau_pred"] - res["tau_truth"]) / PLANCK_TAU_SIGMA
                 for res in results.values()]
    finite = np.concatenate([p[np.isfinite(p)] for p in all_pulls])
    limit = max(1.5, float(np.max(np.abs(finite))) if finite.size else 1.5)
    bins = np.linspace(-limit, limit, 41)
    for i, (name, pull) in enumerate(zip(results, all_pulls)):
        ax.hist(pull[np.isfinite(pull)], bins=bins, histtype="step", lw=1.6,
                color=colors[i % len(colors)],
                label=f"{name} (med |Δτ| = {np.nanmedian(np.abs(pull)):.2f}σ)")
    ax.set_xlabel(r"$\Delta\tau_e / \sigma_{\rm Planck}$  ($\sigma = 0.0073$)")
    ax.set_ylabel("cones")
    ax.set_title("Thomson optical depth error")
    ax.legend(fontsize=8)

    for ax, key, label in ((axes[2], "z_xhi50", r"$z(\bar{x}_{\rm HI} = 0.5)$"),
                           (axes[3], "delta_z", r"$\Delta z = z_{0.75} - z_{0.25}$")):
        for i, (name, res) in enumerate(results.items()):
            error = np.nanmedian(np.abs(res[f"{key}_pred"] - res[f"{key}_truth"]))
            ax.scatter(res[f"{key}_truth"], res[f"{key}_pred"], s=10, alpha=0.7,
                       color=colors[i % len(colors)],
                       label=f"{name} (med |err| = {error:.3f})")
        _one_to_one(ax, first[f"{key}_truth"],
                    *[res[f"{key}_pred"] for res in results.values()])
        ax.set_xlabel(f"truth {label}")
        ax.set_ylabel(f"prediction {label}")
        ax.legend(fontsize=8)
    axes[2].set_title(rf"Midpoint ($\sigma_{{z_{{re}}}}^{{\rm Planck}}$ = {PLANCK_ZRE_SIGMA})")
    axes[3].set_title("Duration")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pdfs(results: dict[str, dict], out_path: Path):
    plt = _plt()
    colors = _colors()
    first = next(iter(results.values()))
    has_density = first["has_density"]
    labels = first["stage_labels"][:-1]
    n_rows = 2 if has_density else 1
    fig, axes = plt.subplots(n_rows, len(labels), figsize=(4.2 * len(labels), 3.6 * n_rows),
                             squeeze=False)
    specs = [("xhi", r"$x_{\rm HI}$")]
    if has_density:
        specs.append(("dtb", r"$\delta T_b$ [mK]"))
    for row, (name, xlabel) in enumerate(specs):
        edges = first[f"{name}_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        for s, stage in enumerate(labels):
            ax = axes[row][s]
            ax.step(centers, first[f"{name}_pdf_truth"][s], where="mid",
                    color="k", lw=2.0, label="truth")
            for i, (model, res) in enumerate(results.items()):
                w1 = res[f"{name}_w1_med"][s]
                ax.step(centers, res[f"{name}_pdf_pred"][s], where="mid",
                        color=colors[i % len(colors)], label=f"{model} (W1={w1:.3g})")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(stage, fontsize=9)
            ax.legend(fontsize=7)
        axes[row][0].set_ylabel("PDF (pooled over cones)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_observable(results: dict[str, dict], out_path: Path):
    plt = _plt()
    colors = _colors()
    first = next(iter(results.values()))
    k = first["k_centers"]
    xbar = 0.5 * (first["xbar_edges"][:-1] + first["xbar_edges"][1:])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0][0]
    ax.loglog(k, first["d2_dtb_truth_med"][-1], "k-", lw=2.2, label="truth")
    for i, (name, res) in enumerate(results.items()):
        ax.loglog(k, res["d2_dtb_pred_med"][-1], color=colors[i % len(colors)], label=name)
    ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\Delta^2_{21,\perp}(k)$ [mK$^2$]")
    ax.set_title("Transverse 21-cm power (active slices, cone median)")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    styles = ["-", "--", ":", "-."]
    for j, kt in enumerate(first["k_targets"]):
        ls = styles[j % len(styles)]
        ax.plot(xbar, first["xbar_d2_truth_med"][:, j], color="k", ls=ls, lw=2.0,
                label=f"truth k={kt:.2f}")
        for i, (name, res) in enumerate(results.items()):
            ax.plot(xbar, res["xbar_d2_pred_med"][:, j], color=colors[i % len(colors)],
                    ls=ls, label=f"{name} k={kt:.2f}" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\bar{x}_{\rm HI}$ (truth)")
    ax.set_ylabel(r"$\Delta^2_{21,\perp}$ [mK$^2$]")
    ax.set_title("21-cm power evolution at fixed k")
    ax.legend(fontsize=7)

    for ax, key, title in ((axes[1][0], "xbar_var", r"Var($\delta T_b$) [mK$^2$]"),
                           (axes[1][1], "xbar_skew", r"Skewness($\delta T_b$)")):
        _band(ax, xbar, first[f"{key}_truth_med"], first[f"{key}_truth_lo"],
              first[f"{key}_truth_hi"], "k", "truth", lw=2.0)
        for i, (name, res) in enumerate(results.items()):
            _band(ax, xbar, res[f"{key}_pred_med"], res[f"{key}_pred_lo"],
                  res[f"{key}_pred_hi"], colors[i % len(colors)], name)
        ax.set_xlabel(r"$\bar{x}_{\rm HI}$ (truth)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[1][1].axhline(0.0, color="k", lw=0.6, ls=":")

    for ax, key, title in ((axes[0][2], "sig_dtb", r"$\delta T_b$"),
                           (axes[1][2], "sig_xhi", r"$x_{\rm HI}$")):
        ax.axhspan(-1, 1, color="grey", alpha=0.18, lw=0)
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        for i, (name, res) in enumerate(results.items()):
            _band(ax, k, res[f"{key}_med"][-1], res[f"{key}_lo"][-1],
                  res[f"{key}_hi"][-1], colors[i % len(colors)], name)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
        ax.set_ylabel(r"$(P_{\rm pred}-P_{\rm true})\,/\,\sigma_{\rm SV,slice}$")
        ax.set_title(f"{title} power bias / single-slice sample variance (active)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_eft(results: dict[str, dict], out_path: Path):
    plt = _plt()
    colors = _colors()
    markers = ["o", "s", "^", "D", "v", "P"]
    first = next(iter(results.values()))
    k = first["k_centers"]
    k_max = first["eft_k_max"]
    labels = first["stage_labels"][:-1]
    stage_colors = _stage_colors(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    curve_specs = (
        (axes[0][0], "T", r"$T(k) = P_{u\delta}/P_{\delta\delta}$", "linear"),
        (axes[0][1], "peps", r"$P_\epsilon(k)$ [Mpc$^2$]", "log"),
        (axes[0][2], "r", r"$r_{u\delta}(k)$", "linear"),
    )
    for ax, key, title, yscale in curve_specs:
        _band(ax, k, first[f"eft_{key}_truth_med"][-1], first[f"eft_{key}_truth_lo"][-1],
              first[f"eft_{key}_truth_hi"][-1], "k", "truth", lw=2.0)
        for i, (name, res) in enumerate(results.items()):
            _band(ax, k, res[f"eft_{key}_pred_med"][-1], res[f"eft_{key}_pred_lo"][-1],
                  res[f"eft_{key}_pred_hi"][-1], colors[i % len(colors)], name)
        ax.axvline(k_max, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        if key == "peps":
            # A strongly smoothed prediction drives P_eps to float round-off;
            # keep the axis on the physically meaningful range.
            floor = first["eft_peps_truth_med"][-1]
            floor = floor[np.isfinite(floor) & (floor > 0)]
            if floor.size:
                ax.set_ylim(bottom=1e-3 * float(floor.min()))
        ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
        ax.set_title(f"{title} (active slices)")
        ax.legend(fontsize=8)

    scatter_specs = (
        (axes[1][0], "eft_bias", 0, r"$b_1$"),
        (axes[1][1], "eft_bias", 1, r"$b_{k^2}$ [Mpc$^2$]"),
        (axes[1][2], "eft_stoch", 0, r"$P_{\epsilon,0}$ [Mpc$^2$]"),
    )
    for ax, key, term, label in scatter_specs:
        if term >= first[f"{key}_truth"].shape[-1]:
            ax.set_visible(False)
            continue
        truths = []
        for i, (name, res) in enumerate(results.items()):
            for s, stage in enumerate(labels):
                t = res[f"{key}_truth"][:, s, term]
                p = res[f"{key}_pred"][:, s, term]
                truths.extend([t, p])
                ax.scatter(t, p, s=12, alpha=0.7, color=stage_colors[s],
                           marker=markers[i % len(markers)],
                           label=stage if i == 0 else None)
        _one_to_one(ax, *truths)
        if key == "eft_stoch":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"truth {label}")
        ax.set_ylabel(f"prediction {label}")
        ax.set_title(f"{label}, k ≤ {k_max:g} Mpc⁻¹ (marker = model)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _nanmedian(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _nanpercentile(values, q) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def write_history_csv(results: dict[str, dict], out_path: Path):
    fields = ["model", "cone_id", "omega_m", "tau_truth", "tau_pred",
              "dtau_over_planck"]
    for key in HISTORY_KEYS:
        fields += [f"{key}_truth", f"{key}_pred"]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, res in results.items():
            for c, cone_id in enumerate(res["cone_ids"]):
                row = {
                    "model": name, "cone_id": cone_id,
                    "omega_m": res["omega_m"][c],
                    "tau_truth": res["tau_truth"][c],
                    "tau_pred": res["tau_pred"][c],
                    "dtau_over_planck":
                        (res["tau_pred"][c] - res["tau_truth"][c]) / PLANCK_TAU_SIGMA,
                }
                for key in HISTORY_KEYS:
                    row[f"{key}_truth"] = res[f"{key}_truth"][c]
                    row[f"{key}_pred"] = res[f"{key}_pred"][c]
                writer.writerow(row)


def _coefficient_name(term: int) -> str:
    """Column name of the k^(2 term) transfer-function coefficient."""
    return "b1" if term == 0 else f"bk{2 * term}"


def write_stage_csv(results: dict[str, dict], out_path: Path):
    first = next(iter(results.values()))
    k_index = [int(np.argmin(np.abs(first["k_centers"] - kt))) for kt in first["k_targets"]]
    k_names = [f"k{kt:.2f}" for kt in first["k_targets"]]
    rows = []
    for name, res in results.items():
        for s, stage in enumerate(res["stage_labels"]):
            row = {
                "model": name, "stage": stage,
                "xhi_w1_med": res["xhi_w1_med"][s],
                "partial_truth_med": res["partial_truth_med"][s],
                "partial_pred_med": res["partial_pred_med"][s],
            }
            for j, kn in zip(k_index, k_names):
                row[f"sig_xhi_{kn}"] = res["sig_xhi_med"][s, j]
            if res["has_density"]:
                row["dtb_w1_med_mk"] = res["dtb_w1_med"][s]
                for j, kn in zip(k_index, k_names):
                    row[f"sig_dtb_{kn}"] = res["sig_dtb_med"][s, j]
                    row[f"d2_dtb_ratio_{kn}"] = res["d2_dtb_ratio_med"][s, j]
                for term in range(res["eft_bias_truth"].shape[-1]):
                    coef = _coefficient_name(term)
                    row[f"{coef}_truth_med"] = _nanmedian(res["eft_bias_truth"][:, s, term])
                    row[f"{coef}_pred_med"] = _nanmedian(res["eft_bias_pred"][:, s, term])
                    row[f"{coef}_abs_pull_med"] = _nanmedian(
                        np.abs(res["eft_bias_pull"][:, s, term]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    row["peps0_ratio_med"] = _nanmedian(
                        res["eft_stoch_pred"][:, s, 0] / res["eft_stoch_truth"][:, s, 0])
            rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def headline_summary(results: dict[str, dict]) -> dict:
    summary = {}
    for name, res in results.items():
        dtau = res["tau_pred"] - res["tau_truth"]
        entry = {
            "n_cones": int(res["n_cones"]),
            "tau_truth_median": _nanmedian(res["tau_truth"]),
            "abs_dtau_median": _nanmedian(np.abs(dtau)),
            "abs_dtau_over_planck_median": _nanmedian(np.abs(dtau)) / PLANCK_TAU_SIGMA,
            "abs_dtau_over_planck_p84": _nanpercentile(np.abs(dtau), 84) / PLANCK_TAU_SIGMA,
        }
        for key in HISTORY_KEYS:
            error = np.abs(res[f"{key}_pred"] - res[f"{key}_truth"])
            entry[f"abs_d{key}_median"] = _nanmedian(error)
        entry["abs_dz_xhi50_over_planck_median"] = (
            entry["abs_dz_xhi50_median"] / PLANCK_ZRE_SIGMA)
        entry["xhi_w1_active_median"] = float(res["xhi_w1_med"][-1])
        entry["partial_fraction_active_truth"] = float(res["partial_truth_med"][-1])
        entry["partial_fraction_active_pred"] = float(res["partial_pred_med"][-1])
        if res["has_density"]:
            entry["b1_abs_pull_active_median"] = _nanmedian(
                np.abs(res["eft_bias_pull"][:, -1, 0]))
            with np.errstate(invalid="ignore", divide="ignore"):
                entry["peps0_ratio_active_median"] = _nanmedian(
                    res["eft_stoch_pred"][:, -1, 0] / res["eft_stoch_truth"][:, -1, 0])
        summary[name] = entry
    return summary


def write_npz(results: dict[str, dict], out_path: Path):
    payload = {}
    for name, res in results.items():
        for key, value in res.items():
            if key == "stage_labels":
                value = np.asarray(value, dtype=str)
            payload[f"{name}/{key}"] = np.asarray(value)
    np.savez_compressed(out_path, **payload)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_from_cubes(sources: dict[str, Callable], cfg: PhysicalConfig, z_grid,
                   out_dir: Path) -> dict:
    """``sources[name]()`` yields (cone_id, pred, truth, density|None, omega_m|None)."""
    results = {}
    for name, source in sources.items():
        accumulator = None
        for item in source():
            cone_id, pred, truth, density, omega_m, *extra = item
            extra = extra[0] if extra else {}
            if accumulator is None:
                accumulator = PhysicalAccumulator(cfg, z_grid, truth.shape[:2])
            accumulator.add_cone(cone_id, pred, truth, density, omega_m, **extra)
        if accumulator is None:
            raise ValueError(f"model {name!r} produced no cones")
        print(f"[{name}] accumulated {len(accumulator.records)} cones")
        results[name] = accumulator.reduce()
        results[name]["eft_k_max"] = cfg.eft_k_max

    out_dir.mkdir(parents=True, exist_ok=True)
    written = ["physical_history.png", "physical_pdfs.png"]
    plot_history(results, out_dir / written[0])
    plot_pdfs(results, out_dir / written[1])
    if next(iter(results.values()))["has_density"]:
        written += ["physical_21cm.png", "physical_eft.png"]
        plot_observable(results, out_dir / written[2])
        plot_eft(results, out_dir / written[3])
    else:
        print("[physical] no density available: skipping dT_b and EFT diagnostics")
    write_history_csv(results, out_dir / "physical_history_per_cone.csv")
    write_stage_csv(results, out_dir / "physical_stage_metrics.csv")
    write_npz(results, out_dir / "physical_results.npz")
    summary = headline_summary(results)
    (out_dir / "physical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    config = {"method": "physical_observables", **asdict(cfg)}
    (out_dir / "physical_config.json").write_text(json.dumps(config, indent=2) + "\n")
    written += ["physical_history_per_cone.csv", "physical_stage_metrics.csv",
                "physical_results.npz", "physical_summary.json", "physical_config.json"]
    for filename in written:
        print(f"Wrote: {out_dir / filename}")
    return results


def _manifest_source(entries: list[dict], dtb_from_npz: bool = False) -> Callable:
    def generate():
        for entry in entries:
            with np.load(entry["npz"]) as data:
                pred = np.asarray(data["pred"])
                truth = np.asarray(data["truth"])
                density = np.asarray(data["density"]) if "density" in data else None
                extra = {}
                if dtb_from_npz:
                    if "tb_pred" not in data or "tb_truth" not in data:
                        raise ValueError(f"{entry['npz']} has no tb_pred/tb_truth for --dtb-from-npz")
                    extra = {"dtb_pred": np.asarray(data["tb_pred"]),
                             "dtb_truth": np.asarray(data["tb_truth"])}
            yield (entry.get("cone_id", entry["npz"]), pred, truth, density,
                   entry.get("omega_m"), extra)

    return generate


def run_from_manifest(manifest_path: Path, cfg: PhysicalConfig, out_dir: Path):
    dtb_from_npz = cfg.dtb_source == "npz"
    from viz.bubble_size_evaluation import _validate_manifest_pairing

    spec = json.loads(Path(manifest_path).read_text())
    if "z_grid" not in spec:
        raise ValueError("manifest needs a top-level z_grid for the history diagnostics")
    _validate_manifest_pairing(spec["models"])
    sources = {name: _manifest_source(entries, dtb_from_npz) for name, entries in spec["models"].items()}
    return run_from_cubes(sources, cfg, np.asarray(spec["z_grid"]), out_dir)


def run_from_checkpoints(checkpoints: dict[str, str], cfg: PhysicalConfig,
                         out_dir: Path, n_cones: int, split: str,
                         save_cubes: Path | None):
    import h5py
    import torch  # noqa: F401
    from dataset import paths
    from dataset.dataset_3d import (
        InputFeatures,
        LightconeCubeCache,
        ParameterNormalization,
        resolve_split,
    )
    from dataset.lightcone_params import PARAM_NAMES
    from modeling import ModelConfig
    from util.run_metadata import load_run_metadata
    from viz.visualize_3d import load_model, predict_cube

    cache = Path(os.environ.get("CUBES_CACHE", paths.CUBES))
    metadata_by_name = {
        name: load_run_metadata(Path(path).parent)
        for name, path in checkpoints.items()
    }
    missing = [name for name, metadata in metadata_by_name.items() if metadata is None]
    if missing:
        raise ValueError(f"run metadata is required for checkpoint models: {missing}")
    first_metadata = metadata_by_name[next(iter(checkpoints))]
    for name, metadata in metadata_by_name.items():
        for contract in ("input_features", "parameter_normalization", "split"):
            if metadata.get(contract) != first_metadata.get(contract):
                raise ValueError(
                    f"checkpoint model {name!r} has different {contract}; "
                    "paired evaluation requires identical preprocessing and split"
                )

    input_features = InputFeatures(first_metadata["input_features"]["name"])
    dataset = LightconeCubeCache(cache, input_features=input_features)
    if first_metadata.get("parameter_normalization"):
        dataset.set_parameter_normalization(
            ParameterNormalization.from_dict(first_metadata["parameter_normalization"])
        )
    train_idx, val_idx, test_idx, _ = resolve_split(dataset, first_metadata)
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split][:n_cones]
    cone_ids = [int(dataset.cone_ids[row]) for row in rows]
    omega_index = PARAM_NAMES.index("OMm")
    omega_m = [
        float(dataset.params[row, omega_index]) if dataset.params is not None else None
        for row in rows
    ]
    z_grid = np.asarray(dataset.target_z, dtype=np.float64)
    print(f"[checkpoints] {split} split: using {len(rows)} cones")

    def make_source(checkpoint_path: str, name: str):
        def generate():
            metadata = metadata_by_name[name]
            model = load_model(
                in_channels=dataset.in_channels,
                checkpoint=Path(checkpoint_path),
                model_config=ModelConfig.from_dict(metadata["model_config"]),
            )
            with h5py.File(cache, "r") as handle:
                for row, cone_id, om in zip(rows, cone_ids, omega_m):
                    # The raw cache density, not the scaled input channel, so
                    # this works for every InputFeatures variant.
                    density = np.asarray(handle["density"][row], dtype=np.float32)
                    _channel, truth, pred = predict_cube(model, dataset[row])
                    if save_cubes is not None:
                        save_cubes.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            save_cubes / f"{name}_cone{cone_id}.npz",
                            truth=truth.astype(np.float32),
                            pred=pred.astype(np.float32),
                            density=density,
                        )
                    yield cone_id, pred, truth, density, om

        return generate

    sources = {name: make_source(path, name) for name, path in checkpoints.items()}
    results = run_from_cubes(sources, cfg, z_grid, out_dir)
    if save_cubes is not None:
        manifest = {
            "z_grid": z_grid.tolist(),
            "models": {
                name: [
                    {"cone_id": cone_id,
                     "npz": str(save_cubes / f"{name}_cone{cone_id}.npz"),
                     "omega_m": om}
                    for cone_id, om in zip(cone_ids, omega_m)
                ]
                for name in checkpoints
            },
        }
        manifest_path = save_cubes / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote: {manifest_path}")
    return results


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def synthetic_cone(size: int = 48, n_z: int = 64, box_mpc: float = 200.0,
                   z_mid: float = 8.0, seed: int = 0):
    """Toy lightcone: Gaussian density and a thresholded, smoothed x_HI field.

    Cells whose smoothed density exceeds the (1 - x_target) quantile are
    ionized, so reionization is inside-out and x_HI anti-correlates with
    density, following a tanh global history centred on ``z_mid``.
    """
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * np.fft.fftfreq(size, d=box_mpc / size)
    kz = 2 * np.pi * np.fft.fftfreq(n_z, d=box_mpc / size)
    k2 = kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2
    amplitude = np.where(k2 > 0, (k2 + 0.01) ** -0.75, 0.0)
    noise = np.fft.fftn(rng.standard_normal((size, size, n_z)))
    delta = np.real(np.fft.ifftn(noise * amplitude))
    delta *= 0.5 / delta.std()
    smooth = np.real(np.fft.ifftn(np.fft.fftn(delta) * np.exp(-k2 * 4.0 ** 2 / 2)))
    z_grid = np.linspace(5.0, 25.0, n_z)
    target = 0.5 * (1 + np.tanh((z_grid - z_mid) / 1.0))
    xhi = np.empty_like(delta)
    for i in range(n_z):
        cut = np.quantile(smooth[:, :, i], target[i])
        xhi[:, :, i] = (smooth[:, :, i] < cut).astype(np.float64)
    return z_grid, delta, xhi


def _transverse_blur(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    kx = 2 * np.pi * np.fft.fftfreq(field.shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(field.shape[1])
    kernel = np.exp(-(kx[:, None] ** 2 + ky[None, :] ** 2) * sigma_cells ** 2 / 2)
    return np.real(np.fft.ifftn(np.fft.fftn(field, axes=(0, 1)) * kernel[:, :, None],
                                axes=(0, 1)))


def _selftest() -> int:
    cfg = PhysicalConfig(n_k_bins=10, xbar_bins=10)
    z_grid, delta, truth = synthetic_cone()

    def run(pred):
        accumulator = PhysicalAccumulator(cfg, z_grid, truth.shape[:2])
        accumulator.add_cone(0, pred, truth, delta)
        return accumulator.reduce()

    perfect = run(truth)
    blurred = run(_transverse_blur(truth, 1.5))
    delayed = run(np.clip(truth + 0.1, 0.0, 1.0))
    no_density = PhysicalAccumulator(cfg, z_grid, truth.shape[:2])
    no_density.add_cone(0, truth, truth)
    no_density = no_density.reduce()

    active = -1
    checks = {
        "tau near Planck for z_mid = 8": 0.045 < perfect["tau_truth"][0] < 0.07,
        "perfect prediction: zero tau error":
            abs(perfect["tau_pred"][0] - perfect["tau_truth"][0]) < 1e-12,
        "perfect prediction: zero PDF distance":
            np.nanmax(perfect["xhi_w1_all"]) < 1e-12,
        "perfect prediction: zero power bias":
            np.nanmax(np.abs(perfect["sig_dtb_med"])) < 1e-9,
        "perfect prediction: identical b1":
            np.allclose(perfect["eft_bias_pred"], perfect["eft_bias_truth"],
                        equal_nan=True),
        "inside-out truth has negative b1": perfect["eft_bias_truth"][0, active, 0] < 0,
        "more neutral prediction lowers tau":
            delayed["tau_pred"][0] < delayed["tau_truth"][0],
        "blur inflates the partially ionized fraction":
            blurred["partial_pred_med"][active] > blurred["partial_truth_med"][active] + 0.05,
        "blur removes small-scale 21-cm power":
            blurred["sig_dtb_med"][active, -1] < -1.0,
        "blur lowers the stochastic term":
            np.nanmedian(blurred["eft_peps_ratio_med"][active]) < 1.0,
        "density-free run skips EFT": not no_density["has_density"],
    }
    for label, passed in checks.items():
        print(f"[{label}] {'PASS' if passed else 'FAIL'}")
    passed = all(bool(v) for v in checks.values())
    print("SELFTEST:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_kv(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def _dtb_settings(args) -> dict:
    if args.dtb_from_npz and not args.manifest:
        raise SystemExit("--dtb-from-npz needs --manifest")
    lo, hi = args.dtb_range or ((-250.0, 100.0) if args.dtb_from_npz else (-5.0, 100.0))
    return {"dtb_source": "npz" if args.dtb_from_npz else "saturated",
            "dtb_range_mk": (float(lo), float(hi)), "dtb_bins": int(round(hi - lo))}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selftest", action="store_true")
    source.add_argument("--manifest", type=Path)
    source.add_argument("--checkpoints", nargs="+", metavar="name=path")
    parser.add_argument("--out", type=Path, default=Path("figures/3d_xhi/eval/physical_out"))
    parser.add_argument("--n-cones", type=int, default=200)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--save-cubes", type=Path, default=None)
    parser.add_argument("--box-mpc", type=float, default=BOX_MPC)
    parser.add_argument("--n-k-bins", type=int, default=15)
    parser.add_argument("--k-targets", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    parser.add_argument("--eft-k-max", type=float, default=0.25)
    parser.add_argument("--eft-terms", type=int, default=2)
    parser.add_argument("--dtb-from-npz", action="store_true",
                        help="manifest only: evaluate the model's own dT_b (npz tb_pred/tb_truth) "
                             "instead of recomputing it from x_HI in the saturated limit")
    parser.add_argument("--dtb-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="dT_b PDF range in mK (1 mK bins); default -5 100, or -250 100 "
                             "with --dtb-from-npz (absorption)")
    args = parser.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    cfg = PhysicalConfig(
        box_mpc=args.box_mpc,
        n_k_bins=args.n_k_bins,
        k_targets=tuple(args.k_targets),
        eft_k_max=args.eft_k_max,
        eft_terms=args.eft_terms,
        **_dtb_settings(args),
    )
    if args.manifest:
        run_from_manifest(args.manifest, cfg, args.out)
    else:
        run_from_checkpoints(
            _parse_kv(args.checkpoints), cfg, args.out,
            n_cones=args.n_cones, split=args.split, save_cubes=args.save_cubes,
        )


if __name__ == "__main__":
    main()
