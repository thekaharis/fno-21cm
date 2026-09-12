#!/usr/bin/env python3
"""Scale- and redshift-resolved power-spectrum evaluation for x_HI lightcones.

Motivation
----------
Whole-volume L2 says *how much* a model is wrong, not *where*.  This module
resolves the error per scale and per epoch, using the two standard emulator
statistics:

* ``ratio(k) = P_pred(k) / P_true(k)`` -- amplitude fidelity: does the model
  put the right amount of fluctuation power at each transverse scale?
* ``r(k) = P_cross(k) / sqrt(P_true P_pred)`` -- phase fidelity: are the
  bubbles in the right *places*?  ``1 - r^2`` is the fraction of predicted
  variance at that scale that is uncorrelated with the truth.

Design choices specific to this lightcone
-----------------------------------------
* Cubes are ``(Nx, Ny, Nz) = (140, 140, 256)``; the LOS axis is uniform in
  *redshift* and mixes geometry with cosmic evolution, so the field is NOT
  statistically stationary along it.  A single 3-D isotropic P(k) (as in
  ``util.metrics_21cm``) averages over that evolution.  Here every statistic
  is built from **2-D transverse spectra of individual LOS slices** and then
  aggregated two ways:

  - as **cylindrical maps** ``ratio(k_perp, k_par)`` / ``r(k_perp, k_par)``
    from Hann-windowed LOS chunks centred on selected redshifts (default
    z = 7, 9, 11) -- log-log wavenumber planes in the style of standard
    21-cm cylindrical power-spectrum comparisons.  The chunk's redshift
    sampling is converted to comoving Mpc with flat LCDM (``OMEGA_M``,
    ``H0_KM_S_MPC`` below), so ``k_par`` is physical and chunk-dependent;
  - by **reionization stage** (the slice's transverse-mean x_HI) -> curves
    that compare cones with different reionization timings fairly.

* Transverse axes are periodic (200 Mpc box), so plain FFTs need no window.
  k is physical: ``k = 2 pi f`` in 1/Mpc; bins are log-spaced from the
  fundamental (~0.031/Mpc) to the transverse Nyquist (~2.2/Mpc).
* Slices whose truth is essentially uniform (fully neutral / fully ionized)
  carry no fluctuation power and would make the ratio meaningless; stage
  binning excludes them by construction and slab cells are masked by a
  truth-variance floor.
* Aggregation across cones uses the median and the 16-84 percentile band,
  so the spread over the 11-parameter sample stays visible.

The core engine (numpy-only) is torch-free and unit-tested via ``--selftest``.
Torch / project imports are needed only for the ``--checkpoints`` driver.

Typical use
-----------
Cluster (predicts cubes from checkpoints, writes figures + CSV + NPZ)::

    python -m viz.power_spectrum_evaluation --checkpoints \
        ufno=checkpoints/3d_xhi/ufno/checkpoints_3d_ufno/best_model_state_dict.pt \
        localfno=checkpoints/3d_xhi/localfno/checkpoints_3d_localfno/best_model_state_dict.pt \
        --n-cones 200 --split test --out figures/3d_xhi/eval/ps_out/

Offline (re-use cubes saved by this script or boundary_band_diagnostic)::

    python -m viz.power_spectrum_evaluation --manifest ps_manifest.json --out figures/3d_xhi/eval/ps_out/

Self-test (no data needed)::

    python -m viz.power_spectrum_evaluation --selftest
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


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BOX_MPC = 200.0          # transverse comoving box size (Mpc)
N_TRANSVERSE = 140       # transverse cells


@dataclass
class SpectrumConfig:
    """Binning configuration for the transverse power-spectrum diagnostic."""
    box_mpc: float = BOX_MPC
    n_bins: int = 15                    # log-spaced k_perp bins fundamental..Nyquist
    # cylindrical (k_perp, k_par) maps: LOS chunks centred on these redshifts
    chunk_z_centers: tuple[float, ...] = (7.0, 9.0, 11.0)
    chunk_cells: int = 32               # LOS cells per chunk (Hann-windowed FFT)
    # slice stages by transverse-mean x_HI; outside [first, last] is excluded
    stage_edges: tuple[float, ...] = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)
    active_range: tuple[float, float] = (0.05, 0.95)  # headline "active" slices
    min_stage_slices: int = 2           # cone contributes to a stage only if
    min_truth_std: float = 1e-3         # mask chunks with ~uniform truth

    @property
    def n_stages(self) -> int:
        return len(self.stage_edges) - 1

    def stage_labels(self) -> list[str]:
        labels = [
            f"x̄_HI {lo:.2f}–{hi:.2f}"
            for lo, hi in zip(self.stage_edges[:-1], self.stage_edges[1:])
        ]
        labels.append(
            f"active {self.active_range[0]:.2f}–{self.active_range[1]:.2f}"
        )
        return labels


# --------------------------------------------------------------------------- #
# Core engine (torch-free)
# --------------------------------------------------------------------------- #
class KBinner:
    """Radial k-binning of a 2-D transverse FFT grid, in physical 1/Mpc.

    Precomputes a (n_valid_modes, n_bins) indicator matrix so binned sums are
    a single matmul per cone.  The k=0 mode and corner modes beyond the axis
    Nyquist are excluded.
    """

    def __init__(self, nx: int, ny: int, box_mpc: float, n_bins: int):
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=box_mpc / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=box_mpc / ny)
        k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).ravel()

        k_fund = 2 * np.pi / box_mpc
        k_ny = np.pi * min(nx, ny) / box_mpc
        edges = np.geomspace(k_fund, k_ny, n_bins + 1)
        edges[0] *= 1 - 1e-9   # keep the fundamental in bin 0
        edges[-1] *= 1 + 1e-9  # keep exact-Nyquist axis modes in the last bin

        idx = np.digitize(k, edges) - 1
        self.valid = (idx >= 0) & (idx < n_bins) & (k > 0)
        self.matrix = np.zeros((int(self.valid.sum()), n_bins))
        self.matrix[np.arange(self.matrix.shape[0]), idx[self.valid]] = 1.0
        self.counts = self.matrix.sum(axis=0)            # modes per bin
        self.edges = edges
        self.centers = np.sqrt(edges[:-1] * edges[1:])   # geometric centers
        # unnormalized |FFT|^2 -> P(k) in Mpc^2:  P = |F|^2 * L^2 / N^4
        self.power_norm = box_mpc ** 2 / float(nx * ny) ** 2

    def binned_slice_sums(self, field: np.ndarray) -> np.ndarray:
        """Per-slice binned sums of |FFT|^2 over transverse modes.

        ``field`` is (Nx, Ny, Nz) with per-slice mean already removed;
        returns (Nz, n_bins) sums of physical power P(k) [Mpc^2].
        """
        f = np.fft.fftn(field, axes=(0, 1))
        power = (np.abs(f) ** 2).reshape(-1, field.shape[2]) * self.power_norm
        return power[self.valid].T @ self.matrix

    def binned_cross_sums(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-slice binned sums of Re(A conj(B)), same layout as above."""
        fa = np.fft.fftn(a, axes=(0, 1))
        fb = np.fft.fftn(b, axes=(0, 1))
        cross = np.real(fa * np.conj(fb)).reshape(-1, a.shape[2]) \
            * self.power_norm
        return cross[self.valid].T @ self.matrix


# Flat LCDM used to convert the LOS redshift sampling into comoving distance
# (21cmFAST / Planck-18 defaults).  Only enters through k_par's physical units.
OMEGA_M = 0.3096
H0_KM_S_MPC = 67.66
C_KM_S = 299792.458


def los_cell_size_mpc(z: float, dz_cell: float) -> float:
    """Comoving size of one LOS cell at redshift z:  dD = c dz / H(z)."""
    e_z = np.sqrt(OMEGA_M * (1.0 + z) ** 3 + (1.0 - OMEGA_M))
    return C_KM_S / (H0_KM_S_MPC * e_z) * dz_cell


@dataclass(frozen=True)
class ZChunk:
    """One Hann-windowed LOS chunk for the cylindrical (k_perp, k_par) maps."""
    z_center: float
    start: int
    stop: int
    los_cell_mpc: float
    kpar_centers: np.ndarray     # discrete rfft wavenumbers, DC excluded
    kpar_edges: np.ndarray       # pcolormesh edges (all positive, log-safe)
    window: np.ndarray           # Hann taper (the chunk is not LOS-periodic)


def build_chunks(cfg: SpectrumConfig, z_grid: np.ndarray | None,
                 nz: int) -> list[ZChunk]:
    """LOS chunks centred on cfg.chunk_z_centers; out-of-range ones skipped."""
    if z_grid is None:
        print("[chunks] no z grid available; skipping cylindrical maps")
        return []
    z = np.asarray(z_grid, dtype=float)
    dz = float(z[1] - z[0])
    chunks = []
    for zc in cfg.chunk_z_centers:
        i = int(np.argmin(np.abs(z - zc)))
        start = i - cfg.chunk_cells // 2
        stop = start + cfg.chunk_cells
        if start < 0 or stop > nz:
            print(f"[chunks] z={zc:g}: chunk exceeds the LOS range; skipped")
            continue
        cell = los_cell_size_mpc(zc, dz)
        f = 2 * np.pi * np.fft.rfftfreq(cfg.chunk_cells, d=cell)
        df = float(f[1])
        edges = np.concatenate([[f[1] - df / 2], f[1:] + df / 2])
        chunks.append(ZChunk(z_center=float(zc), start=start, stop=stop,
                             los_cell_mpc=cell, kpar_centers=f[1:],
                             kpar_edges=edges,
                             window=np.hanning(cfg.chunk_cells)))
    return chunks


def _demean_slices(field: np.ndarray) -> np.ndarray:
    return field - field.mean(axis=(0, 1), keepdims=True)


def _safe_ratio_r(pt: np.ndarray, pp: np.ndarray, cross: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """ratio = pp/pt and r = cross/sqrt(pt*pp); NaN where truth power is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(pt > 0, pp / pt, np.nan)
        denom = np.sqrt(pt * pp)
        r = np.where(denom > 0, cross / denom, np.nan)
    return ratio, r


@dataclass
class PowerSpectrumAccumulator:
    """Aggregates transverse spectra over many cones, memory-safely.

    Per cone it stores only stage-/slab-reduced binned sums (a few kB), never
    raw voxels or per-mode spectra.
    """
    cfg: SpectrumConfig
    binner: KBinner
    chunks: list[ZChunk] = field(default_factory=list)
    per_cone: list[dict] = field(default_factory=list)

    def add_cone(self, cone_id, pred: np.ndarray, truth: np.ndarray):
        cfg = self.cfg
        pred = np.asarray(pred, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        assert pred.shape == truth.shape and pred.ndim == 3, \
            "expect (Nx,Ny,Nz) cubes"

        t = _demean_slices(truth)
        p = _demean_slices(pred)
        pt = self.binner.binned_slice_sums(t)        # (Nz, n_bins)
        pp = self.binner.binned_slice_sums(p)
        cross = self.binner.binned_cross_sums(t, p)

        xbar = truth.mean(axis=(0, 1))               # (Nz,) stage coordinate
        slice_std = truth.std(axis=(0, 1))
        nb = self.binner.centers.size

        # ---- stage reduction (last row = "active" pseudo-stage) ----------- #
        n_rows = cfg.n_stages + 1
        stage_pt = np.zeros((n_rows, nb))
        stage_pp = np.zeros((n_rows, nb))
        stage_cross = np.zeros((n_rows, nb))
        stage_n = np.zeros(n_rows, dtype=np.int64)

        stage_idx = np.digitize(xbar, cfg.stage_edges) - 1
        in_stage = (stage_idx >= 0) & (stage_idx < cfg.n_stages)
        for s in range(cfg.n_stages):
            sel = in_stage & (stage_idx == s)
            stage_pt[s] = pt[sel].sum(axis=0)
            stage_pp[s] = pp[sel].sum(axis=0)
            stage_cross[s] = cross[sel].sum(axis=0)
            stage_n[s] = int(sel.sum())
        active = (xbar >= cfg.active_range[0]) & (xbar <= cfg.active_range[1])
        stage_pt[-1] = pt[active].sum(axis=0)
        stage_pp[-1] = pp[active].sum(axis=0)
        stage_cross[-1] = cross[active].sum(axis=0)
        stage_n[-1] = int(active.sum())

        # ---- cylindrical (k_perp, k_par) chunks --------------------------- #
        cyl_pt, cyl_pp, cyl_cross, cyl_ok = [], [], [], []
        for ch in self.chunks:
            tc = truth[:, :, ch.start:ch.stop]
            pc = pred[:, :, ch.start:ch.stop]
            cyl_ok.append(
                bool(slice_std[ch.start:ch.stop].mean() > cfg.min_truth_std)
            )
            spt, spp, scr = self._cylindrical_sums(tc, pc, ch)
            cyl_pt.append(spt)
            cyl_pp.append(spp)
            cyl_cross.append(scr)

        self.per_cone.append({
            "cone_id": cone_id,
            "stage_pt": stage_pt, "stage_pp": stage_pp,
            "stage_cross": stage_cross, "stage_n": stage_n,
            "cyl_pt": cyl_pt, "cyl_pp": cyl_pp,
            "cyl_cross": cyl_cross, "cyl_ok": cyl_ok,
        })

    def _cylindrical_sums(self, truth_c: np.ndarray, pred_c: np.ndarray,
                          chunk: ZChunk) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
        """Hann-windowed 3-D FFT of one LOS chunk, binned to (k_par, k_perp).

        Returns (pt, pp, cross) of shape (n_kpar - 1, n_perp_bins); the
        k_par = 0 plane (pure transverse power) is dropped so both map axes
        are strictly positive wavenumbers.  Ratio and r need no physical
        normalization -- it cancels cell by cell.
        """
        w = chunk.window
        # rfft needs real input, so the LOS transform runs first
        tf = np.fft.fftn(
            np.fft.rfft(_demean_slices(truth_c) * w, axis=2), axes=(0, 1))
        pf = np.fft.fftn(
            np.fft.rfft(_demean_slices(pred_c) * w, axis=2), axes=(0, 1))
        b = self.binner

        def bin_perp(x: np.ndarray) -> np.ndarray:
            flat = x.reshape(-1, x.shape[2])[b.valid]      # (n_modes, n_kpar)
            return (b.matrix.T @ flat).T[1:]               # (n_kpar-1, n_perp)

        return (bin_perp(np.abs(tf) ** 2), bin_perp(np.abs(pf) ** 2),
                bin_perp(np.real(tf * np.conj(pf))))

    # -- reduction ---------------------------------------------------------- #
    def reduce(self) -> dict:
        """Median and 16-84 percentiles across cones of ratio(k), r(k).

        Returns stage-resolved curves, slab (k, z) maps, and the active-window
        truth/pred dimensionless power spectra Delta^2(k) = k^2 P(k) / 2 pi.
        """
        cfg = self.cfg
        if not self.per_cone:
            raise ValueError("no cones accumulated")
        k = self.binner.centers
        counts = self.binner.counts

        # stage curves: per-cone ratio/r, NaN out under-populated stages
        ratios, rs, d2t, d2p = [], [], [], []
        for rec in self.per_cone:
            ratio, r = _safe_ratio_r(
                rec["stage_pt"], rec["stage_pp"], rec["stage_cross"])
            thin = rec["stage_n"] < cfg.min_stage_slices
            ratio[thin], r[thin] = np.nan, np.nan
            ratios.append(ratio)
            rs.append(r)
            # active-window mean power per mode -> Delta^2(k)
            n_active = max(int(rec["stage_n"][-1]), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                pt_mean = rec["stage_pt"][-1] / (n_active * counts)
                pp_mean = rec["stage_pp"][-1] / (n_active * counts)
            scale = k ** 2 / (2 * np.pi)
            valid = rec["stage_n"][-1] >= cfg.min_stage_slices
            d2t.append(scale * pt_mean if valid else np.full_like(k, np.nan))
            d2p.append(scale * pp_mean if valid else np.full_like(k, np.nan))
        ratios = np.stack(ratios)     # (n_cones, n_stages+1, n_bins)
        rs = np.stack(rs)

        # cylindrical maps: per-cone ratio/r with uniform-truth chunks masked
        cyl_ratios, cyl_rs = [], []          # each: (n_cones, nkpar-1, n_perp)
        for c in range(len(self.chunks)):
            ratios_c, rs_c = [], []
            for rec in self.per_cone:
                ratio, r = _safe_ratio_r(
                    rec["cyl_pt"][c], rec["cyl_pp"][c], rec["cyl_cross"][c])
                if not rec["cyl_ok"][c]:
                    ratio[:], r[:] = np.nan, np.nan
                ratios_c.append(ratio)
                rs_c.append(r)
            cyl_ratios.append(np.stack(ratios_c))
            cyl_rs.append(np.stack(rs_c))

        def med_lo_hi(x):
            return (np.nanmedian(x, axis=0),
                    np.nanpercentile(x, 16, axis=0),
                    np.nanpercentile(x, 84, axis=0))

        # all-NaN cells (stages/slabs with no truth power anywhere) are
        # legitimate here; keep them NaN without the RuntimeWarning noise
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ratio_med, ratio_lo, ratio_hi = med_lo_hi(ratios)
            r_med, r_lo, r_hi = med_lo_hi(rs)
            cyl_ratio_med = (np.stack([np.nanmedian(x, axis=0)
                                       for x in cyl_ratios])
                             if self.chunks else np.zeros((0, 0, 0)))
            cyl_r_med = (np.stack([np.nanmedian(x, axis=0) for x in cyl_rs])
                         if self.chunks else np.zeros((0, 0, 0)))
            d2_truth_med = np.nanmedian(np.stack(d2t), axis=0)
            d2_pred_med = np.nanmedian(np.stack(d2p), axis=0)

        return {
            "k_centers": k,
            "k_edges": self.binner.edges,
            "stage_labels": cfg.stage_labels(),
            "stage_ratio_med": ratio_med, "stage_ratio_lo": ratio_lo,
            "stage_ratio_hi": ratio_hi,
            "stage_r_med": r_med, "stage_r_lo": r_lo, "stage_r_hi": r_hi,
            "cyl_ratio_med": cyl_ratio_med, "cyl_r_med": cyl_r_med,
            "cyl_z_centers": np.array([c.z_center for c in self.chunks]),
            "cyl_kpar_edges": (np.stack([c.kpar_edges for c in self.chunks])
                               if self.chunks else np.zeros((0, 0))),
            "n_cones": len(self.per_cone),
            "d2_truth_med": d2_truth_med, "d2_pred_med": d2_pred_med,
        }


def k_crossing(k: np.ndarray, curve: np.ndarray, level: float) -> float:
    """Smallest k where ``curve`` first drops below ``level`` (log-interp).

    NaN if the curve never drops below the level in the sampled range.
    """
    m = np.isfinite(curve)
    k, c = k[m], curve[m]
    if len(k) == 0:
        return np.nan
    below = np.flatnonzero(c < level)
    if below.size == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return float(k[0])
    f = (level - c[i - 1]) / (c[i] - c[i - 1])
    return float(np.exp(np.log(k[i - 1]) + f * (np.log(k[i]) - np.log(k[i - 1]))))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
K_BANDS = ((0.0, 0.3), (0.3, 1.0), (1.0, np.inf))   # 1/Mpc: large/mid/small
K_BAND_NAMES = ("large_k<0.3", "mid_k0.3-1", "small_k>1")


def plot_overlay(results: dict[str, dict], out_path: Path):
    """Headline figure: active-window Delta^2(k), ratio(k), r(k) per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    first = next(iter(results.values()))
    k = first["k_centers"]

    ax = axes[0]
    ax.loglog(k, first["d2_truth_med"], "k-", lw=2.2, label="truth")
    for i, (name, res) in enumerate(results.items()):
        ax.loglog(k, res["d2_pred_med"], color=colors[i % len(colors)],
                  label=name)
    ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\Delta^2_{x_{\rm HI}}(k_\perp)$")
    ax.set_title("Transverse power (active slices, cone median)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhspan(0.95, 1.05, color="grey", alpha=0.18, lw=0)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.plot(k, res["stage_ratio_med"][-1], color=c, label=name)
        ax.fill_between(k, res["stage_ratio_lo"][-1],
                        res["stage_ratio_hi"][-1], color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    # log y: high-k ratios can blow up by orders of magnitude where the truth
    # has ~no power; a linear axis would flatten the ±5% region into a line
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_{\rm pred}/P_{\rm true}$")
    ax.set_title("Power ratio (grey band: ±5%)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axhline(0.9, color="grey", lw=0.8, ls="--")
    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        k90 = k_crossing(k, res["stage_r_med"][-1], 0.9)
        label = name if not np.isfinite(k90) else f"{name} (r<0.9 at {k90:.2f})"
        ax.plot(k, res["stage_r_med"][-1], color=c, label=label)
        ax.fill_between(k, res["stage_r_lo"][-1], res["stage_r_hi"][-1],
                        color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
    ax.set_ylabel(r"$r(k_\perp)$")
    ax.set_title("Cross-correlation (phase fidelity)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_cylindrical_maps(results: dict[str, dict], out_ratio: Path,
                          out_r: Path) -> bool:
    """Cylindrical (k_perp, k_par) maps, one row per model, one column per
    redshift chunk; log-log wavenumber axes, grey = no truth power.

    Returns False (and writes nothing) when no chunks were available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    first = next(iter(results.values()))
    z_centers = first["cyl_z_centers"]
    n_chunks = len(z_centers)
    if n_chunks == 0:
        return False
    names = list(results)
    k_edges = first["k_edges"]

    ratio_cmap = plt.get_cmap("RdBu_r").copy()
    ratio_cmap.set_bad("0.85")
    r_cmap = plt.get_cmap("viridis").copy()
    r_cmap.set_bad("0.85")

    # Data-driven color limits: fixed ranges (ratio 0..2, r 0..1) rendered
    # near-uniform maps when the actual values sat within a few percent of
    # perfect. Ratio maps are shown as log2(P_pred/P_true) with a symmetric
    # robust limit; r maps get a robust lower limit. Small floors keep noise
    # from being amplified into fake structure when models are near-perfect.
    ratio_vals = np.concatenate([
        np.log2(res["cyl_ratio_med"][np.isfinite(res["cyl_ratio_med"])
                                     & (res["cyl_ratio_med"] > 0)])
        for res in results.values()
    ])
    ratio_lim = max(0.1, float(np.percentile(np.abs(ratio_vals), 98)))
    r_vals = np.concatenate([
        res["cyl_r_med"][np.isfinite(res["cyl_r_med"])]
        for res in results.values()
    ])
    r_lo = min(0.9, float(np.percentile(r_vals, 2)))

    specs = [
        ("cyl_ratio_med", out_ratio,
         r"$\log_2(P_{\rm pred}/P_{\rm true})$",
         {"cmap": ratio_cmap,
          "norm": TwoSlopeNorm(vmin=-ratio_lim, vcenter=0.0,
                               vmax=ratio_lim)},
         lambda a: np.log2(np.where(a > 0, a, np.nan))),
        ("cyl_r_med", out_r, r"$r(k_\perp, k_\parallel)$",
         {"cmap": r_cmap, "vmin": r_lo, "vmax": 1.0},
         None),
    ]

    for key, out_path, label, style, transform in specs:
        fig, axes = plt.subplots(len(names), n_chunks,
                                 figsize=(4.6 * n_chunks, 3.8 * len(names)),
                                 squeeze=False)
        for row, name in enumerate(names):
            res = results[name]
            for col in range(n_chunks):
                ax = axes[row][col]
                raw = res[key][col]
                if transform is not None:
                    raw = transform(raw)
                data = np.ma.masked_invalid(raw)
                pcm = ax.pcolormesh(k_edges, res["cyl_kpar_edges"][col],
                                    data, **style)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
                ax.set_ylabel(r"$k_\parallel$  [Mpc$^{-1}$]")
                ax.set_title(f"{name} — z ≈ {z_centers[col]:g}")
                fig.colorbar(pcm, ax=ax, pad=0.02, label=label)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return True


def plot_stage_curves(results: dict[str, dict], out_path: Path):
    """ratio(k) and r(k) per reionization stage, one row per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 2,
                             figsize=(12, 3.6 * len(names)), squeeze=False)
    first = next(iter(results.values()))
    k = first["k_centers"]
    stage_labels = first["stage_labels"][:-1]     # exclude "active"
    cmap = plt.get_cmap("viridis")

    for row, name in enumerate(names):
        res = results[name]
        for col, (key, ylabel, ref) in enumerate([
            ("stage_ratio_med", r"$P_{\rm pred}/P_{\rm true}$", 1.0),
            ("stage_r_med", r"$r(k_\perp)$", 0.9),
        ]):
            ax = axes[row][col]
            ax.axhline(ref, color="grey", lw=0.8, ls="--")
            for s, label in enumerate(stage_labels):
                ax.plot(k, res[key][s],
                        color=cmap(s / max(len(stage_labels) - 1, 1)),
                        label=label)
            ax.set_xscale("log")
            if col == 0:
                ax.set_yscale("log")   # ratio spans orders of magnitude
            else:
                ax.set_ylim(0, 1.05)
            ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name} by reionization stage")
            ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_csv(results: dict[str, dict], out_path: Path):
    """Per model x stage: k-band medians of |ratio-1| and 1-r, plus k(r<0.9)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, res in results.items():
        k = res["k_centers"]
        for s, stage in enumerate(res["stage_labels"]):
            row = {"model": name, "stage": stage, "n_cones": res["n_cones"],
                   "k_r_below_0.9": round(k_crossing(k, res["stage_r_med"][s],
                                                     0.9), 4)}
            for (lo, hi), band in zip(K_BANDS, K_BAND_NAMES):
                sel = (k >= lo) & (k < hi)
                ratio = res["stage_ratio_med"][s][sel]
                r = res["stage_r_med"][s][sel]
                with np.errstate(all="ignore"):
                    row[f"abs_ratio_err_{band}"] = round(
                        float(np.nanmedian(np.abs(ratio - 1.0))), 4)
                    row[f"one_minus_r_{band}"] = round(
                        float(np.nanmedian(1.0 - r)), 4)
            rows.append(row)
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_npz(results: dict[str, dict], out_path: Path):
    """Raw reduced arrays for later re-plotting (thesis figures)."""
    payload = {}
    for name, res in results.items():
        for key, val in res.items():
            if key == "stage_labels":
                val = np.array(val, dtype=object)
            payload[f"{name}/{key}"] = val
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_from_cubes(cube_source: dict[str, Callable[[], "iter"]],
                   cfg: SpectrumConfig, out_dir: Path,
                   z_grid: np.ndarray | None = None) -> dict:
    """cube_source: {model_name: callable -> iterator of (cone_id, pred, truth)}."""
    binner = None
    chunks: list[ZChunk] = []
    results: dict[str, dict] = {}
    for name, source in cube_source.items():
        acc = None
        n = 0
        for cone_id, pred, truth in source():
            if binner is None:
                binner = KBinner(truth.shape[0], truth.shape[1],
                                 cfg.box_mpc, cfg.n_bins)
                chunks = build_chunks(cfg, z_grid, truth.shape[2])
            if acc is None:
                acc = PowerSpectrumAccumulator(cfg, binner, chunks=chunks)
            acc.add_cone(cone_id, pred, truth)
            n += 1
        if acc is None:
            raise ValueError(f"model {name!r} produced no cones")
        print(f"[{name}] accumulated {n} cones")
        results[name] = acc.reduce()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(results, out_dir / "ps_overlay.png")
    wrote_cyl = plot_cylindrical_maps(results, out_dir / "ps_cyl_ratio.png",
                                      out_dir / "ps_cyl_r.png")
    plot_stage_curves(results, out_dir / "ps_stage_curves.png")
    write_csv(results, out_dir / "ps_metrics.csv")
    write_npz(results, out_dir / "ps_results.npz")
    files = ["ps_overlay.png"]
    if wrote_cyl:
        files += ["ps_cyl_ratio.png", "ps_cyl_r.png"]
    files += ["ps_stage_curves.png", "ps_metrics.csv", "ps_results.npz"]
    for f in files:
        print(f"Wrote: {out_dir / f}")
    return results


def _manifest_source(model_cones: list[dict]) -> Callable:
    def gen():
        for entry in model_cones:
            data = np.load(entry["npz"])
            yield entry.get("cone_id", entry["npz"]), data["pred"], data["truth"]
    return gen


def run_from_manifest(manifest_path: Path, cfg: SpectrumConfig, out_dir: Path):
    spec = json.loads(Path(manifest_path).read_text())
    z_grid = np.array(spec["z_grid"]) if "z_grid" in spec else None
    sources = {name: _manifest_source(cones)
               for name, cones in spec["models"].items()}
    return run_from_cubes(sources, cfg, out_dir, z_grid=z_grid)


def run_from_checkpoints(checkpoints: dict[str, str], cfg: SpectrumConfig,
                         out_dir: Path, n_cones: int, split: str,
                         save_cubes: Path | None):
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

    from dataset import paths
    cache = Path(os.environ.get("CUBES_CACHE", paths.CUBES))
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

    train_idx, val_idx, test_idx, _ = resolve_split(dataset, first_meta)
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
    return run_from_cubes(sources, cfg, out_dir, z_grid=z_grid)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Synthetic verification with known filters, noise, and wavenumbers."""
    rng = np.random.default_rng(0)
    nx = ny = 140
    nz = 48
    cfg = SpectrumConfig(n_bins=12, chunk_z_centers=(15.0,), chunk_cells=32)
    binner = KBinner(nx, ny, cfg.box_mpc, cfg.n_bins)
    k = binner.centers
    z_grid = np.linspace(5.0, 25.0, nz)
    chunks = build_chunks(cfg, z_grid, nz)

    def fourier_blur(field, sigma_vox):
        """Exact Gaussian low-pass, applied per transverse slice.

        scipy's real-space gaussian_filter truncates the kernel at 4 sigma;
        at high k its transfer function is dominated by sign-flipping
        truncation ripples, which would break the exact-filter expectations
        (r ~ 1, Gaussian power suppression) that tests 2 and 3 rely on.
        """
        kxv = 2 * np.pi * np.fft.fftfreq(field.shape[0])
        kyv = 2 * np.pi * np.fft.fftfreq(field.shape[1])
        w = np.exp(-0.5 * sigma_vox ** 2
                   * (kxv[:, None] ** 2 + kyv[None, :] ** 2))[:, :, None]
        return np.fft.ifftn(np.fft.fftn(field, axes=(0, 1)) * w,
                            axes=(0, 1)).real

    # Truth: red-spectrum GRF around x_HI ~ 0.5 (every slice mid-stage/active).
    truth = fourier_blur(rng.standard_normal((nx, ny, nz)), 3.0)
    truth = 0.5 + 0.2 * truth / truth.std()

    def stats(pred, field=truth):
        acc = PowerSpectrumAccumulator(cfg, binner)
        acc.add_cone(0, pred, field)
        res = acc.reduce()
        return res["stage_ratio_med"][-1], res["stage_r_med"][-1]

    ok = True

    # 1) perfect prediction -> ratio = 1, r = 1 at every k
    ratio, r = stats(truth.copy())
    c1 = np.allclose(ratio, 1.0, atol=1e-9) and np.allclose(r, 1.0, atol=1e-9)
    ok &= c1
    print(f"[1] identity: max|ratio-1|={np.abs(ratio-1).max():.2e}, "
          f"max|r-1|={np.abs(r-1).max():.2e}  -> {'PASS' if c1 else 'FAIL'}")

    # 2) linear blur -> power suppressed at high k, phases preserved (r ~ 1)
    blurred = fourier_blur(truth, 2.0)
    ratio, r = stats(blurred)
    c2 = ratio[0] > 0.8 and ratio[-1] < 0.1 and np.nanmin(r) > 0.98
    ok &= c2
    print(f"[2] blur: ratio {ratio[0]:.2f} -> {ratio[-1]:.2e}, "
          f"min r={np.nanmin(r):.4f}  -> {'PASS' if c2 else 'FAIL'}")

    # 3) additive white noise -> decorrelation and excess power at high k only
    noisy = truth + 0.05 * rng.standard_normal(truth.shape)
    ratio, r = stats(noisy)
    c3 = r[0] > 0.9 and r[-1] < 0.5 and ratio[-1] > 2.0 and abs(ratio[0] - 1) < 0.1
    ok &= c3
    print(f"[3] noise: r {r[0]:.3f} -> {r[-1]:.3f}, "
          f"ratio[-1]={ratio[-1]:.1f}  -> {'PASS' if c3 else 'FAIL'}")

    # 4) physical wavenumbers: a pure sinusoid lands in the right k bin
    x = np.arange(nx)
    mode = 8                                    # k = 2 pi * 8 / 200 Mpc^-1
    wave = np.broadcast_to(np.sin(2 * np.pi * mode * x / nx)[:, None, None],
                           (nx, ny, nz)).copy()
    pt = binner.binned_slice_sums(_demean_slices(wave))
    k_true = 2 * np.pi * mode / cfg.box_mpc
    b = int(np.argmax(pt[0]))
    c4 = binner.edges[b] <= k_true < binner.edges[b + 1]
    ok &= c4
    print(f"[4] k units: sinusoid k={k_true:.4f} in bin "
          f"[{binner.edges[b]:.4f}, {binner.edges[b+1]:.4f})  "
          f"-> {'PASS' if c4 else 'FAIL'}")

    # 5) saturated cone (x_HI ~ 1 everywhere) -> everything masked, no blowups
    flat = np.full((nx, ny, nz), 0.995) + 1e-5 * rng.standard_normal((nx, ny, nz))
    acc = PowerSpectrumAccumulator(cfg, binner, chunks=chunks)
    acc.add_cone(0, flat.copy(), flat)
    res = acc.reduce()
    c5 = (np.all(~np.isfinite(res["stage_ratio_med"]))
          and np.all(~np.isfinite(res["cyl_ratio_med"])))
    ok &= c5
    print(f"[5] saturated cone fully masked  -> {'PASS' if c5 else 'FAIL'}")

    # 6) cylindrical identity -> ratio = 1, r = 1 in every (k_perp, k_par) cell
    acc = PowerSpectrumAccumulator(cfg, binner, chunks=chunks)
    acc.add_cone(0, truth.copy(), truth)
    res = acc.reduce()
    cr, crr = res["cyl_ratio_med"], res["cyl_r_med"]
    c6 = (np.all(np.isfinite(cr)) and np.allclose(cr, 1.0, atol=1e-9)
          and np.allclose(crr, 1.0, atol=1e-9))
    ok &= c6
    print(f"[6] cylindrical identity: max|ratio-1|={np.abs(cr-1).max():.2e}  "
          f"-> {'PASS' if c6 else 'FAIL'}")

    # 7) k_par localization: sin(kx x) sin(kz z) lands in the right map cell
    ch = chunks[0]
    j_par = 4                                     # cycles per chunk
    mode = 8                                      # transverse cycles per box
    wave = (np.sin(2 * np.pi * mode * np.arange(nx) / nx)[:, None, None]
            * np.sin(2 * np.pi * j_par * np.arange(nz) / cfg.chunk_cells)
            [None, None, :]) * np.ones((nx, ny, nz))
    acc7 = PowerSpectrumAccumulator(cfg, binner, chunks=chunks)
    pt7, _, _ = acc7._cylindrical_sums(
        wave[:, :, ch.start:ch.stop], wave[:, :, ch.start:ch.stop], ch)
    row, col = np.unravel_index(int(np.argmax(pt7)), pt7.shape)
    k_perp_true = 2 * np.pi * mode / cfg.box_mpc
    exp_col = int(np.digitize(k_perp_true, binner.edges) - 1)
    exp_row = j_par - 1                           # DC plane dropped
    c7 = (row, col) == (exp_row, exp_col)
    ok &= c7
    print(f"[7] cylindrical localization: peak at (k_par row, k_perp bin)="
          f"({row},{col}), expected ({exp_row},{exp_col})  "
          f"-> {'PASS' if c7 else 'FAIL'}")

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
    src.add_argument("--selftest", action="store_true",
                     help="run synthetic verification")
    src.add_argument("--manifest", type=Path,
                     help="JSON manifest of saved npz cubes")
    src.add_argument("--checkpoints", nargs="+", metavar="name=path",
                     help="cluster mode: build cubes from checkpoints")
    src.add_argument("--replot", type=Path, metavar="NPZ",
                     help="re-render all figures from a saved ps_results.npz "
                          "(no GPU/data needed; for plot-style iterations)")
    ap.add_argument("--out", type=Path, default=Path("figures/3d_xhi/eval/ps_out"))
    ap.add_argument("--n-cones", type=int, default=200)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--save-cubes", type=Path, default=None)
    ap.add_argument("--box-mpc", type=float, default=BOX_MPC)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--chunk-z", type=float, nargs="+", default=[7.0, 9.0, 11.0],
                    help="redshift centres of the cylindrical (k_perp, k_par) "
                         "map chunks")
    ap.add_argument("--chunk-cells", type=int, default=32,
                    help="LOS cells per cylindrical chunk")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    if args.replot:
        data = np.load(args.replot, allow_pickle=True)
        results: dict[str, dict] = {}
        for flat_key in data.files:
            name, _, key = flat_key.partition("/")
            arr = data[flat_key]
            if key == "stage_labels":
                arr = [str(v) for v in arr.tolist()]
            results.setdefault(name, {})[key] = arr
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        plot_overlay(results, out / "ps_overlay.png")
        plot_cylindrical_maps(results, out / "ps_cyl_ratio.png",
                              out / "ps_cyl_r.png")
        plot_stage_curves(results, out / "ps_stage_curves.png")
        print(f"[replot] figures re-rendered into {out}")
        return

    cfg = SpectrumConfig(box_mpc=args.box_mpc, n_bins=args.n_bins,
                         chunk_z_centers=tuple(args.chunk_z),
                         chunk_cells=args.chunk_cells)

    if args.manifest:
        run_from_manifest(args.manifest, cfg, args.out)
    else:
        run_from_checkpoints(_parse_kv(args.checkpoints), cfg, args.out,
                             n_cones=args.n_cones, split=args.split,
                             save_cubes=args.save_cubes)


if __name__ == "__main__":
    main()
