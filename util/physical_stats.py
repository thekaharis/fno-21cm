"""Physical summary statistics for emulated reionization fields (numpy only).

The functions here turn an x_HI lightcone into quantities a reionization
audience reads directly, independent of any network or training code:

* ``thomson_optical_depth`` / ``history_features`` -- the CMB optical depth
  tau_e and the redshifts at which the global neutral fraction crosses
  0.25 / 0.5 / 0.75, to be compared with the Planck 2018 uncertainties
  (``PLANCK_TAU_SIGMA``, ``PLANCK_ZRE_SIGMA``).
* ``brightness_temperature_mk`` -- the 21-cm brightness temperature in the
  saturated spin-temperature limit without redshift-space distortions.
  Truth and prediction share the same density, so differences are purely the
  emulator's x_HI error propagated to the observable. It is *not* the
  21cmFAST ``brightness_temp`` field, which also carries T_S and velocities.
* ``slice_moments`` -- per-slice mean / variance / skewness.
* ``fit_even_polynomial`` -- weighted least squares in powers of k^2, used for
  the EFT-style transfer function T(k) = b_1 + b_{nabla^2} k^2 + ... and the
  stochastic term P_eps(k) = P_eps,0 + P_eps,2 k^2 + ...
* ``histogram_w1`` -- Wasserstein-1 distance between two binned distributions.

Cosmology defaults follow the 21cmFAST (Planck 2018) values; Omega_m can be
passed per cone because it is one of the sampled parameters.
"""
from __future__ import annotations

import numpy as np


# 21cmFAST default cosmology (Planck 2018).  Omega_m is sampled per cone.
HUBBLE_H = 0.6766
OMEGA_B = 0.04897
OMEGA_M = 0.30966
Y_HE = 0.245
Z_HE_DOUBLE = 3.0       # HeII -> HeIII taken as instantaneous at this redshift

# Planck 2018 (TT,TE,EE+lowE+lensing): tau = 0.0544 +- 0.0073,
# instantaneous-reionization midpoint z_re = 7.67 +- 0.73.
PLANCK_TAU = 0.0544
PLANCK_TAU_SIGMA = 0.0073
PLANCK_ZRE_SIGMA = 0.73

SIGMA_T_M2 = 6.6524587321e-29
C_M_S = 2.99792458e8
M_PROTON_KG = 1.67262192369e-27
G_SI = 6.67430e-11
MPC_M = 3.0856775814913673e22

# numpy < 2.0 only has trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def hubble_rate_si(z, omega_m: float = OMEGA_M, h: float = HUBBLE_H):
    """H(z) in 1/s for flat LCDM; radiation is neglected (<1% at z <= 25)."""
    h0 = h * 100.0e3 / MPC_M
    z = np.asarray(z, dtype=np.float64)
    return h0 * np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def thomson_optical_depth(
    z_grid,
    xhi_mean,
    omega_m: float = OMEGA_M,
    omega_b: float = OMEGA_B,
    h: float = HUBBLE_H,
    y_he: float = Y_HE,
    z_he_double: float = Z_HE_DOUBLE,
    n_low: int = 2000,
) -> float:
    """CMB Thomson optical depth of a global reionization history.

    Helium is singly ionized alongside hydrogen and doubly ionized below
    ``z_he_double``. The history is taken as fully ionized below the grid and
    fully neutral above it; the mean ionized fraction is clipped to [0, 1] so
    an emulator overshoot cannot produce unphysical electron densities.
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    xhi_mean = np.asarray(xhi_mean, dtype=np.float64)
    order = np.argsort(z_grid)
    z_grid, xhi_mean = z_grid[order], xhi_mean[order]
    ionized = 1.0 - np.clip(xhi_mean, 0.0, 1.0)

    f_he = y_he / (4.0 * (1.0 - y_he))
    rho_crit = 3.0 * (h * 100.0e3 / MPC_M) ** 2 / (8.0 * np.pi * G_SI)
    n_h0 = rho_crit * omega_b * (1.0 - y_he) / M_PROTON_KG

    z_low = np.linspace(0.0, z_grid[0], n_low, endpoint=False)
    z = np.concatenate([z_low, z_grid])
    x_hii = np.concatenate([np.ones_like(z_low), ionized])
    x_e = x_hii * (1.0 + f_he) + np.where(z < z_he_double, f_he, 0.0)
    integrand = (
        C_M_S * SIGMA_T_M2 * n_h0 * (1.0 + z) ** 2 * x_e
        / hubble_rate_si(z, omega_m, h)
    )
    return float(_trapezoid(integrand, z))


def history_crossing(z_grid, xhi_mean, level: float) -> float:
    """Redshift at which the global x_HI first reaches ``level`` from below.

    Scans from low to high redshift, i.e. returns the latest-time crossing,
    linearly interpolated. NaN when the level is never reached, or is already
    exceeded at the lowest redshift on the grid (crossing not resolved).
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    xhi_mean = np.asarray(xhi_mean, dtype=np.float64)
    order = np.argsort(z_grid)
    z, x = z_grid[order], xhi_mean[order]
    above = np.flatnonzero(x >= level)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    i = int(above[0])
    fraction = (level - x[i - 1]) / (x[i] - x[i - 1])
    return float(z[i - 1] + fraction * (z[i] - z[i - 1]))


def history_features(z_grid, xhi_mean) -> dict:
    """Crossing redshifts for x_HI = 0.25 / 0.5 / 0.75 and the duration.

    ``delta_z = z(x_HI = 0.75) - z(x_HI = 0.25)`` is positive for a history
    that reionizes with time.
    """
    z25 = history_crossing(z_grid, xhi_mean, 0.25)
    z50 = history_crossing(z_grid, xhi_mean, 0.50)
    z75 = history_crossing(z_grid, xhi_mean, 0.75)
    return {"z_xhi25": z25, "z_xhi50": z50, "z_xhi75": z75,
            "delta_z": z75 - z25}


def brightness_temperature_mk(
    xhi,
    delta,
    z_grid,
    omega_m: float = OMEGA_M,
    omega_b: float = OMEGA_B,
    h: float = HUBBLE_H,
):
    """21-cm brightness temperature in mK for an (Nx, Ny, Nz) lightcone.

    dT_b = 27 x_HI (1 + delta) (Omega_b h^2 / 0.023)
           * sqrt(0.15 / (Omega_m h^2) * (1 + z) / 10)  mK

    (Furlanetto, Oh & Briggs 2006) with T_S >> T_CMB and no velocity term.
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    t0 = (
        27.0 * (omega_b * h ** 2 / 0.023)
        * np.sqrt(0.15 / (omega_m * h ** 2) * (1.0 + z_grid) / 10.0)
    )
    return t0[None, None, :] * np.asarray(xhi) * (1.0 + np.asarray(delta))


def slice_moments(field) -> dict:
    """Mean, variance and skewness of every transverse slice of a lightcone."""
    field = np.asarray(field, dtype=np.float64)
    mean = field.mean(axis=(0, 1))
    centered = field - mean
    variance = (centered ** 2).mean(axis=(0, 1))
    third = (centered ** 3).mean(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        skewness = np.where(variance > 0, third / variance ** 1.5, np.nan)
    return {"mean": mean, "variance": variance, "skewness": skewness}


def fit_even_polynomial(k, y, variance, k_max: float, n_terms: int = 2) -> dict:
    """Weighted least squares ``y(k) = sum_j c_j k^(2j)`` over ``k <= k_max``.

    Returns the coefficients, their 1-sigma errors from the WLS covariance,
    the reduced chi^2 and the number of bins used. All-NaN when fewer than
    ``n_terms + 1`` usable bins are available.
    """
    k = np.asarray(k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    use = (
        (k <= k_max) & np.isfinite(y) & np.isfinite(variance) & (variance > 0)
    )
    empty = {
        "coef": np.full(n_terms, np.nan), "sigma": np.full(n_terms, np.nan),
        "chi2_red": float("nan"), "n_bins": int(use.sum()),
    }
    if use.sum() < n_terms + 1:
        return empty
    design = k[use, None] ** (2 * np.arange(n_terms))[None, :]
    weight = 1.0 / variance[use]
    normal = design.T @ (design * weight[:, None])
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return empty
    coef = covariance @ (design.T @ (weight * y[use]))
    residual = y[use] - design @ coef
    dof = int(use.sum()) - n_terms
    return {
        "coef": coef,
        "sigma": np.sqrt(np.diag(covariance)),
        "chi2_red": float(np.sum(weight * residual ** 2) / dof),
        "n_bins": int(use.sum()),
    }


def histogram_w1(counts_a, counts_b, edges) -> float:
    """Wasserstein-1 distance between two histograms on shared ``edges``."""
    counts_a = np.asarray(counts_a, dtype=np.float64)
    counts_b = np.asarray(counts_b, dtype=np.float64)
    total_a, total_b = counts_a.sum(), counts_b.sum()
    if total_a <= 0 or total_b <= 0:
        return float("nan")
    cdf_gap = np.cumsum(counts_a / total_a - counts_b / total_b)
    return float(np.sum(np.abs(cdf_gap) * np.diff(edges)))
