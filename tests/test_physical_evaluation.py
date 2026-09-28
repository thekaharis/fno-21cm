from __future__ import annotations

import json

import numpy as np
import pytest

from util.physical_stats import (
    PLANCK_TAU,
    PLANCK_TAU_SIGMA,
    brightness_temperature_mk,
    fit_even_polynomial,
    histogram_w1,
    history_features,
    slice_moments,
    thomson_optical_depth,
)
from viz.physical_evaluation import (
    PhysicalAccumulator,
    PhysicalConfig,
    _transverse_blur,
    run_from_manifest,
    synthetic_cone,
)


Z_GRID = np.linspace(5.0, 25.0, 256)


def test_instantaneous_reionization_reproduces_planck_tau():
    # Planck's z_re = 7.67 is the midpoint of the tau -> z_re mapping.
    history = (Z_GRID > 7.67).astype(float)
    tau = thomson_optical_depth(Z_GRID, history)
    assert abs(tau - PLANCK_TAU) < 0.2 * PLANCK_TAU_SIGMA


def test_optical_depth_increases_with_earlier_reionization():
    taus = [thomson_optical_depth(Z_GRID, (Z_GRID > z).astype(float))
            for z in (6.0, 8.0, 10.0)]
    assert taus[0] < taus[1] < taus[2]


def test_optical_depth_clips_unphysical_neutral_fractions():
    history = (Z_GRID > 8.0).astype(float)
    overshoot = np.where(history > 0, 1.3, -0.2)
    assert thomson_optical_depth(Z_GRID, overshoot) == pytest.approx(
        thomson_optical_depth(Z_GRID, history))


def test_history_features_of_tanh_history():
    history = 0.5 * (1 + np.tanh((Z_GRID - 8.0) / 0.5))
    features = history_features(Z_GRID, history)
    assert features["z_xhi50"] == pytest.approx(8.0, abs=0.01)
    assert features["delta_z"] == pytest.approx(2 * 0.5 * np.arctanh(0.5), abs=0.02)


def test_unresolved_crossing_is_nan():
    assert np.isnan(history_features(Z_GRID, np.ones_like(Z_GRID))["z_xhi50"])
    assert np.isnan(history_features(Z_GRID, np.zeros_like(Z_GRID))["z_xhi50"])


def test_brightness_temperature_normalization():
    # Neutral, mean-density gas at z = 9 with Omega_b h^2 = 0.023 and
    # Omega_m h^2 = 0.15 gives exactly 27 mK.
    h = 0.7
    dtb = brightness_temperature_mk(
        np.ones((1, 1, 1)), np.zeros((1, 1, 1)), [9.0],
        omega_m=0.15 / h ** 2, omega_b=0.023 / h ** 2, h=h,
    )
    assert dtb[0, 0, 0] == pytest.approx(27.0)


def test_slice_moments_skewness_sign():
    rng = np.random.default_rng(0)
    field = rng.exponential(size=(64, 64, 2))
    field[:, :, 1] *= -1
    skew = slice_moments(field)["skewness"]
    assert skew[0] > 1.0 and skew[1] < -1.0


def test_even_polynomial_fit_recovers_coefficients():
    k = np.geomspace(0.03, 0.3, 12)
    y = -0.8 + 5.0 * k ** 2
    fit = fit_even_polynomial(k, y, np.full_like(k, 1e-6), k_max=0.25, n_terms=2)
    np.testing.assert_allclose(fit["coef"], [-0.8, 5.0], rtol=1e-8)
    assert fit["n_bins"] == int((k <= 0.25).sum())


def test_even_polynomial_fit_needs_enough_bins():
    fit = fit_even_polynomial([0.1, 0.2], [1.0, 1.0], [1.0, 1.0], 0.25, 2)
    assert np.all(np.isnan(fit["coef"]))


def test_histogram_w1_is_shift_distance():
    edges = np.linspace(0.0, 10.0, 11)
    a = np.zeros(10)
    b = np.zeros(10)
    a[2], b[5] = 1.0, 1.0
    assert histogram_w1(a, b, edges) == pytest.approx(3.0)
    assert histogram_w1(a, a, edges) == 0.0


def _single_cone(pred_fn, density=True):
    cfg = PhysicalConfig(n_k_bins=10, xbar_bins=10)
    z_grid, delta, truth = synthetic_cone(size=40, n_z=48)
    accumulator = PhysicalAccumulator(cfg, z_grid, truth.shape[:2])
    accumulator.add_cone(0, pred_fn(truth), truth, delta if density else None)
    return accumulator.reduce()


def test_linear_tracer_bias_is_recovered_exactly():
    # u = b1 * delta per slice makes T(k) = b1 at every k and P_eps = 0.
    cfg = PhysicalConfig(n_k_bins=10)
    rng = np.random.default_rng(1)
    delta = rng.standard_normal((32, 32, 16))
    truth = 0.5 - 0.1 * delta
    accumulator = PhysicalAccumulator(cfg, np.linspace(6, 9, 16), (32, 32))
    accumulator.add_cone(0, truth, truth, delta)
    result = accumulator.reduce()
    transfer = result["eft_T_truth_med"][-1]
    np.testing.assert_allclose(transfer[np.isfinite(transfer)], -0.1, rtol=1e-8)


def test_perfect_prediction_has_no_physical_error():
    result = _single_cone(lambda truth: truth)
    assert result["tau_pred"][0] == result["tau_truth"][0]
    assert np.nanmax(result["xhi_w1_all"]) == 0.0
    assert np.nanmax(np.abs(result["sig_xhi_med"])) < 1e-9
    np.testing.assert_array_equal(result["eft_bias_pred"], result["eft_bias_truth"])


def test_blurred_prediction_is_flagged():
    result = _single_cone(lambda truth: _transverse_blur(truth, 2.0))
    active = -1
    assert result["partial_pred_med"][active] > result["partial_truth_med"][active]
    assert result["sig_dtb_med"][active, -1] < 0
    assert np.nanmedian(result["eft_peps_ratio_med"][active]) < 1.0


def test_density_free_cone_skips_observable():
    result = _single_cone(lambda truth: truth, density=False)
    assert not result["has_density"]
    assert "eft_bias_truth" not in result
    assert "xhi_w1_med" in result


def test_manifest_run_writes_outputs(tmp_path):
    z_grid, delta, truth = synthetic_cone(size=32, n_z=40)
    entries = {"a": [], "b": []}
    for name, sigma in (("a", 0.5), ("b", 2.0)):
        path = tmp_path / f"{name}.npz"
        np.savez(path, truth=truth, pred=_transverse_blur(truth, sigma), density=delta)
        entries[name].append({"cone_id": 1, "npz": str(path), "omega_m": 0.31})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"z_grid": z_grid.tolist(), "models": entries}))

    out = tmp_path / "out"
    run_from_manifest(manifest, PhysicalConfig(n_k_bins=8, xbar_bins=8), out)
    for name in ("physical_history.png", "physical_pdfs.png", "physical_21cm.png",
                 "physical_eft.png", "physical_stage_metrics.csv",
                 "physical_history_per_cone.csv", "physical_summary.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "physical_summary.json").read_text())
    assert summary["b"]["xhi_w1_active_median"] > summary["a"]["xhi_w1_active_median"]
