from __future__ import annotations

import numpy as np
import pytest

from util.metrics_21cm import (
    mean_free_path_samples_2d,
    trace_ray_to_neutral_2d,
)
from viz.bubble_size_evaluation import BubbleSizeAccumulator, BubbleSizeConfig


def test_axis_aligned_ray_has_exact_boundary_distance():
    mask = np.zeros((8, 8), dtype=bool)
    mask[:3, :] = True

    distance = trace_ray_to_neutral_2d(
        mask,
        start_xy=(1.5, 3.5),
        direction_xy=(1.0, 0.0),
        cell_size_mpc=(2.0, 2.0),
        max_distance_mpc=20.0,
    )

    assert distance == pytest.approx(3.0)


def test_ray_wraps_periodically_before_hitting_neutral_cell():
    mask = np.zeros((8, 8), dtype=bool)
    mask[[7, 0], :] = True

    distance = trace_ray_to_neutral_2d(
        mask,
        start_xy=(7.5, 2.5),
        direction_xy=(1.0, 0.0),
        cell_size_mpc=(1.0, 1.0),
        max_distance_mpc=10.0,
    )

    assert distance == pytest.approx(1.5)


def test_fully_ionized_ray_is_right_censored():
    distance = trace_ray_to_neutral_2d(
        np.ones((8, 8), dtype=bool),
        start_xy=(2.5, 2.5),
        direction_xy=(1.0, 0.25),
        cell_size_mpc=(1.0, 1.0),
        max_distance_mpc=8.0,
    )

    assert np.isnan(distance)


def test_mean_free_path_sampling_is_reproducible():
    mask = np.zeros((24, 24), dtype=bool)
    yy, xx = np.ogrid[:24, :24]
    mask[(xx - 12) ** 2 + (yy - 12) ** 2 <= 7 ** 2] = True

    first = mean_free_path_samples_2d(
        mask, 200, (1.0, 1.0), 24.0, seed=123
    )
    second = mean_free_path_samples_2d(
        mask, 200, (1.0, 1.0), 24.0, seed=123
    )

    np.testing.assert_array_equal(
        first["distances_mpc"], second["distances_mpc"]
    )
    assert first["n_censored"] == second["n_censored"]


def _bubble_cube(radius: float, size: int = 32, nz: int = 5) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    ionized = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= radius ** 2
    field = np.ones((size, size, nz), dtype=np.float64)
    field[ionized, :] = 0.0
    return field


def test_accumulator_identity_has_zero_distribution_error():
    truth = _bubble_cube(radius=10)
    cfg = BubbleSizeConfig(
        box_mpc=32.0,
        rays_per_slice=128,
        slices_per_stage=2,
        n_bins=10,
        max_distance_mpc=32.0,
        seed=9,
    )
    accumulator = BubbleSizeAccumulator(cfg, truth.shape[:2])
    accumulator.add_cone(7, truth.copy(), truth)
    result = accumulator.reduce()
    valid = result["n_valid_cones"] > 0

    assert np.all(result["restricted_wasserstein_mpc_med"][valid] == 0.0)
    assert np.all(result["js_divergence_med"][valid] == 0.0)
    np.testing.assert_array_equal(
        result["truth_mass_med"][valid], result["pred_mass_med"][valid]
    )


def test_accumulator_detects_larger_predicted_bubble():
    truth = _bubble_cube(radius=7)
    pred = _bubble_cube(radius=12)
    cfg = BubbleSizeConfig(
        box_mpc=32.0,
        rays_per_slice=512,
        slices_per_stage=3,
        n_bins=12,
        max_distance_mpc=32.0,
        seed=11,
    )
    accumulator = BubbleSizeAccumulator(cfg, truth.shape[:2])
    accumulator.add_cone(3, pred, truth)
    result = accumulator.reduce()
    valid = result["n_valid_cones"] > 0

    assert np.all(result["restricted_wasserstein_mpc_med"][valid] > 0)
    assert np.all(result["relative_mean_bias_med"][valid] > 0)


def test_no_ionized_prediction_is_counted_as_underflow_failure():
    truth = _bubble_cube(radius=9)
    pred = np.ones_like(truth)
    cfg = BubbleSizeConfig(
        box_mpc=32.0,
        rays_per_slice=128,
        slices_per_stage=2,
        n_bins=10,
        max_distance_mpc=32.0,
        seed=13,
    )
    accumulator = BubbleSizeAccumulator(cfg, truth.shape[:2])
    accumulator.add_cone(5, pred, truth)
    result = accumulator.reduce()
    valid = result["n_valid_cones"] > 0

    assert np.all(result["pred_underflow_fraction_med"][valid] == 1.0)
    assert np.all(result["restricted_wasserstein_mpc_med"][valid] > 0)


def test_nonfinite_distance_configuration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        BubbleSizeConfig(max_distance_mpc=float("nan"))
