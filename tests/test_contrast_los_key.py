"""The LOS key and the whole-cube refit path.

The key is what decides which theta a slice receives, so an error here is
invisible in the loss and shows up only as a schedule that never finds the
responsive band.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from contrast import (
    ContrastComposed,
    ContrastOutput,
    SteppedThetaSchedule,
    isotonic,
    los_key,
    los_mean,
)
from util import contrast_refit as refit


def _cone(n_los: int = 64, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """One monotone x_HI(z) curve plus per-slice noise, as (1, n_los)."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 1.0, n_los)
    truth = 1.0 / (1.0 + np.exp(-(z - 0.5) * 12.0))
    return (truth + rng.normal(0.0, noise, n_los))[None, :], truth[None, :]


def test_pav_returns_a_monotone_fit() -> None:
    noisy, _ = _cone()
    fitted, _ = isotonic(noisy)
    assert np.all(np.diff(fitted[0]) >= -1e-9)


def test_isotonic_picks_the_decreasing_direction_when_that_fits_better() -> None:
    noisy, _ = _cone()
    fitted, _ = isotonic(noisy[:, ::-1].copy())
    assert np.all(np.diff(fitted[0]) <= 1e-9)


def test_isotonic_recovers_a_monotone_curve_better_than_the_raw_means() -> None:
    noisy, truth = _cone(noise=0.05, seed=3)
    fitted, _ = isotonic(noisy)
    raw_mae = float(np.abs(noisy - truth).mean())
    fit_mae = float(np.abs(fitted - truth).mean())
    # The whole point of the change: pooling each slice with its neighbours
    # must materially beat the per-slice estimate, not merely tie it.
    assert fit_mae < 0.5 * raw_mae


def test_isotonic_leaves_an_already_monotone_curve_alone() -> None:
    _, truth = _cone(noise=0.0)
    fitted, sse = isotonic(truth)
    assert sse == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(fitted, truth)


def test_los_mean_is_per_slice_not_per_cube() -> None:
    x = torch.zeros(2, 1, 4, 4, 8)
    x[:, :, :, :, 4:] = 1.0
    m = los_mean(x)
    assert m.shape == (2, 8)
    assert torch.allclose(m[0], torch.tensor([0.0] * 4 + [1.0] * 4))


def test_los_key_mean_mode_is_the_unsmoothed_mean() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 4, 4, 16)
    assert torch.allclose(los_key(x, "mean"), los_mean(x))


def test_los_key_monotone_mode_is_monotone_along_the_los_axis() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 4, 4, 16)
    key = los_key(x, "monotone")
    assert key.shape == (2, 16)
    diffs = torch.diff(key, dim=1)
    assert bool((diffs >= -1e-6).all()) or bool((diffs <= 1e-6).all())


def test_contrast_output_rejects_an_unknown_key_mode() -> None:
    with pytest.raises(ValueError, match="key mode"):
        ContrastOutput("xhi", schedule_kind="stepped", key_mode="nonsense")


def test_xhi_contrast_gives_each_los_slice_its_own_theta() -> None:
    out = ContrastOutput("xhi", schedule_kind="stepped", n_bins=4,
                         key_mode="mean")
    # Bins are log-spaced over (0, 1]; force a visible split between a nearly
    # ionised half of the cone and a nearly neutral one.
    with torch.no_grad():
        out.schedule.raw.copy_(torch.tensor([-4.0, -4.0, 4.0, 4.0]))
    x = torch.full((1, 1, 4, 4, 8), 0.6)
    x[..., :4] = 0.001
    y = out(x)
    # theta is small on the near-ionised half, so its values are pushed hard
    # toward 0; the near-identity half must come back essentially unchanged.
    # The comparison has to be relative -- the sharpened half lives at ~1e-3,
    # where any absolute tolerance calls everything equal.
    assert float(y[..., :4].detach().max()) < 0.01 * float(x[..., :4].max())
    assert torch.allclose(y[..., 4:], x[..., 4:], rtol=0.02)


def test_2d_path_is_untouched_by_the_key_mode() -> None:
    x = torch.rand(3, 1, 8, 8)
    a = ContrastOutput("xhi", schedule_kind="stepped", key_mode="mean")(x)
    b = ContrastOutput("xhi", schedule_kind="stepped", key_mode="monotone")(x)
    assert torch.allclose(a, b)


# --------------------------------------------------------------- refit path

class _Base(torch.nn.Module):
    """Deterministic stand-in for a trained operator: a blurred step in LOS."""

    def forward(self, x):
        return torch.sigmoid((x[:, :1] - 0.5) * 4.0)


class _Model(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.fno = ContrastComposed(_Base(), "xhi", schedule_kind="stepped", **kw)


def _loader(n_cubes: int = 4, n_los: int = 32):
    torch.manual_seed(0)
    out = []
    for _ in range(n_cubes):
        x = torch.rand(1, 1, 4, 4, n_los)
        y = (torch.rand(1, 1, 4, 4, n_los) > 0.5).float()
        out.append({"x": x, "y": y})
    return out


def test_collect_keeps_cubes_whole_by_default() -> None:
    model = _Model()
    pred, truth = refit.collect_base_outputs(
        model, _loader(), "cpu", max_samples=64)
    assert pred.dim() == 5
    assert pred.shape[-1] == 32
    assert pred.shape[0] == 2                 # 64 slices / 32 per cube
    assert pred.shape == truth.shape


def test_collect_can_still_flatten_to_slices() -> None:
    model = _Model()
    pred, truth = refit.collect_base_outputs(
        model, _loader(), "cpu", max_samples=64, flatten=True)
    assert pred.dim() == 4
    assert len(pred) == 64
    assert pred.shape == truth.shape


def test_fit_on_whole_cubes_sees_the_objective_in_3d() -> None:
    """The objective must be handed 5-D tensors, not transverse planes."""
    seen = []

    def objective(out, y):
        seen.append(tuple(out.shape))
        return ((out - y) ** 2).mean()

    model = _Model()
    pred, truth = refit.collect_base_outputs(
        model, _loader(), "cpu", max_samples=64)
    stats = refit.fit_schedule(pred, truth, steps=3, objective=objective,
                               batch=2, template=model.fno.contrast.schedule)
    assert all(len(s) == 5 for s in seen)
    assert stats["n_slices"] == 64            # counted in slices, not cubes


def test_refit_installs_a_finite_schedule_and_reports_slices() -> None:
    model = _Model()
    stats = refit.refit_and_install(model, _loader(), "cpu", max_samples=64,
                                    steps=5, batch=2)
    assert stats["rejected"] == 0.0
    assert stats["n_slices"] == 64
    assert all(np.isfinite(v) for v in stats["thetas"])
    assert all(v >= 0.25 for v in stats["thetas"])          # theta floor
    assert sum(stats["bin_counts"]) == 64
    assert "refit rejected" not in refit.summary_line(stats)


def test_refit_key_matches_the_key_inference_will_use() -> None:
    """A theta fitted for a bin must be the theta that bin then receives."""
    model = _Model(key_mode="mean")
    pred, _ = refit.collect_base_outputs(model, _loader(), "cpu", max_samples=64)
    key, _ = refit._schedule_key(pred, model.fno.contrast.key_mode)
    assert torch.allclose(key, los_key(pred, "mean"))

    model = _Model(key_mode="monotone")
    key, _ = refit._schedule_key(pred, model.fno.contrast.key_mode)
    assert torch.allclose(key, los_key(pred, "monotone"))


def test_refit_reports_key_jitter_and_band_occupancy() -> None:
    model = _Model()
    stats = refit.refit_and_install(model, _loader(), "cpu", max_samples=64,
                                    steps=5, batch=2)
    assert stats["key_jitter"] >= 0.0
    assert 0.0 <= stats["frac_band"] <= 1.0
    assert "key jitter" in refit.summary_line(stats)


def test_key_jitter_is_zero_for_an_already_monotone_key() -> None:
    """Jitter must measure noise only -- a clean monotone cone has none."""
    n_los = 32
    pred = torch.zeros(1, 1, 4, 4, n_los)
    pred[..., :] = torch.linspace(0.0, 1.0, n_los).view(1, 1, 1, 1, -1)
    key, _ = refit._schedule_key(pred, "monotone")
    diag = refit._key_diagnostics(pred, key)
    assert diag["key_jitter"] == pytest.approx(0.0, abs=1e-9)


def test_key_jitter_is_positive_when_the_means_are_noisy() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 4, 4, 32)
    key, _ = refit._schedule_key(pred, "monotone")
    assert refit._key_diagnostics(pred, key)["key_jitter"] > 1e-3


def test_key_diagnostics_are_absent_for_2d_predictions() -> None:
    pred = torch.rand(4, 1, 8, 8)
    key, _ = refit._schedule_key(pred, "mean")
    assert refit._key_diagnostics(pred, key) == {}


def test_stepped_schedule_maps_a_2d_key_without_reshaping() -> None:
    sched = SteppedThetaSchedule(n_bins=6)
    key = torch.rand(3, 16)
    assert sched(key).shape == (3, 16)
