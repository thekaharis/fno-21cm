"""Tests for the z_re loss configuration (H2Loss2d, LOSS_* env wiring)."""

from __future__ import annotations

import importlib

import pytest
import torch

from losses import AbsoluteLoss, H2Loss2d, RelativeLoss, WeightedLoss


def test_h2_zero_for_identical_fields() -> None:
    loss = H2Loss2d()
    y = torch.randn(2, 1, 16, 16)
    assert loss.abs(y, y) == pytest.approx(0.0)
    assert loss.rel(y, y) == pytest.approx(0.0)


def test_h2_rel_matches_abs_over_target_norm() -> None:
    loss = H2Loss2d()
    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 16)
    y = torch.rand(2, 1, 16, 16) + 0.5
    _, norm = loss._squared_error_and_norm(x, y, None)
    expected = (loss.abs(x, y, take_root=True) / 1.0)  # sum-reduced abs
    # rel reduces the per-sample ratio; reconstruct it from per-sample terms
    error, _ = loss._squared_error_and_norm(x, y, None)
    per_sample = error.sqrt() / norm.sqrt().clamp_min(1e-8)
    assert loss.rel(x, y) == pytest.approx(float(per_sample.sum()))
    assert loss.abs(x, y) == pytest.approx(float(error.sqrt().sum()))
    assert expected > 0


def test_h2_rel_is_finite_for_zero_target() -> None:
    loss = H2Loss2d()
    x = torch.rand(1, 1, 8, 8)
    y = torch.zeros(1, 1, 8, 8)
    value = loss.rel(x, y)
    assert torch.isfinite(value)
    assert value > 0


def test_h2_mean_reduction_scales_with_batch() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 1, 8, 8)
    y = torch.randn(4, 1, 8, 8)
    summed = H2Loss2d(reduction="sum").abs(x, y)
    averaged = H2Loss2d(reduction="mean").abs(x, y)
    assert summed == pytest.approx(4.0 * averaged)


def test_h2_gradient_flows() -> None:
    loss = H2Loss2d()
    x = torch.rand(1, 1, 8, 8, requires_grad=True)
    y = torch.rand(1, 1, 8, 8)
    loss.rel(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _reload_fno_zre(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    for key in (
        "LOSS_L2_WEIGHT",
        "LOSS_H1_WEIGHT",
        "LOSS_L1_WEIGHT",
        "LOSS_H2_WEIGHT",
        "LOSS_RELATIVE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import fno_zre

    return importlib.reload(fno_zre)


def test_zre_default_loss_is_absolute_l2_h1(monkeypatch) -> None:
    fno_zre = _reload_fno_zre(monkeypatch, {})
    train_loss, eval_losses, description = fno_zre.build_losses()
    assert description == "0.5*absL2 + 0.5*absH1"
    assert set(eval_losses) == {"l2", "h1", "l1", "h2", "masked_mse"}
    torch.manual_seed(2)
    x = torch.rand(1, 1, 8, 8)
    y = torch.rand(1, 1, 8, 8)
    from neuralop import H1Loss, LpLoss

    expected = 0.5 * LpLoss(d=2, p=2).abs(x, y) + 0.5 * H1Loss(d=2).abs(x, y)
    assert train_loss(x, y) == pytest.approx(float(expected))


def test_zre_relative_l1_h2_loss(monkeypatch) -> None:
    fno_zre = _reload_fno_zre(
        monkeypatch,
        {
            "LOSS_L2_WEIGHT": "0.0",
            "LOSS_H1_WEIGHT": "0.0",
            "LOSS_L1_WEIGHT": "0.5",
            "LOSS_H2_WEIGHT": "0.5",
            "LOSS_RELATIVE": "1",
        },
    )
    train_loss, _, description = fno_zre.build_losses()
    assert description == "0.5*relL1 + 0.5*relH2"
    torch.manual_seed(3)
    x = torch.rand(2, 1, 8, 8)
    y = torch.rand(2, 1, 8, 8) + 0.25
    from neuralop import LpLoss

    expected = 0.5 * LpLoss(d=2, p=1).rel(x, y) + 0.5 * H2Loss2d().rel(x, y)
    assert train_loss(x, y) == pytest.approx(float(expected))


def test_zre_l2only_loss(monkeypatch) -> None:
    fno_zre = _reload_fno_zre(
        monkeypatch,
        {"LOSS_L2_WEIGHT": "1.0", "LOSS_H1_WEIGHT": "0.0"},
    )
    train_loss, _, description = fno_zre.build_losses()
    assert description == "1.0*absL2"
    torch.manual_seed(4)
    x = torch.rand(1, 1, 8, 8)
    y = torch.rand(1, 1, 8, 8)
    from neuralop import LpLoss

    assert train_loss(x, y) == pytest.approx(float(LpLoss(d=2, p=2).abs(x, y)))


def test_zre_all_zero_weights_rejected(monkeypatch) -> None:
    fno_zre = _reload_fno_zre(
        monkeypatch,
        {"LOSS_L2_WEIGHT": "0.0", "LOSS_H1_WEIGHT": "0.0"},
    )
    with pytest.raises(SystemExit):
        fno_zre.build_losses()
