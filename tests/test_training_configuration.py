from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from fno_21cm_3d import (
    LoggingTrainer,
    _build_h1_loss,
    _loss_mode,
    _seed_everything,
)
from losses import IonizedWallRMSE, ScheduledWeightedLoss, WeightedLoss


def test_logging_trainer_rejects_trainer_owned_ddp() -> None:
    with pytest.raises(ValueError, match="single DDP wrapper"):
        LoggingTrainer(
            model=nn.Identity(),
            n_epochs=1,
            device="cpu",
            use_distributed=True,
        )


def test_h1_uses_periodic_xy_and_nonperiodic_redshift() -> None:
    loss = _build_h1_loss()
    assert loss.periodic_in_x is True
    assert loss.periodic_in_y is True
    assert loss.periodic_in_z is False
    assert tuple(loss.measure) == (1.0, 1.0, 1.0)


def test_h1_redshift_derivative_uses_only_centered_interior_cells() -> None:
    loss = _build_h1_loss()
    z = torch.arange(8, dtype=torch.float32)
    field = z.view(1, 1, 1, 1, 8).expand(1, 1, 4, 4, 8)

    terms, _ = loss.compute_terms(
        field,
        torch.zeros_like(field),
        quadrature=(1.0, 1.0, 1.0),
    )
    dz = terms[3].reshape(1, 1, 4, 4, 6)

    assert torch.allclose(dz, torch.ones_like(dz))


def test_h1_los_derivative_has_no_endpoint_stencil() -> None:
    loss = _build_h1_loss()
    prediction = torch.zeros(1, 1, 4, 4, 8)
    target = torch.zeros_like(prediction)
    prediction[..., 0] = 1.0
    prediction[..., -1] = -1.0

    prediction_terms, target_terms = loss.compute_terms(
        prediction,
        target,
        quadrature=(1.0, 1.0, 1.0),
    )

    # Endpoint errors remain in the H1 value term. They influence only the
    # adjacent centered stencil; there are no one-sided derivative samples
    # located at the endpoints.
    assert not torch.equal(prediction_terms[0], target_terms[0])
    dz = prediction_terms[3].reshape(1, 1, 4, 4, 6)
    expected = torch.zeros_like(dz)
    expected[..., 0] = -0.5
    expected[..., -1] = -0.5
    assert torch.equal(dz, expected)


def test_scheduled_loss_ramps_only_h1_term() -> None:
    constant = lambda out, y, **kwargs: out.new_tensor(1.0)
    loss = ScheduledWeightedLoss(
        (0.5, constant),
        (0.5, constant),
        (0.0, constant),
        (0.25, constant),
        warmup_terms=(1,),
        warmup_epochs=5,
    )
    prediction = torch.zeros(1)

    loss.set_epoch(0)
    assert loss.active_weights == (0.5, 0.0, 0.0, 0.25)
    assert loss(prediction, prediction).item() == pytest.approx(0.75)

    loss.set_epoch(2)
    assert loss.active_weights == (0.5, 0.2, 0.0, 0.25)
    assert loss(prediction, prediction).item() == pytest.approx(0.95)

    loss.set_epoch(5)
    assert loss.active_weights == (0.5, 0.5, 0.0, 0.25)
    assert loss(prediction, prediction).item() == pytest.approx(1.25)


def test_relative_h1_is_zero_for_identical_fields() -> None:
    loss = _build_h1_loss()
    field = torch.rand(2, 1, 4, 4, 8)
    assert loss.rel(field, field).item() == pytest.approx(0.0)


def test_relative_h1_is_scale_invariant() -> None:
    loss = _build_h1_loss()
    torch.manual_seed(0)
    prediction = torch.rand(2, 1, 4, 4, 8)
    target = torch.rand(2, 1, 4, 4, 8)

    base = loss.rel(prediction, target).item()
    scaled = loss.rel(100.0 * prediction, 100.0 * target).item()

    assert base > 0
    assert scaled == pytest.approx(base, rel=1e-5)


def test_relative_h1_matches_abs_over_target_norm() -> None:
    loss = _build_h1_loss()
    torch.manual_seed(1)
    prediction = torch.rand(1, 1, 4, 4, 8)
    target = torch.rand(1, 1, 4, 4, 8)

    absolute = loss.abs(prediction, target).item()
    target_norm = loss.abs(torch.zeros_like(target), target).item()

    assert loss.rel(prediction, target).item() == pytest.approx(
        absolute / target_norm, rel=1e-5
    )


def test_weighted_loss_accumulates_per_term_means() -> None:
    loss = WeightedLoss(
        (0.5, lambda out, y, **_: (out - y).abs().mean()),
        (0.0, lambda out, y, **_: out.new_tensor(99.0)),
        term_names=("l2", "h1"),
    )
    out, y = torch.tensor([2.0]), torch.tensor([1.0])

    loss(out, y)
    loss(out + 2.0, y)
    means = loss.pop_term_means()

    assert means == {"l2": pytest.approx(2.0)}  # mean of |1| and |3|
    assert "h1" not in means  # zero-weight term never evaluated
    assert loss.pop_term_means() == {}  # accumulator was reset


def test_scheduled_loss_term_logging_respects_warmup() -> None:
    constant = lambda out, y, **_: out.new_tensor(1.0)
    loss = ScheduledWeightedLoss(
        (0.5, constant),
        (0.5, constant),
        warmup_terms=(1,),
        warmup_epochs=5,
        term_names=("l2", "h1"),
    )
    prediction = torch.zeros(1)

    loss.set_epoch(0)
    loss(prediction, prediction)
    assert set(loss.pop_term_means()) == {"l2"}

    loss.set_epoch(5)
    loss(prediction, prediction)
    assert set(loss.pop_term_means()) == {"l2", "h1"}


def test_loss_mode_env_validation(monkeypatch) -> None:
    monkeypatch.setenv("LOSS_H1_MODE", "Relative")
    assert _loss_mode("LOSS_H1_MODE") == "relative"
    monkeypatch.delenv("LOSS_H1_MODE")
    assert _loss_mode("LOSS_H1_MODE") == "absolute"
    monkeypatch.setenv("LOSS_H1_MODE", "sometimes")
    with pytest.raises(ValueError, match="LOSS_H1_MODE"):
        _loss_mode("LOSS_H1_MODE")


def test_ionized_wall_loss_penalizes_only_excess_on_ionized_side() -> None:
    target = torch.zeros(1, 1, 7, 7, 3)
    target[..., 3:, 3:, :] = 1.0
    loss = IonizedWallRMSE(band_kernel_size=3, threshold=0.5)

    prediction = target.clone()
    prediction[..., 2, 3, 1] = 0.4
    prediction[..., 1, 1, 1] = 0.9
    prediction[..., 2, 2, 1] = -0.5

    mask = loss.wall_mask(target)
    assert mask[..., 2, 3, 1]
    assert not mask[..., 1, 1, 1]
    assert loss(prediction, target).item() > 0

    underprediction = target.clone()
    underprediction[..., 2, 3, 1] = -0.5
    assert loss(underprediction, target).item() == pytest.approx(0.0)


def test_ionized_wall_mask_wraps_periodic_xy_but_not_z() -> None:
    target = torch.zeros(1, 1, 5, 5, 3)
    target[..., 0, 0, 1] = 1.0
    loss = IonizedWallRMSE(band_kernel_size=3)
    mask = loss.wall_mask(target)

    assert mask[..., -1, 0, 1]
    assert mask[..., 0, -1, 1]
    assert not mask[..., 0, 0, 0]


def test_seed_everything_repeats_python_numpy_and_torch() -> None:
    _seed_everything(123)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )
    _seed_everything(123)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
