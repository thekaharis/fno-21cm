from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from fno_21cm_3d import LoggingTrainer, _build_h1_loss, _seed_everything
from losses import IonizedWallRMSE, ScheduledWeightedLoss


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
