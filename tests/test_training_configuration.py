from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fno_21cm_3d import LoggingTrainer, _build_h1_loss


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


def test_h1_redshift_derivative_does_not_wrap_endpoints() -> None:
    loss = _build_h1_loss()
    z = torch.arange(8, dtype=torch.float32)
    field = z.view(1, 1, 1, 1, 8).expand(1, 1, 4, 4, 8)

    terms, _ = loss.compute_terms(
        field,
        torch.zeros_like(field),
        quadrature=(1.0, 1.0, 1.0),
    )
    dz = terms[3].reshape_as(field)

    assert torch.allclose(dz, torch.ones_like(dz))
