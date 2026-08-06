from __future__ import annotations

import pytest
import torch

from modeling import ModelConfig, build_model
from legacy.arch.siren_fno_3d import SirenFNO3d, SpectralConv3dSiren
from util.spectral_weights import extract_spectral_weight_profiles


def test_signed_quadrant_coordinates_match_rfft_layout() -> None:
    layer = SpectralConv3dSiren(2, 3, n_modes=(2, 2, 3), hidden_dim=8)
    coordinates = layer._quadrant_coordinates(
        (8, 10, 12),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(coordinates) == 4
    assert all(grid.shape == (2, 2, 3, 3) for grid in coordinates)
    assert torch.equal(coordinates[0][:, 0, 0, 0], torch.tensor([0.0, 0.5]))
    assert torch.equal(coordinates[1][:, 0, 0, 0], torch.tensor([-1.0, -0.5]))
    assert torch.equal(coordinates[2][0, :, 0, 1], torch.tensor([-1.0, -0.5]))
    assert torch.equal(
        coordinates[0][0, 0, :, 2],
        torch.tensor([0.0, 0.5, 1.0]),
    )


def test_spectral_layer_preserves_shape_and_backpropagates() -> None:
    layer = SpectralConv3dSiren(
        3,
        4,
        n_modes=(2, 2, 3),
        hidden_dim=8,
        feature_dim=8,
    )
    x = torch.randn(2, 3, 8, 10, 12, requires_grad=True)
    output = layer(x)

    assert output.shape == (2, 4, 8, 10, 12)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.real_weight.last.weight.grad is not None


def test_spectral_layer_rejects_modes_larger_than_fft_grid() -> None:
    layer = SpectralConv3dSiren(2, 2, n_modes=(5, 2, 2), hidden_dim=8)
    with pytest.raises(ValueError, match="exceeds FFT limits"):
        layer(torch.randn(1, 2, 8, 8, 8))


def test_siren_fno_preserves_unpadded_lightcone_shape() -> None:
    model = SirenFNO3d(
        n_modes=(2, 2, 3),
        hidden_channels=4,
        in_channels=2,
        out_channels=1,
        n_layers=2,
        padding=(0, 0, 2),
        siren_hidden_dim=8,
        siren_feature_dim=8,
    )
    output = model(torch.randn(1, 2, 8, 10, 12))

    assert output.shape == (1, 1, 8, 10, 12)
    assert torch.isfinite(output).all()


def test_temperature_scaled_sigmoid_bounds_output() -> None:
    model = SirenFNO3d(
        n_modes=(2, 2, 2),
        hidden_channels=4,
        in_channels=2,
        n_layers=1,
        padding=(0, 0, 0),
        siren_hidden_dim=8,
        siren_feature_dim=8,
        output_sigmoid=True,
        sigmoid_temperature=2.0,
    )
    output = model(torch.randn(1, 2, 8, 8, 8))

    assert torch.all(output > 0)
    assert torch.all(output < 1)


def test_sigmoid_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sigmoid_temperature"):
        SirenFNO3d(
            n_modes=(2, 2, 2),
            hidden_channels=4,
            in_channels=2,
            sigmoid_temperature=0.0,
        )


def test_model_factory_builds_sirenfno_and_profiles_generated_weights() -> None:
    config = ModelConfig(
        kind="sirenfno",
        modes=(2, 2, 2),
        hidden_channels=4,
        n_layers=1,
        siren_hidden_dim=8,
        siren_feature_dim=8,
        siren_padding=(0, 0, 0),
    )
    model = build_model(config, in_channels=2)
    output = model(torch.randn(1, 2, 8, 8, 8))
    profiles = extract_spectral_weight_profiles(model)

    assert output.shape == (1, 1, 8, 8, 8)
    assert len(profiles) == 1
    assert profiles[0].x.shape == (3,)
    assert profiles[0].z.shape == (2,)
