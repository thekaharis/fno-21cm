"""Tests for the hybrid LocalWNO/FNO architecture."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from local_fno_3d import LocalFNO3d, QuadrantSpectralConv3d
from modeling import ModelConfig, build_3d_model
from models_zre_2d import LocalFNO2d, QuadrantSpectralConv2d
from wavelet_operator import HaarWaveletOperator


@pytest.mark.parametrize("ndim", [2, 3])
def test_haar_operator_identity_reconstruction(ndim: int) -> None:
    operator = HaarWaveletOperator(channels=3, ndim=ndim, levels=2)
    identity = torch.eye(3)
    with torch.no_grad():
        operator.lowpass_weight.copy_(identity)
        for weights in operator.detail_weights:
            weights.copy_(identity.expand_as(weights))
    x = torch.randn((2, 3) + (8,) * ndim)

    output = operator(x)

    assert torch.allclose(output, x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
def test_haar_operator_shape_and_gradients(ndim: int) -> None:
    operator = HaarWaveletOperator(channels=2, ndim=ndim, levels=2)
    x = torch.randn((1, 2) + (8,) * ndim, requires_grad=True)

    output = operator(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert operator.lowpass_weight.grad is not None
    assert all(weight.grad is not None for weight in operator.detail_weights)


def test_haar_operator_rejects_channel_mismatch() -> None:
    operator = HaarWaveletOperator(channels=2, ndim=2, levels=2)

    with pytest.raises(ValueError, match="expected 2 input channels, got 3"):
        operator(torch.randn(1, 3, 8, 8))


def test_localwno_2d_replaces_only_windowed_branches() -> None:
    model = LocalFNO2d(
        in_channels=3,
        base_width=4,
        local_window=(4, 4),
        local_modes=(2, 2),
        global_modes=(1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="wavelet",
        wavelet_levels=2,
    )
    output = model(torch.randn(1, 3, 9, 11))

    assert output.shape == (1, 1, 9, 11)
    for block in (model.encoder0, model.encoder1,
                  model.decoder1, model.decoder0):
        assert isinstance(block.spectral, HaarWaveletOperator)
    for block in model.bottleneck:
        assert isinstance(block.spectral, QuadrantSpectralConv2d)


def test_localwno_3d_forward_backward_and_configuration() -> None:
    env = {
        "MODEL_KIND": "localwno",
        "N_MODES_X": "1",
        "N_MODES_Y": "1",
        "N_MODES_Z": "2",
        "LOCALFNO_WINDOW_X": "4",
        "LOCALFNO_WINDOW_Y": "4",
        "LOCALFNO_WINDOW_Z": "4",
        "LOCALFNO_BASE_WIDTH": "4",
        "LOCALFNO_SPECTRAL_RANK": "2",
        "LOCALFNO_PATCH_CHUNK_SIZE": "4",
        "LOCALWNO_LEVELS": "2",
    }
    with patch.dict(os.environ, env, clear=True):
        config = ModelConfig.from_env()
    model = build_3d_model(config, in_channels=2)
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)

    output = model(x)
    output.mean().backward()

    assert isinstance(model, LocalFNO3d)
    assert output.shape == (1, 1, 8, 8, 8)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert config.default_checkpoint_dir.name == "checkpoints_3d_localwno"
    assert "wavelet=haar levels=2" in config.describe()
    assert ModelConfig.from_dict(config.to_dict()) == config
    for block in (model.encoder0, model.encoder1,
                  model.decoder1, model.decoder0):
        assert isinstance(block.spectral, HaarWaveletOperator)
        assert block.spectral.lowpass_weight.grad is not None
    for block in model.bottleneck:
        assert isinstance(block.spectral, QuadrantSpectralConv3d)


def test_wavelet_levels_must_fit_local_window() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        LocalFNO2d(
            in_channels=2,
            local_window=(4, 4),
            local_operator="wavelet",
            wavelet_levels=3,
        )
