"""Tests for the pluggable local/global operator registry and the WHNO."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

import operators
from local_fno_3d import LocalFNO3d, QuadrantSpectralConv3d
from modeling import ModelConfig, OperatorSlots, build_3d_model
from models_zre_2d import LocalFNO2d, QuadrantSpectralConv2d
from operators import (
    ConvUNetOperator,
    WalshHadamardOperator,
    build_operator,
    crop_to_original,
    operator_hyperparameters,
    pad_to_operator_size,
    resolve_operator_name,
)
from wavelet_operator import HaarWaveletOperator


# ---------------------------------------------------------------- transform


@pytest.mark.parametrize("ndim", [2, 3])
def test_walsh_full_band_identity_is_the_identity(ndim: int) -> None:
    """Keeping every sequency mode with identity mixing must reconstruct."""
    operator = WalshHadamardOperator(3, ndim=ndim, modes=(8,) * ndim)
    with torch.no_grad():
        operator.weight.zero_()
        for channel in range(3):
            operator.weight[channel, channel] = 1.0
    x = torch.randn((2, 3) + (8,) * ndim)

    assert torch.allclose(operator(x), x, atol=1e-5, rtol=1e-5)


def test_walsh_basis_is_orthonormal_and_sequency_ordered() -> None:
    operator = WalshHadamardOperator(1, ndim=2, modes=(8, 8))
    basis = operator._basis(
        8, 8, device=torch.device("cpu"), dtype=torch.float32
    )

    assert torch.allclose(basis @ basis.T, torch.eye(8), atol=1e-6)
    assert torch.allclose(basis.abs(), torch.full((8, 8), 8**-0.5), atol=1e-6)
    # Sequency is the Walsh analogue of frequency: row k changes sign k times.
    sign_changes = [
        int((row[:-1] * row[1:] < 0).sum()) for row in basis
    ]
    assert sign_changes == list(range(8))


def test_walsh_truncation_is_a_projection() -> None:
    """Identity mixing on a truncated band is idempotent (P^2 = P)."""
    operator = WalshHadamardOperator(2, ndim=2, modes=(3, 5))
    with torch.no_grad():
        operator.weight.zero_()
        for channel in range(2):
            operator.weight[channel, channel] = 1.0
    x = torch.randn(1, 2, 8, 16)

    once = operator(x)
    assert torch.allclose(operator(once), once, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(once, x, atol=1e-3)


@pytest.mark.parametrize("ndim", [2, 3])
def test_walsh_shape_and_gradients(ndim: int) -> None:
    modes = (3, 5) if ndim == 2 else (3, 5, 6)
    sizes = (8, 16) if ndim == 2 else (8, 16, 8)
    operator = WalshHadamardOperator(4, ndim=ndim, modes=modes)
    x = torch.randn((2, 4) + sizes, requires_grad=True)

    output = operator(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert operator.weight.grad is not None
    assert torch.isfinite(operator.weight.grad).all()


def test_walsh_parameter_count_is_one_real_block_per_mode() -> None:
    operator = WalshHadamardOperator(16, ndim=3, modes=(6, 6, 12))

    assert operator.weight.numel() == 16 * 16 * 6 * 6 * 12
    assert not torch.is_complex(operator.weight)


def test_walsh_rejects_non_power_of_two_and_channel_mismatch() -> None:
    operator = WalshHadamardOperator(2, ndim=2, modes=(2, 2))

    with pytest.raises(ValueError, match="power of two"):
        operator(torch.randn(1, 2, 12, 8))
    with pytest.raises(ValueError, match="expected 2 input channels"):
        operator(torch.randn(1, 3, 8, 8))


def test_walsh_bases_stay_out_of_the_state_dict() -> None:
    """Cached bases are shape-dependent, so they must not be checkpointed."""
    operator = WalshHadamardOperator(2, ndim=2, modes=(2, 2))
    operator(torch.randn(1, 2, 8, 8))

    assert list(operator.state_dict()) == ["weight"]


@pytest.mark.parametrize("ndim", [2, 3])
def test_conv_unet_operator_preserves_shape(ndim: int) -> None:
    operator = ConvUNetOperator(6, ndim=ndim, depth=2)
    x = torch.randn((2, 6) + (8,) * ndim, requires_grad=True)

    output = operator(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ------------------------------------------------------------------ padding


def test_padding_lifts_a_non_power_of_two_shape_and_crops_back() -> None:
    x = torch.randn(1, 2, 35, 35, 64)

    padded, amounts = pad_to_operator_size(
        x, "hadamard", {}, ("circular", "circular", "replicate")
    )

    # The production bottleneck shape: both transverse axes need more than
    # their own extent in padding, which one F.pad call cannot deliver.
    assert tuple(padded.shape[-3:]) == (64, 64, 64)
    assert amounts == (29, 29, 0)
    assert torch.equal(crop_to_original(padded, amounts), x)


def test_padding_is_a_no_op_when_the_shape_already_fits() -> None:
    x = torch.randn(1, 2, 16, 32)

    padded, amounts = pad_to_operator_size(
        x, "hadamard", {}, ("circular", "circular")
    )

    assert amounts == (0, 0)
    assert padded is x


# ----------------------------------------------------------------- registry


def test_registry_resolves_aliases_and_rejects_unknown_names() -> None:
    assert resolve_operator_name("whno") == "hadamard"
    assert resolve_operator_name("WNO") == "wavelet"
    assert resolve_operator_name("unet") == "cnn"
    with pytest.raises(ValueError, match="unknown operator"):
        resolve_operator_name("fourier_but_better")


def test_registry_rejects_hyperparameters_an_operator_does_not_have() -> None:
    with pytest.raises(ValueError, match="no hyperparameter 'levels'"):
        operator_hyperparameters("hadamard", {"levels": 2})
    assert operator_hyperparameters("hadamard") == {"ordering": "sequency"}


def test_every_registered_operator_builds_and_preserves_shape() -> None:
    modes = (2, 2, 2)
    for name in operators.operator_names():
        hyperparameters = operator_hyperparameters(name, {})
        if name == "cnn":
            hyperparameters["depth"] = 1
        operator = build_operator(
            name, channels=4, ndim=3, modes=modes,
            hyperparameters=hyperparameters,
        )
        x = torch.randn(2, 4, 8, 8, 8)
        output = operator(x)
        assert output.shape == x.shape, name


# ------------------------------------------------------------ model slots


LOCAL_GLOBAL_PAIRS = [
    ("fourier", "fourier"),
    ("wavelet", "fourier"),
    ("hadamard", "fourier"),
    ("hadamard", "hadamard"),
    ("cnn", "cnn"),
    ("cnn", "fourier"),
    ("fourier", "cnn"),
    ("wavelet", "hadamard"),
]


def _slot_kwargs(name: str) -> dict | None:
    if name == "cnn":
        return {"depth": 1}
    if name == "wavelet":
        return {"levels": 2}
    return None


@pytest.mark.parametrize("local,global_", LOCAL_GLOBAL_PAIRS)
def test_3d_skeleton_accepts_every_slot_pairing(local, global_) -> None:
    model = LocalFNO3d(
        in_channels=2,
        base_width=4,
        local_window=(4, 4, 4),
        local_modes=(2, 2, 2),
        global_modes=(1, 1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator=local,
        global_operator=global_,
        local_operator_kwargs=_slot_kwargs(local),
        global_operator_kwargs=_slot_kwargs(global_),
    )
    # An odd shape exercises the whole-volume padding path in the global slot.
    x = torch.randn(1, 2, 9, 11, 13, requires_grad=True)

    output = model(x)
    output.mean().backward()

    assert output.shape == (1, 1, 9, 11, 13)
    assert torch.all((output > 0) & (output < 1))
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("local,global_", LOCAL_GLOBAL_PAIRS)
def test_2d_skeleton_accepts_every_slot_pairing(local, global_) -> None:
    model = LocalFNO2d(
        in_channels=3,
        base_width=4,
        local_window=(4, 4),
        local_modes=(2, 2),
        global_modes=(1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator=local,
        global_operator=global_,
        local_operator_kwargs=_slot_kwargs(local),
        global_operator_kwargs=_slot_kwargs(global_),
    )
    x = torch.randn(1, 3, 9, 11, requires_grad=True)

    output = model(x)
    output.mean().backward()

    assert output.shape == (1, 1, 9, 11)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_localwhno_replaces_only_the_windowed_branches() -> None:
    model = LocalFNO3d(
        in_channels=2,
        base_width=4,
        local_window=(4, 4, 4),
        local_modes=(2, 2, 2),
        global_modes=(1, 1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="hadamard",
    )

    for block in (model.encoder0, model.encoder1,
                  model.decoder1, model.decoder0):
        assert isinstance(block.spectral, WalshHadamardOperator)
        assert block.window_grid is not None
    for block in model.bottleneck:
        assert isinstance(block.spectral, QuadrantSpectralConv3d)


def test_cnn_local_slot_skips_windowing_and_rank_projection() -> None:
    model = LocalFNO2d(
        in_channels=3,
        base_width=4,
        local_window=(4, 4),
        local_modes=(2, 2),
        global_modes=(1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="cnn",
        local_operator_kwargs={"depth": 1},
    )

    assert model.local_windowed is False
    assert model.encoder0.window_grid is None
    # A convolution is already local; the rank bottleneck would only throttle
    # it, so it runs at full block width.
    assert isinstance(model.encoder0.in_projection, torch.nn.Identity)
    assert model.encoder0.spectral.channels == model.base_width
    for block in model.bottleneck:
        assert isinstance(block.spectral, QuadrantSpectralConv2d)


def test_cnn_local_slot_can_be_forced_through_the_window_grid() -> None:
    model = LocalFNO2d(
        in_channels=3,
        base_width=4,
        local_window=(4, 4),
        local_modes=(2, 2),
        global_modes=(1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="cnn",
        local_operator_kwargs={"depth": 1},
        local_windowed=True,
    )

    assert model.encoder0.window_grid is not None
    assert model(torch.randn(1, 3, 8, 8)).shape == (1, 1, 8, 8)


def test_local_window_must_suit_the_local_operator() -> None:
    with pytest.raises(ValueError, match="powers of two"):
        LocalFNO2d(in_channels=2, local_window=(12, 12),
                   local_operator="hadamard")
    with pytest.raises(ValueError, match="divisible by 8"):
        LocalFNO2d(in_channels=2, local_window=(4, 4),
                   local_operator="cnn", local_windowed=True,
                   local_operator_kwargs={"depth": 3})


def test_walsh_modes_may_span_the_whole_window() -> None:
    """Unlike the rFFT, sequency truncation is limited by the size itself."""
    model = LocalFNO2d(
        in_channels=2,
        base_width=4,
        local_window=(8, 8),
        local_modes=(8, 8),
        global_modes=(1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="hadamard",
    )

    assert model(torch.randn(1, 2, 8, 8)).shape == (1, 1, 8, 8)


# ------------------------------------------------------- legacy compatibility


def test_legacy_siren_flag_still_selects_both_siren_slots() -> None:
    model = LocalFNO3d(
        in_channels=2,
        base_width=4,
        local_window=(4, 4, 4),
        local_modes=(2, 2, 2),
        global_modes=(1, 1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        siren=True,
    )

    assert model.local_operator == "siren_fourier"
    assert model.global_operator == "siren_fourier"


def test_legacy_wavelet_levels_reaches_the_haar_operator() -> None:
    model = LocalFNO3d(
        in_channels=2,
        base_width=4,
        local_window=(8, 8, 8),
        local_modes=(2, 2, 2),
        global_modes=(1, 1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
        local_operator="wavelet",
        wavelet_levels=3,
    )

    assert isinstance(model.encoder0.spectral, HaarWaveletOperator)
    assert model.encoder0.spectral.levels == 3


def test_siren_and_wavelet_remain_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="siren cannot be combined"):
        LocalFNO3d(in_channels=2, siren=True, local_operator="wavelet")


def test_registry_rewrite_keeps_the_localfno_state_dict_keys() -> None:
    """Existing checkpoints must still load: no renamed or extra parameters."""
    legacy = LocalFNO3d(
        in_channels=2, base_width=4, local_window=(4, 4, 4),
        local_modes=(2, 2, 2), global_modes=(1, 1, 2), spectral_rank=2,
        patch_chunk_size=4,
    )
    explicit = LocalFNO3d(
        in_channels=2, base_width=4, local_window=(4, 4, 4),
        local_modes=(2, 2, 2), global_modes=(1, 1, 2), spectral_rank=2,
        patch_chunk_size=4, local_operator="fourier", global_operator="fourier",
    )

    assert list(legacy.state_dict()) == list(explicit.state_dict())
    assert "encoder0.spectral.weights1" in legacy.state_dict()
    assert "encoder0.in_projection.weight" in legacy.state_dict()

    # Optimizer state dicts are keyed by parameter position, and runs resume
    # from them, so the registration order must stay in_projection ->
    # operator -> out_projection.
    block = [
        key for key in legacy.state_dict()
        if key.startswith("encoder0.")
    ][:6]
    assert block == [
        "encoder0.in_projection.weight",
        "encoder0.spectral.weights1",
        "encoder0.spectral.weights2",
        "encoder0.spectral.weights3",
        "encoder0.spectral.weights4",
        "encoder0.out_projection.weight",
    ]


# ------------------------------------------------------------- configuration


def _env(**overrides) -> dict:
    env = {
        "N_MODES_X": "1", "N_MODES_Y": "1", "N_MODES_Z": "2",
        "LOCALFNO_WINDOW_X": "4", "LOCALFNO_WINDOW_Y": "4",
        "LOCALFNO_WINDOW_Z": "4",
        "LOCALFNO_MODES_X": "2", "LOCALFNO_MODES_Y": "2",
        "LOCALFNO_MODES_Z": "2",
        "LOCALFNO_BASE_WIDTH": "4", "LOCALFNO_SPECTRAL_RANK": "2",
        "LOCALFNO_PATCH_CHUNK_SIZE": "4",
    }
    env.update(overrides)
    return env


def test_localwhno_kind_builds_trains_and_round_trips() -> None:
    with patch.dict(os.environ, _env(MODEL_KIND="localwhno"), clear=True):
        config = ModelConfig.from_env()
    model = build_3d_model(config, in_channels=2)
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)

    output = model(x)
    output.mean().backward()

    assert isinstance(model, LocalFNO3d)
    assert output.shape == (1, 1, 8, 8, 8)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert config.default_checkpoint_dir.name == "checkpoints_3d_localwhno"
    assert "walsh=hadamard order=sequency" in config.describe()
    assert ModelConfig.from_dict(config.to_dict()) == config
    for block in (model.encoder0, model.encoder1,
                  model.decoder1, model.decoder0):
        assert isinstance(block.spectral, WalshHadamardOperator)
        assert block.spectral.weight.grad is not None
    for block in model.bottleneck:
        assert isinstance(block.spectral, QuadrantSpectralConv3d)


def test_localop_kind_pairs_operators_freely() -> None:
    env = _env(
        MODEL_KIND="localop",
        LOCAL_OPERATOR="whno",
        GLOBAL_OPERATOR="cnn",
        CNN_DEPTH="1",
        WHNO_ORDERING="natural",
    )
    with patch.dict(os.environ, env, clear=True):
        config = ModelConfig.from_env()
    model = build_3d_model(config, in_channels=2)

    assert config.local_operator == "hadamard"
    assert config.global_operator == "cnn"
    assert config.whno_ordering == "natural"
    assert config.default_checkpoint_dir.name == "checkpoints_3d_local_whno_cnn"
    assert ModelConfig.from_dict(config.to_dict()) == config
    assert model(torch.randn(1, 2, 8, 8, 8)).shape == (1, 1, 8, 8, 8)
    assert model.encoder0.spectral.ordering == "natural"
    assert isinstance(model.bottleneck[0].spectral, ConvUNetOperator)


def test_shorthand_kinds_refuse_a_contradicting_operator_pair() -> None:
    with pytest.raises(ValueError, match="use kind='localop'"):
        ModelConfig(kind="localwno", local_operator="hadamard")
    with patch.dict(
        os.environ, _env(MODEL_KIND="localwno", LOCAL_OPERATOR="hadamard"),
        clear=True,
    ):
        with pytest.raises(ValueError, match="MODEL_KIND=localop"):
            OperatorSlots.from_env("localwno")


def test_operator_slots_from_env_matches_the_model_config_path() -> None:
    env = _env(MODEL_KIND="localwhno", WHNO_ORDERING="sequency")
    with patch.dict(os.environ, env, clear=True):
        slots = OperatorSlots.from_env("localwhno")
        config = ModelConfig.from_env()

    assert (slots.local, slots.global_) == (
        config.local_operator, config.global_operator
    )
    assert slots.model_kwargs()["local_operator_kwargs"] == {
        "ordering": "sequency"
    }
    assert slots.checkpoint_tag == "localwhno"
    assert slots.metadata()["local_windowed"] is True


def test_config_rejects_a_local_window_the_operator_cannot_process() -> None:
    with patch.dict(
        os.environ,
        _env(MODEL_KIND="localwhno", LOCALFNO_WINDOW_X="12",
             LOCALFNO_WINDOW_Y="12", LOCALFNO_WINDOW_Z="12"),
        clear=True,
    ):
        with pytest.raises(ValueError, match="powers of two"):
            ModelConfig.from_env()
