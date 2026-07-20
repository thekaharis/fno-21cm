"""Tests for the LocalSirenFNO variants (SIREN-generated quadrant weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_fno_3d import LocalFNO3d, QuadrantSpectralConv3dSiren
from models_zre_2d import LocalFNO2d, QuadrantSpectralConv2dSiren


SIREN_KWARGS = dict(siren_hidden_dim=16, siren_feature_dim=8)


def _small_3d(**overrides) -> LocalFNO3d:
    torch.manual_seed(0)
    return LocalFNO3d(
        in_channels=2,
        base_width=8,
        local_window=(8, 8, 8),
        local_modes=(2, 2, 3),
        global_modes=(2, 2, 2),
        spectral_rank=4,
        siren=True,
        **{**SIREN_KWARGS, **overrides},
    )


def _small_2d(**overrides) -> LocalFNO2d:
    torch.manual_seed(0)
    return LocalFNO2d(
        in_channels=3,
        base_width=8,
        local_window=(8, 8),
        local_modes=(2, 3),
        global_modes=(2, 2),
        spectral_rank=4,
        siren=True,
        **{**SIREN_KWARGS, **overrides},
    )


def test_forward_and_gradients_3d():
    model = _small_3d()
    x = torch.randn(1, 2, 16, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)
    assert torch.isfinite(y).all()
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    y.sum().backward()
    for branch in (model.encoder0, model.encoder1, model.bottleneck[0],
                   model.bottleneck[1], model.decoder1, model.decoder0):
        for net in (branch.spectral.real_weight, branch.spectral.imag_weight):
            grad = net.last.weight.grad
            assert grad is not None and float(grad.abs().sum()) > 0


def test_forward_and_gradients_2d():
    model = _small_2d()
    x = torch.randn(1, 3, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 16, 16)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for branch in (model.encoder0, model.bottleneck[0], model.decoder0):
        grad = branch.spectral.real_weight.last.weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0


def test_cached_weights_match_uncached():
    conv3 = _small_3d().encoder0.spectral
    x3 = torch.randn(3, 4, 8, 8, 8)
    cached = conv3.materialize_weights(device=x3.device, dtype=x3.dtype)
    assert torch.allclose(conv3(x3), conv3(x3, weights=cached))

    conv2 = _small_2d().encoder0.spectral
    x2 = torch.randn(3, 4, 8, 8)
    cached = conv2.materialize_weights(device=x2.device, dtype=x2.dtype)
    assert torch.allclose(conv2(x2), conv2(x2, weights=cached))


def test_spectral_weight_tensor_shapes():
    conv3 = _small_3d().encoder0.spectral
    tensors = conv3.spectral_weight_tensors()
    assert len(tensors) == 4
    assert all(tuple(t.shape) == (4, 4, 2, 2, 3) for t in tensors)
    assert all(t.is_complex() for t in tensors)

    conv2 = _small_2d().encoder0.spectral
    tensors = conv2.spectral_weight_tensors()
    assert len(tensors) == 2
    assert all(tuple(t.shape) == (4, 4, 2, 3) for t in tensors)


def test_dc_mode_decoupled_2d():
    conv = QuadrantSpectralConv2dSiren(4, (2, 3), hidden_dim=16,
                                       feature_dim=8)
    constant = torch.ones(2, 4, 16, 16)
    assert float(conv(constant).abs().max()) < 1e-5


def test_dense_checkpoints_still_load():
    """siren=False must keep the original architecture and state-dict keys."""
    torch.manual_seed(0)
    dense = LocalFNO3d(in_channels=2, base_width=8, local_window=(8, 8, 8),
                       local_modes=(2, 2, 3), global_modes=(2, 2, 2),
                       spectral_rank=4)
    assert hasattr(dense.encoder0.spectral, "weights1")
    reloaded = LocalFNO3d(in_channels=2, base_width=8, local_window=(8, 8, 8),
                          local_modes=(2, 2, 3), global_modes=(2, 2, 2),
                          spectral_rank=4)
    reloaded.load_state_dict(dense.state_dict())


def test_spectral_history_extraction_finds_all_branches():
    from util.spectral_weights import extract_spectral_weight_profiles

    profiles = extract_spectral_weight_profiles(_small_3d())
    names = [profile.layer for profile in profiles]
    assert len(names) == 6
    assert any("encoder0" in name for name in names)
    assert any("bottleneck" in name for name in names)


def test_mode_weight_viz_extraction_finds_all_branches():
    from viz.localfno_mode_weights import extract_branches

    branches = extract_branches(_small_3d())
    assert len(branches) == 6
    assert branches[0].n_modes == (2, 2, 3)


@pytest.mark.parametrize("task", ["3d", "zre"])
def test_mode_weight_viz_end_to_end_siren(tmp_path, task, monkeypatch):
    from modeling import TrainerModel
    from viz.localfno_mode_weights import main

    model = TrainerModel(_small_3d() if task == "3d" else _small_2d())
    checkpoint = tmp_path / "best_model_state_dict.pt"
    torch.save(model.state_dict(), checkpoint)

    # SIREN checkpoints encode no mode counts in their shapes; without run
    # metadata the diagnostic reads them from the training switches.
    monkeypatch.setenv("LOCALFNO_WINDOW_X", "8")
    monkeypatch.setenv("LOCALFNO_WINDOW_Y", "8")
    if task == "3d":
        monkeypatch.setenv("LOCALFNO_WINDOW_Z", "8")
        for axis, value in zip("XYZ", (2, 2, 3)):
            monkeypatch.setenv(f"LOCALFNO_MODES_{axis}", str(value))
            monkeypatch.setenv(f"N_MODES_{axis}", "2")
    else:
        for axis, value in zip("XY", (2, 3)):
            monkeypatch.setenv(f"LOCALFNO_MODES_{axis}", str(value))
            monkeypatch.setenv(f"LOCALFNO_GLOBAL_MODES_{axis}", "2")

    output_dir = main(
        [
            "--task", task,
            "--checkpoint", str(checkpoint),
            "--output-dir", str(tmp_path / "figures"),
        ]
    )
    produced = {path.name for path in output_dir.iterdir()}
    assert {
        "mode_weight_profiles.png",
        "mode_weight_planes.png",
        "mode_weight_cutoff_ratio.png",
        "mode_weight_profiles.csv",
    } <= produced


def test_mode_weight_viz_builds_siren_from_metadata(tmp_path, monkeypatch):
    from modeling import TrainerModel, load_checkpoint
    from util.run_metadata import write_run_metadata
    from viz.localfno_mode_weights import _build_model, extract_branches

    for variable in (
        "LOCALFNO_WINDOW_X", "LOCALFNO_WINDOW_Y", "LOCALFNO_WINDOW_Z",
        "LOCALFNO_MODES_X", "LOCALFNO_MODES_Y", "LOCALFNO_MODES_Z",
        "N_MODES_X", "N_MODES_Y", "N_MODES_Z",
    ):
        monkeypatch.delenv(variable, raising=False)

    # n_hidden=2 exercises the trunk-depth inference from checkpoint shapes.
    model = TrainerModel(_small_3d(siren_n_hidden=2))
    checkpoint = tmp_path / "best_model_state_dict.pt"
    torch.save(model.state_dict(), checkpoint)
    write_run_metadata(tmp_path, {
        "task": "3d",
        "model_config": {
            "kind": "localsirenfno",
            "modes": [2, 2, 2],
            "localfno_window": [8, 8, 8],
            "localfno_modes": [2, 2, 3],
        },
    })

    rebuilt = _build_model("3d", checkpoint)
    report = load_checkpoint(rebuilt, checkpoint)
    assert report.matched == report.total
    assert not report.missing and not report.unexpected
    branches = extract_branches(rebuilt)
    assert len(branches) == 6
    assert branches[0].n_modes == (2, 2, 3)


def test_modeling_registration(tmp_path):
    from modeling import ModelConfig, TrainerModel, build_3d_model, \
        load_checkpoint

    config = ModelConfig(
        kind="localsirenfno",
        modes=(2, 2, 2),
        localfno_window=(8, 8, 8),
        localfno_modes=(2, 2, 3),
        localfno_base_width=8,
        localfno_spectral_rank=4,
        siren_hidden_dim=16,
        siren_feature_dim=8,
    )
    assert config.default_checkpoint_dir.name == "checkpoints_3d_localsirenfno"
    assert "LocalSirenFNO" in config.describe()
    rebuilt = ModelConfig.from_dict(config.to_dict())
    assert rebuilt == config

    model = build_3d_model(config, in_channels=2)
    assert isinstance(model, LocalFNO3d)
    assert isinstance(model.encoder0.spectral, QuadrantSpectralConv3dSiren)

    wrapped = TrainerModel(model)
    checkpoint = tmp_path / "best_model_state_dict.pt"
    torch.save(wrapped.state_dict(), checkpoint)
    fresh = TrainerModel(build_3d_model(config, in_channels=2))
    report = load_checkpoint(fresh, checkpoint)
    assert report.matched == report.total
    assert not report.missing and not report.unexpected


def test_invalid_kind_rejected():
    from modeling import ModelConfig

    with pytest.raises(ValueError, match="localsirenfno"):
        ModelConfig(kind="bogus")
