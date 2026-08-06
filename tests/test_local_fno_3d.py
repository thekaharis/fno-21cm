from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from local_fno_3d import (
    LocalFNO3d,
    OverlapAddWindow3d,
    SpectralResidualBlock3d,
)
from modeling import ModelConfig, TrainerModel, build_3d_model, load_checkpoint
from util.spectral_weights import (
    SpectralWeightHistory,
    extract_spectral_weight_profiles,
)
from util.run_metadata import write_run_metadata


@pytest.mark.parametrize("offset", [(0, 0, 0), (1, 1, 2)])
@pytest.mark.parametrize("shape", [(7, 9, 11), (8, 8, 12)])
@pytest.mark.parametrize("field", ["constant", "impulse", "random"])
def test_overlap_add_identity_has_no_seams(offset, shape, field) -> None:
    if field == "constant":
        x = torch.full((1, 2, *shape), 3.25)
    elif field == "impulse":
        x = torch.zeros(1, 2, *shape)
        x[..., shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0
    else:
        x = torch.randn(1, 2, *shape)
    grid = OverlapAddWindow3d(
        (4, 4, 6),
        offset=offset,
        chunk_size=3,
    )
    output = grid.apply(x, lambda patches: patches)

    assert output.shape == x.shape
    assert torch.allclose(output, x, atol=1e-5, rtol=1e-5)


def test_window_grid_wraps_xy_and_replicates_z() -> None:
    x = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 1, 3, 4, 5)
    captured = []
    grid = OverlapAddWindow3d((2, 2, 2), chunk_size=100)

    def inspect(patches):
        captured.append(patches.detach().clone())
        return patches

    grid.apply(x, inspect)
    patches = captured[0]
    first = patches[0, 0]
    analysis = grid._window(device=x.device, dtype=x.dtype)
    expected = x[0, 0][
        torch.tensor([2, 0])[:, None, None],
        torch.tensor([3, 0])[None, :, None],
        torch.tensor([0, 0])[None, None, :],
    ] * analysis
    assert torch.equal(first, expected)


def test_local_spectral_block_preserves_odd_shape_and_backpropagates() -> None:
    block = SpectralResidualBlock3d(
        channels=4,
        modes=(2, 2, 2),
        spectral_rank=3,
        window_size=(4, 4, 4),
        offset=(1, 1, 1),
        patch_chunk_size=2,
    )
    x = torch.randn(1, 4, 7, 9, 11, requires_grad=True)
    output = block(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("weights1", "weights2", "weights3", "weights4"):
        gradient = getattr(block.spectral, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def _small_model() -> LocalFNO3d:
    return LocalFNO3d(
        in_channels=2,
        base_width=4,
        local_window=(4, 4, 4),
        local_modes=(2, 2, 2),
        global_modes=(1, 1, 2),
        spectral_rank=2,
        patch_chunk_size=4,
    )


def test_local_fno_preserves_shape_bounds_output_and_profiles_weights() -> None:
    model = _small_model()
    x = torch.randn(1, 2, 9, 11, 13, requires_grad=True)
    output = model(x)
    output.mean().backward()
    profiles = extract_spectral_weight_profiles(model)

    assert output.shape == (1, 1, 9, 11, 13)
    assert torch.all(output > 0)
    assert torch.all(output < 1)
    assert len(profiles) == 6
    assert any(profile.x.shape != profiles[0].x.shape for profile in profiles)


def test_local_fno_config_environment_metadata_and_checkpoint_round_trip() -> None:
    env = {
        "MODEL_KIND": "localfno",
        "N_MODES_X": "1",
        "N_MODES_Y": "1",
        "N_MODES_Z": "2",
        "LOCALFNO_WINDOW_X": "4",
        "LOCALFNO_WINDOW_Y": "4",
        "LOCALFNO_WINDOW_Z": "4",
        "LOCALFNO_MODES_X": "2",
        "LOCALFNO_MODES_Y": "2",
        "LOCALFNO_MODES_Z": "2",
        "LOCALFNO_BASE_WIDTH": "4",
        "LOCALFNO_SPECTRAL_RANK": "2",
        "LOCALFNO_PATCH_CHUNK_SIZE": "3",
    }
    with patch.dict(os.environ, env, clear=True):
        config = ModelConfig.from_env()
    restored = ModelConfig.from_dict(config.to_dict())
    assert restored == config
    assert config.default_checkpoint_dir.name == "checkpoints_3d_localfno"

    torch.manual_seed(19)
    source = TrainerModel(build_3d_model(config, in_channels=2))
    sample = torch.randn(1, 2, 8, 8, 8)
    expected = source(x=sample)
    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        torch.save(source.state_dict(), checkpoint.name)
        target = TrainerModel(build_3d_model(config, in_channels=2))
        report = load_checkpoint(target, checkpoint.name)
        actual = target(x=sample)
    assert report.matched == report.total
    assert torch.equal(actual, expected)


def test_mixed_local_and_global_profiles_can_be_recorded(tmp_path) -> None:
    history_path = tmp_path / "history.npz"
    history = SpectralWeightHistory(history_path, _small_model(), reset=True)
    history.record(-1)
    history.record(0)

    import numpy as np
    with np.load(history_path, allow_pickle=False) as values:
        assert values["epochs"].tolist() == [-1, 0]
        assert values["x"].shape[1] == 6
        assert np.isnan(values["x"]).any()


def test_visualization_loader_uses_explicit_checkpoint_metadata(tmp_path) -> None:
    from viz.visualize_3d import load_model as load_visualization_model

    config = ModelConfig(
        kind="localfno",
        modes=(1, 1, 2),
        localfno_window=(4, 4, 4),
        localfno_modes=(2, 2, 2),
        localfno_base_width=4,
        localfno_spectral_rank=2,
        localfno_patch_chunk_size=3,
    )
    source = TrainerModel(build_3d_model(config, in_channels=2))
    checkpoint = tmp_path / "best_model_state_dict.pt"
    torch.save(source.state_dict(), checkpoint)
    write_run_metadata(tmp_path, {"model_config": config.to_dict()})

    loaded = load_visualization_model(
        in_channels=2,
        checkpoint=checkpoint,
        device="cpu",
    )
    output = loaded(x=torch.randn(1, 2, 8, 8, 8))
    assert output.shape == (1, 1, 8, 8, 8)


@pytest.mark.skipif(
    os.environ.get("RUN_LOCALFNO_PRODUCTION_SHAPE") != "1",
    reason="set RUN_LOCALFNO_PRODUCTION_SHAPE=1 on an H200 smoke-test node",
)
def test_production_shape_forward() -> None:
    model = LocalFNO3d(in_channels=13).eval().cuda()
    with torch.no_grad():
        output = model(torch.randn(1, 13, 140, 140, 256, device="cuda"))
    assert output.shape == (1, 1, 140, 140, 256)
    assert torch.isfinite(output).all()
