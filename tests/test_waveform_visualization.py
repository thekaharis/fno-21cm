"""All-branch reports use saved deployment geometry, not guessed resolutions."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from modeling import ModelConfig, TrainerModel, build_model
from viz.learned_waveforms import bank_geometry, bank_names, render_all


def fixture_run(ndim=2, *, windowed=True, modes=2, local="learned_waveform", global_="learned_waveform", transform="tied"):
    config = ModelConfig(kind="localop", ndim=ndim, local_operator=local, global_operator=global_,
                         localfno_base_width=4, localfno_spectral_rank=2,
                         localfno_window=(4,) * ndim, localfno_modes=(modes,) * ndim,
                         modes=(modes,) * ndim, local_windowed=windowed,
                         waveform_local_bins=7, waveform_global_bins=9, waveform_transform=transform)
    torch.manual_seed(23)
    state = TrainerModel(build_model(config, 2)).state_dict()
    metadata = {"task": "3d" if ndim == 3 else "zre", "model_config": config.to_dict(),
                "input_features": {"spatial_shape": [13] * ndim}}
    return state, metadata


@pytest.mark.parametrize("ndim", [2, 3])
def test_shapes_branches_and_shared_bottleneck_from_metadata(ndim):
    state, metadata = fixture_run(ndim)
    geometry = bank_geometry(state, metadata)
    assert [g["branch"] for g in geometry] == ["encoder0", "encoder1", "bottleneck.0", "decoder1", "decoder0"]
    assert [g["shape"] for g in geometry] == [(4,) * ndim, (4,) * ndim, (3,) * ndim, (4,) * ndim, (4,) * ndim]
    assert geometry[2]["shared_blocks"] == ["bottleneck.0", "bottleneck.1"]
    wrapped = {"module." + k: v for k, v in state.items()}
    assert all(g["bank"].startswith("module.fno.") for g in bank_geometry(wrapped, metadata))


def test_unwindowed_shapes_follow_pooling_and_override_is_explicit():
    state, metadata = fixture_run(windowed=False)
    assert [g["shape"] for g in bank_geometry(state, metadata)] == [(13, 13), (6, 6), (3, 3), (6, 6), (13, 13)]
    assert bank_geometry(state, metadata, input_shape=(17, 19))[2]["shape"] == (4, 4)


def test_old_metadata_requires_input_shape_but_local_only_needs_no_input_grid():
    state, metadata = fixture_run()
    metadata.pop("input_features")
    with pytest.raises(ValueError, match="--input-shape"):
        bank_geometry(state, metadata)
    assert len(bank_geometry(state, metadata, input_shape=(12, 12))) == 5
    with pytest.raises(ValueError, match="2 dimensions"):
        bank_geometry(state, metadata, input_shape=(12, 12, 12))
    state, metadata = fixture_run(global_="fourier")
    metadata.pop("input_features")
    assert len(bank_geometry(state, metadata)) == 4


def test_dc_only_banks_are_identified_using_operator_metadata():
    state, metadata = fixture_run(modes=1)
    assert len(bank_names(state, metadata)) == 5
    assert len(bank_geometry(state, metadata)) == 5


@pytest.mark.parametrize("ndim", [2, 3])
def test_overviews_details_and_numerical_exports(tmp_path, ndim):
    state, metadata = fixture_run(ndim)
    manifest = render_all(state, metadata, tmp_path, max_modes=2, checkpoint="example.pt")
    assert manifest["spatial_dimensions"] == ndim
    assert len(manifest["banks"]) == 5
    assert (tmp_path / "overview_bins.png").stat().st_size > 1000
    assert (tmp_path / "overview_modes.png").stat().st_size > 1000
    for entry in manifest["banks"]:
        assert (tmp_path / entry["figure"]).stat().st_size > 1000
        with np.load(tmp_path / entry["arrays"]) as arrays:
            for axis, n in enumerate(entry["shape"]):
                u = arrays[f"axis{axis}_orthonormal"]
                assert u.shape == (n, 2)
                np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    assert json.loads((tmp_path / "manifest.json").read_text())["checkpoint"] == "example.pt"


def test_cli_checkpoint_directory_and_old_list_mode(tmp_path):
    state, metadata = fixture_run()
    torch.save(state, tmp_path / "best_model_state_dict.pt")
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))
    result = subprocess.run([sys.executable, "-m", "viz.learned_waveforms", "--checkpoint-dir", str(tmp_path),
                             "--out-dir", str(tmp_path / "report"), "--max-modes", "2"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "5 bank plots" in result.stdout
    assert (tmp_path / "report" / "manifest.json").is_file()
    result = subprocess.run([sys.executable, "-m", "viz.learned_waveforms", "--checkpoint",
                             str(tmp_path / "best_model_state_dict.pt")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 5


def test_dc_only_report(tmp_path):
    state, metadata = fixture_run(modes=1)
    manifest = render_all(state, metadata, tmp_path, max_modes=2)
    assert len(manifest["banks"]) == 5


@pytest.mark.parametrize("modes", [1, 3])
def test_separate_banks_export_both_roles_without_overwriting(tmp_path, modes):
    state, metadata = fixture_run(modes=modes, transform="separate")
    manifest = render_all(state, metadata, tmp_path, max_modes=2)
    assert len(manifest["banks"]) == 10
    assert len({entry["figure"] for entry in manifest["banks"]}) == 10
    assert len({entry["arrays"] for entry in manifest["banks"]}) == 10
    for entry in manifest["banks"]:
        assert entry["role"] in {"analysis", "synthesis"}
        assert entry["role"] in entry["figure"]
        assert (tmp_path / entry["figure"]).stat().st_size > 1000
        with np.load(tmp_path / entry["arrays"]) as arrays:
            for axis in range(2):
                u = arrays[f"axis{axis}_orthonormal"]
                np.testing.assert_allclose(u.T @ u, np.eye(modes), atol=1e-12)
