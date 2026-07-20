"""Tests for the checkpoint-driven LocalFNO mode-weight diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_fno_3d import LocalFNO3d
from models_zre_2d import LocalFNO2d
from viz.localfno_mode_weights import (
    assemble_plane,
    extract_branches,
    main,
)


BRANCH_NAMES = (
    "encoder0",
    "encoder1",
    "bottleneck.0",
    "bottleneck.1",
    "decoder1",
    "decoder0",
)


def _small_3d() -> LocalFNO3d:
    torch.manual_seed(0)
    return LocalFNO3d(
        in_channels=2,
        base_width=8,
        local_window=(8, 8, 8),
        local_modes=(2, 2, 3),
        global_modes=(2, 2, 2),
        spectral_rank=4,
    )


def _small_2d() -> LocalFNO2d:
    torch.manual_seed(0)
    return LocalFNO2d(
        in_channels=3,
        base_width=8,
        local_window=(8, 8),
        local_modes=(2, 3),
        global_modes=(2, 2),
        spectral_rank=4,
    )


def test_extracts_all_six_branches_3d():
    branches = extract_branches(_small_3d())
    assert tuple(branch.name for branch in branches) == BRANCH_NAMES
    local = branches[0]
    assert local.windowed and local.n_modes == (2, 2, 3)
    assert set(local.profiles) == {"x", "y", "z", "shell"}
    # Negative quadrants reach |k_x| = m_x, so profiles span 0..m_x.
    assert local.profiles["x"].size == 3
    assert local.profiles["z"].size == 3
    bottleneck = branches[2]
    assert not bottleneck.windowed and bottleneck.n_modes == (2, 2, 2)


def test_extracts_all_six_branches_2d():
    branches = extract_branches(_small_2d())
    assert tuple(branch.name for branch in branches) == BRANCH_NAMES
    local = branches[0]
    assert local.n_modes == (2, 3)
    assert set(local.profiles) == {"x", "y", "shell"}
    assert local.profiles["x"].size == 3
    assert local.profiles["y"].size == 3


def test_plane_assembly_places_quadrants_3d():
    conv = _small_3d().encoder0.spectral
    for weight in (conv.weights1, conv.weights2, conv.weights3, conv.weights4):
        weight.data.zero_()
    # weights3 covers (+x, -y); its (0, 0) entry is k_x=0, k_y=-m_y.
    conv.weights3.data[0, 0, 0, 0, 0] = 1.0
    plane = assemble_plane(
        [
            w.detach().abs().square().mean(dim=(0, 1)).numpy()
            for w in (conv.weights1, conv.weights2, conv.weights3,
                      conv.weights4)
        ]
    )
    mx, my = conv.n_modes[0], conv.n_modes[1]
    assert plane.shape == (2 * mx, 2 * my, conv.n_modes[2])
    nonzero = np.argwhere(plane > 0)
    assert nonzero.tolist() == [[mx, 0, 0]]


def test_plane_assembly_places_blocks_2d():
    conv = _small_2d().encoder0.spectral
    conv.weights1.data.zero_()
    conv.weights2.data.zero_()
    # weights2 covers negative k_x; its row 0 is k_x = -m_x.
    conv.weights2.data[0, 0, 0, 1] = 1.0
    plane = assemble_plane(
        [
            w.detach().abs().square().mean(dim=(0, 1)).numpy()
            for w in (conv.weights1, conv.weights2)
        ]
    )
    mx, my = conv.n_modes
    assert plane.shape == (2 * mx, my)
    nonzero = np.argwhere(plane > 0)
    assert nonzero.tolist() == [[0, 1]]


def test_rejects_models_without_quadrant_branches():
    with pytest.raises(ValueError, match="quadrant"):
        extract_branches(torch.nn.Conv3d(2, 2, 1))


@pytest.mark.parametrize("task", ["3d", "zre"])
def test_end_to_end_from_checkpoint(tmp_path, task, monkeypatch):
    from modeling import TrainerModel

    model = TrainerModel(_small_3d() if task == "3d" else _small_2d())
    checkpoint = tmp_path / "best_model_state_dict.pt"
    torch.save(model.state_dict(), checkpoint)

    if task == "3d":
        settings = (
            ("LOCALFNO_WINDOW_X", "8"), ("LOCALFNO_WINDOW_Y", "8"),
            ("LOCALFNO_WINDOW_Z", "8"), ("LOCALFNO_MODES_X", "2"),
            ("LOCALFNO_MODES_Y", "2"), ("LOCALFNO_MODES_Z", "3"),
            ("N_MODES_X", "2"), ("N_MODES_Y", "2"), ("N_MODES_Z", "2"),
            ("LOCALFNO_BASE_WIDTH", "8"), ("LOCALFNO_SPECTRAL_RANK", "4"),
        )
    else:
        settings = (
            ("LOCALFNO_WINDOW_X", "8"), ("LOCALFNO_WINDOW_Y", "8"),
            ("LOCALFNO_MODES_X", "2"), ("LOCALFNO_MODES_Y", "3"),
            ("LOCALFNO_GLOBAL_MODES_X", "2"), ("LOCALFNO_GLOBAL_MODES_Y", "2"),
            ("LOCALFNO_BASE_WIDTH", "8"), ("LOCALFNO_SPECTRAL_RANK", "4"),
        )
    for name, value in settings:
        monkeypatch.setenv(name, value)

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
