from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectral_weights import (
    SpectralWeightHistory,
    extract_spectral_weight_profiles,
)
from visualize_spectral_weights import high_low_ratio, load_history
from visualize_spectral_weights import (
    plot_cutoff_ratios,
    plot_evolution,
    plot_profiles,
    write_csv,
)


class FakeFactorizedWeight:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self.tensor


class FakeSpectralConv(nn.Module):
    def __init__(self):
        super().__init__()
        tensor = torch.ones(2, 3, 4, 4, 3, dtype=torch.cfloat)
        tensor[..., 0, 2, 1] = 9
        self.weight = FakeFactorizedWeight(tensor)
        self.n_modes = (4, 4, 4)


class FakeFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral = FakeSpectralConv()


def test_extract_profiles_folds_centered_transverse_modes():
    profile = extract_spectral_weight_profiles(FakeFNO())[0]

    assert profile.layer == "spectral"
    assert profile.x.shape == (3,)
    assert profile.y.shape == (3,)
    assert profile.z.shape == (3,)
    assert profile.x[2] > profile.x[0]
    assert profile.y[0] > profile.y[1]
    assert profile.z[1] > profile.z[0]


def test_history_records_initial_and_epoch_snapshots(tmp_path):
    model = FakeFNO()
    path = tmp_path / "spectral_weight_history.npz"
    history = SpectralWeightHistory(path, model, reset=True)
    history.record(-1)
    model.spectral.weight.tensor[..., 0, 2, 1] = 18
    history.record(0)

    saved = load_history(path)
    np.testing.assert_array_equal(saved["epochs"], [-1, 0])
    assert saved["x"].shape[:2] == (2, 1)
    assert saved["x"][1, 0, 2] > saved["x"][0, 0, 2]


def test_high_low_ratio_detects_outer_mode_collapse():
    values = np.asarray([[[8.0, 4.0, 1.0, 0.5]]], dtype=np.float32)
    ratio = high_low_ratio(values)
    np.testing.assert_allclose(ratio, [[0.0625]])


def test_history_renders_all_diagnostics(tmp_path):
    model = FakeFNO()
    history_path = tmp_path / "spectral_weight_history.npz"
    recorder = SpectralWeightHistory(history_path, model, reset=True)
    recorder.record(-1)
    recorder.record(0)
    history = load_history(history_path)

    outputs = (
        tmp_path / "evolution.png",
        tmp_path / "profiles.png",
        tmp_path / "ratios.png",
        tmp_path / "history.csv",
    )
    plot_evolution(history, outputs[0])
    plot_profiles(history, outputs[1])
    plot_cutoff_ratios(history, outputs[2])
    write_csv(history, outputs[3])

    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
