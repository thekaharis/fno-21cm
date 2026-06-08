from __future__ import annotations

import numpy as np
import pytest
import torch

from metrics_21cm import compute_physical_metrics
from models_ufno import UFNOWrapped, pad_ufno_spatial


def test_transverse_padding_commutes_with_xy_transpose():
    cube = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    padded = pad_ufno_spatial(cube, pad_x=2, pad_y=1, pad_z=2)
    transposed = pad_ufno_spatial(
        cube.transpose(2, 3), pad_x=1, pad_y=2, pad_z=2
    ).transpose(2, 3)

    assert torch.equal(padded, transposed)
    assert torch.equal(padded[..., -1], padded[..., -2])


@pytest.mark.parametrize("variant", ["default", "anisotropic_z", "los1d"])
def test_all_ufno_variants_preserve_cube_shape(variant):
    model = UFNOWrapped(
        2, 2, 2,
        width=8,
        in_channels=2,
        out_channels=1,
        norm="groupnorm",
        unet_variant=variant,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 8, 8, 8))
    assert output.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(output).all()


def test_physical_metrics_are_exact_for_identical_fields():
    rng = np.random.default_rng(7)
    truth = rng.random((8, 8, 8))
    metrics = compute_physical_metrics(truth, truth.copy(), np.linspace(5, 12, 8))

    assert metrics["voxel_rmse"] == 0.0
    assert metrics["active_window_rmse"] == 0.0
    assert metrics["global_history_rmse"] == 0.0
    assert np.allclose(metrics["fourier"]["cross_correlation"], 1.0)
    assert (
        metrics["ionized_regions_truth"]
        == metrics["ionized_regions_pred"]
    )
