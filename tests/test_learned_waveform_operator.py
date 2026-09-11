"""Numerical and integration contracts of the learned orthonormal transform."""

from __future__ import annotations

import copy
import io
import math
import os
from unittest.mock import patch

import pytest
import torch

from learned_waveform_operator import (
    LearnedWaveformOperator, WaveformBank, waveform_diagnostics,
    waveform_parameter_groups,
)
from local_fno_3d import LocalFNO3d, SpectralResidualBlock3d
from modeling import ModelConfig, TrainerModel, build_model
from models_zre_2d import LocalFNO2d, SpectralResidualBlock2d
from operators import build_operator, validate_operator


def identity_mix(op):
    with torch.no_grad():
        op.weight.zero_()
        if op.phase_weight is not None:
            op.phase_weight.zero_()
        for channel in range(op.channels):
            op.weight[channel, channel] = 1


@pytest.mark.parametrize("shape,modes", [((16, 32), (6, 12)), ((35, 35, 64), (16, 16, 16))])
def test_orthonormal_columns_dc_and_retained_space_roundtrip(shape, modes):
    torch.manual_seed(13)
    op = LearnedWaveformOperator(2, len(shape), modes).double()
    bases = op.materialize_transform(shape, device="cpu", dtype=torch.float64)
    for n, m, u in zip(shape, modes, bases):
        torch.testing.assert_close(u.T @ u, torch.eye(m, dtype=u.dtype), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(u[:, 0], torch.full((n,), n ** -.5, dtype=u.dtype))
    c = torch.randn(1, 2, *modes, dtype=torch.float64)
    x = op.contract(c, bases, analysis=False)
    torch.testing.assert_close(op.contract(x, bases, analysis=True), c, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_complete_odd_grid_identity_and_truncated_projection(ndim):
    op = LearnedWaveformOperator(2, ndim, (5,) * ndim, bins=7).double()
    identity_mix(op)
    x = torch.randn((1, 2) + (5,) * ndim, dtype=torch.float64)
    torch.testing.assert_close(op(x), x, atol=1e-12, rtol=1e-12)
    truncated = LearnedWaveformOperator(2, ndim, (3,) * ndim, bins=7).double()
    identity_mix(truncated)
    projected = truncated(x)
    assert not torch.allclose(projected, x)
    torch.testing.assert_close(truncated(projected), projected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(truncated(torch.ones_like(x)), torch.ones_like(x))


def test_bin_gradients_through_resampling_qr_analysis_and_synthesis():
    torch.manual_seed(5)
    op = LearnedWaveformOperator(2, 2, (4, 3), bins=7).double()
    x = torch.randn(1, 2, 9, 7, dtype=torch.float64)
    table = op.bank.tables["0"]
    assert torch.autograd.gradcheck(
        lambda t: torch.func.functional_call(op, {"bank.tables.0": t}, (x,)),
        (table,), fast_mode=True, atol=1e-5, rtol=1e-4,
    )
    op(x).square().mean().backward()
    for p in op.bank.parameters():
        assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert torch.isfinite(op.weight.grad).all()


def test_all_real_first_step_learning_and_no_stale_graph_after_updates():
    torch.manual_seed(8)
    op = LearnedWaveformOperator(2, 2, (4, 4), bins=7)
    optim = torch.optim.Adam(waveform_parameter_groups(op, lr=.01, weight_decay=.1))
    before = {name: p.detach().clone() for name, p in op.bank.named_parameters()}
    x = torch.randn(2, 2, 9, 9)
    for step in range(2):
        optim.zero_grad()
        output = op(x)
        assert not output.is_complex()
        output.square().mean().backward()
        optim.step()
        for name, p in op.bank.named_parameters():
            assert not torch.equal(before[name], p), (step, name)
    assert all(not p.is_complex() for p in op.parameters())
    assert all(not value.requires_grad for value in op.bank.last_diagnostics.values())
    assert all(not value.requires_grad for value in op.bank._resampler_cache.values())
    assert list(op.state_dict()) == ["weight", "phase_weight", "bank.tables.0", "bank.tables.1"]


def test_resampler_filters_unresolved_harmonics_before_dilation():
    bank = WaveformBank(2, (9, 3), bins=31).double()
    centers = (torch.arange(31, dtype=torch.float64) + .5) / 31
    table = torch.sin(2 * math.pi * 3 * centers)
    samples = bank._resampler(16, 9, device=table.device, dtype=table.dtype) @ table
    # k=4: harmonic 3 would alias at frequency 12 on N=16. It is removed.
    torch.testing.assert_close(samples[:, 6:], torch.zeros_like(samples[:, 6:]), atol=1e-14, rtol=0)
    # k=1 still contains harmonic 3, so this is not a zero resampler.
    assert samples[:, 0].norm() > 1


@pytest.mark.parametrize("value", [0., 1., float("nan"), float("inf")])
def test_invalid_candidates_fail_before_qr(value):
    bank = WaveformBank(2, (3, 3), bins=7)
    with torch.no_grad():
        bank.tables["0"].fill_(value)
    with pytest.raises(RuntimeError, match="waveform axis 0"):
        bank.materialize_transform((8, 8), device="cpu", dtype=torch.float32)


def test_nyquist_shape_hyperparameter_and_input_validation():
    with pytest.raises(ValueError, match="Nyquist excluded"):
        validate_operator("waveform", (8, 8), (8, 3))
    with pytest.raises(ValueError, match="odd"):
        build_operator("waveform", channels=2, ndim=2, modes=(3, 3), hyperparameters={"bins": 8})
    op = LearnedWaveformOperator(2, 2, (3, 3))
    with pytest.raises(ValueError, match="expected"):
        op(torch.randn(1, 1, 8, 8))
    with pytest.raises(ValueError, match="real floating"):
        op(torch.randn(1, 2, 8, 8, dtype=torch.complex64))
    with pytest.raises(ValueError, match="does not match"):
        op(torch.randn(1, 2, 8, 8), transform=(torch.ones(7, 3), torch.ones(8, 3)))


def test_dc_only_axes_have_no_unused_parameters():
    op = LearnedWaveformOperator(2, 2, (1, 3), bins=7)
    assert list(op.bank.tables) == ["1"]
    op(torch.randn(1, 2, 5, 5)).square().sum().backward()
    assert all(p.grad is not None for p in op.parameters())


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("transform", ["tied", "separate"])
def test_patch_chunking_preserves_outputs_gradients_and_prepares_once(ndim, transform):
    block_class = SpectralResidualBlock2d if ndim == 2 else SpectralResidualBlock3d
    block = block_class(4, (3,) * ndim, 2, window_size=(4,) * ndim,
                        patch_chunk_size=1, operator="waveform",
                        operator_kwargs={"bins": 7, "transform": transform}).double()
    with torch.no_grad():
        block.spectral.phase_weight.normal_(std=.1)
    other = copy.deepcopy(block)
    other.window_grid.chunk_size = 1000
    x = torch.randn((1, 4) + (5,) * ndim, dtype=torch.float64)
    with patch.object(block.spectral, "materialize_transform", wraps=block.spectral.materialize_transform) as prepare:
        a = block(x)
        assert prepare.call_count == 1
    b = other(x)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    a.square().mean().backward()
    b.square().mean().backward()
    for p, q in zip(block.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10)


def small_model(ndim=2, *, output_sigmoid=True, transform="tied"):
    cls = LocalFNO2d if ndim == 2 else LocalFNO3d
    return cls(in_channels=2, base_width=4, spectral_rank=2,
               local_window=(4,) * ndim, local_modes=(3,) * ndim,
               global_modes=(3,) * ndim, local_operator="waveform", global_operator="waveform",
               local_operator_kwargs={"bins": 7, "transform": transform},
               global_operator_kwargs={"bins": 9, "transform": transform},
               patch_chunk_size=1000, output_sigmoid=output_sigmoid)


@pytest.mark.parametrize("ndim,output_sigmoid", [(2, True), (2, False), (3, True)])
@pytest.mark.parametrize("transform", ["tied", "separate"])
def test_unet_branches_shared_bottleneck_and_task_heads(ndim, output_sigmoid, transform):
    model = small_model(ndim, output_sigmoid=output_sigmoid, transform=transform)
    banks = [m for m in model.modules() if isinstance(m, WaveformBank)]
    assert len(banks) == (10 if transform == "separate" else 5)
    assert model.bottleneck[1].spectral.synthesis_bank is None
    assert model.bottleneck[1].spectral.bank is None
    assert model.bottleneck[0].spectral.weight is not model.bottleneck[1].spectral.weight
    x = torch.randn((1, 2) + (12,) * ndim)
    with patch.object(model.bottleneck[0].spectral, "materialize_transform",
                      wraps=model.bottleneck[0].spectral.materialize_transform) as prepare:
        output = model(x)
        assert prepare.call_count == 1
    assert output.shape == (1, 1) + (12,) * ndim
    if output_sigmoid:
        assert bool(((output > 0) & (output < 1)).all())
    output.square().mean().backward()
    for bank in banks:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                   for p in bank.parameters())
    assert all(isinstance(value, float) for value in waveform_diagnostics(model).values())


@pytest.mark.parametrize("local,global_", [("waveform", "fourier"), ("hadamard", "waveform")])
def test_mixed_existing_and_learned_slots(local, global_):
    model = LocalFNO2d(in_channels=2, base_width=4, spectral_rank=2,
                       local_window=(4, 4), local_modes=(2, 2), global_modes=(2, 2),
                       local_operator=local, global_operator=global_)
    output = model(torch.randn(1, 2, 16, 16))
    output.mean().backward()
    assert output.shape == (1, 1, 16, 16)


def test_configuration_environment_metadata_factory_and_optimizer():
    env = {"MODEL_KIND": "localop", "LOCAL_OPERATOR": "waveform", "GLOBAL_OPERATOR": "waveform",
           "WAVEFORM_LOCAL_BINS": "7", "WAVEFORM_GLOBAL_BINS": "9",
           "WAVEFORM_CONDITION_LIMIT": "1000", "WAVEFORM_LR_RATIO": "0.25",
           "WAVEFORM_INIT": "sine"}
    with patch.dict(os.environ, env, clear=True):
        config = ModelConfig.from_env(ndim=2)
    assert ModelConfig.from_dict(config.to_dict()) == config
    assert config.checkpoint_tag == "local_lwf_lwf"
    assert config.operator_slots()[0][1] == {"bins": 7, "condition_limit": 1000., "init": "sine", "transform": "tied"}
    assert config.operator_slots()[1][1] == {"bins": 9, "condition_limit": 1000., "init": "sine", "transform": "tied"}
    model = TrainerModel(build_model(config, in_channels=2))
    groups = waveform_parameter_groups(model, lr=.01, weight_decay=.1,
                                      waveform_lr_ratio=config.waveform_lr_ratio)
    assert groups[1]["lr"] == .0025 and groups[1]["weight_decay"] == 0
    parameters = [p for group in groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in model.parameters()}) == len({id(p) for p in parameters})
    assert len(groups[1]["params"]) == 10
    legacy = torch.nn.Linear(2, 2)
    optim = torch.optim.Adam(waveform_parameter_groups(legacy, lr=.01, weight_decay=.1))
    assert len(optim.param_groups) == 1


def test_checkpoint_and_optimizer_resume_reproduce_next_step():
    torch.manual_seed(19)
    model = small_model()
    optim = torch.optim.Adam(waveform_parameter_groups(model, lr=.001, weight_decay=.01))
    x = torch.randn(1, 2, 12, 12)

    def step(m, o):
        o.zero_grad()
        m(x).square().mean().backward()
        o.step()

    step(model, optim)
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optim.state_dict()}, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    other = small_model()
    other.load_state_dict(state["model"], strict=True)
    other_optim = torch.optim.Adam(waveform_parameter_groups(other, lr=.001, weight_decay=.01))
    other_optim.load_state_dict(state["optimizer"])
    step(model, optim)
    step(other, other_optim)
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_resampling_rebuilds_orthogonality_on_new_grid_without_new_parameters():
    op = LearnedWaveformOperator(2, 2, (5, 5), bins=7).double()
    parameters = list(op.parameters())
    for shape in ((9, 11), (15, 17)):
        bases = op.materialize_transform(shape, device="cpu", dtype=torch.float64)
        for u in bases:
            torch.testing.assert_close(u.T @ u, torch.eye(5, dtype=u.dtype))
        op(torch.randn(1, 2, *shape, dtype=torch.float64)).sum().backward()
    assert all(a is b for a, b in zip(parameters, op.parameters()))
    op.float()
    assert not op.bank._resampler_cache and not op.bank.last_diagnostics


def test_cpu_autocast_keeps_transform_float32_and_gradients_finite():
    model = small_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(1, 2, 12, 12))
        loss = output.float().square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(v.dtype == torch.float32 for m in model.modules() if isinstance(m, WaveformBank)
               for v in m.last_diagnostics.values())


def test_metrics_trainer_optimizes_tables_and_serializes_diagnostics(tmp_path):
    import json
    from torch.utils.data import DataLoader
    from util.neuralop_setup import prefer_local_neuralop
    prefer_local_neuralop()
    from training import MetricsTrainer

    torch.manual_seed(11)
    model = TrainerModel(small_model())
    before = model.fno.encoder0.spectral.bank.tables["0"].detach().clone()
    loader = DataLoader([
        {"x": torch.randn(2, 12, 12), "y": torch.rand(1, 12, 12)} for _ in range(2)
    ], batch_size=1)
    optim = torch.optim.Adam(waveform_parameter_groups(model, lr=.001, weight_decay=.01))
    path = tmp_path / "metrics.jsonl"
    trainer = MetricsTrainer(model=model, n_epochs=2, device="cpu", data_processor=None,
                             wandb_log=False, eval_interval=1, use_distributed=False,
                             verbose=False, metrics_path=path)

    def loss(out, y, **_):
        return (out - y).square().mean()

    trainer.train(train_loader=loader, test_loaders={"val": loader}, optimizer=optim,
                  scheduler=torch.optim.lr_scheduler.StepLR(optim, 1), regularizer=None,
                  training_loss=loss, eval_losses={"mse": loss})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert "waveform_fno_encoder0_spectral_bank_axis0_condition" in rows[-1]
    assert not torch.equal(before, model.fno.encoder0.spectral.bank.tables["0"])


def test_checkpoint_bank_plot_exports_actual_orthonormal_matrices(tmp_path):
    import numpy as np
    from viz.learned_waveforms import bank_names, load_bank, render_bank

    model = TrainerModel(small_model())
    state = model.state_dict()
    names = bank_names(state)
    assert len(names) == 5
    name = "fno.encoder0.spectral.bank"
    bank = load_bank(state, name)
    for axis in ("0", "1"):
        torch.testing.assert_close(bank.tables[axis], state[f"{name}.tables.{axis}"].double())
    render_bank(bank, (4, 4), tmp_path / "waveforms.png")
    assert (tmp_path / "waveforms.png").stat().st_size > 1000
    with np.load(tmp_path / "waveforms.npz") as data:
        u = data["axis0_orthonormal"]
        np.testing.assert_allclose(u.T @ u, np.eye(3), atol=1e-12)
