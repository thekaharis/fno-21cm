"""Independent orthonormal banks preserve initialization and learn distinct maps."""

import copy

import pytest
import torch

from learned_waveform_operator import LearnedWaveformOperator, WaveformTransform
from modeling import ModelConfig, TrainerModel, build_model
from waveform_training import warm_start


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("init", ["random", "sine", "square"])
def test_separate_starts_exactly_tied_without_extra_rng_draws(ndim, init):
    torch.manual_seed(12)
    tied = LearnedWaveformOperator(2, ndim, (3,) * ndim, bins=7, init=init).double()
    rng = torch.get_rng_state()
    torch.manual_seed(12)
    separate = LearnedWaveformOperator(2, ndim, (3,) * ndim, bins=7, init=init,
                                       transform="separate").double()
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn((1, 2) + (7,) * ndim, dtype=torch.float64)
    torch.testing.assert_close(tied(x), separate(x), atol=0, rtol=0)
    for axis, table in separate.bank.tables.items():
        other = separate.synthesis_bank.tables[axis]
        assert table.data_ptr() != other.data_ptr()
        torch.testing.assert_close(table, other, atol=0, rtol=0)


@pytest.mark.parametrize("ndim", [2, 3])
def test_synthesis_maps_to_different_subspace_without_inverse_constraint(ndim):
    op = LearnedWaveformOperator(1, ndim, (3,) * ndim, bins=7, transform="separate").double()
    with torch.no_grad():
        op.weight.fill_(1)
        op.phase_weight.zero_()
    analysis = tuple(torch.eye(5, dtype=torch.float64)[:, :3] for _ in range(ndim))
    synthesis = tuple(torch.eye(5, dtype=torch.float64)[:, 2:] for _ in range(ndim))
    x = torch.randn((1, 1) + (5,) * ndim, dtype=torch.float64)
    qin, qout = analysis[0], synthesis[0]
    for axis in range(1, ndim):
        qin = torch.kron(qin, analysis[axis])
        qout = torch.kron(qout, synthesis[axis])
    actual = op(x, transform=WaveformTransform(analysis, synthesis))
    expected = (qout @ qin.T @ x.flatten()).reshape_as(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.allclose(actual.flatten(), qin @ qin.T @ x.flatten())


def test_both_banks_have_correct_independent_gradients_and_remain_orthonormal():
    torch.manual_seed(13)
    op = LearnedWaveformOperator(2, 2, (3, 3), bins=7, init="square", transform="separate").double()
    x = torch.randn(2, 2, 9, 9, dtype=torch.float64)
    target = torch.randn_like(x)
    loss = lambda: (op(x) - target).square().mean()
    loss().backward()
    for bank in (op.bank, op.synthesis_bank):
        for p in bank.tables.values():
            assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
            expected = p.grad[0].item()
            with torch.no_grad():
                saved = p[0].item()
                p[0] = saved + 1e-5
                plus = loss().item()
                p[0] = saved - 1e-5
                minus = loss().item()
                p[0] = saved
            assert (plus - minus) / 2e-5 == pytest.approx(expected, rel=1e-5, abs=1e-8)
    assert not torch.allclose(op.bank.tables["0"].grad, op.synthesis_bank.tables["0"].grad)
    torch.optim.SGD(op.parameters(), lr=.1).step()
    assert not torch.equal(op.bank.tables["0"], op.synthesis_bank.tables["0"])
    transform = op.materialize_transform((9, 9), device="cpu", dtype=torch.float64)
    for basis in transform.analysis + transform.synthesis:
        torch.testing.assert_close(basis.T @ basis, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)


def config(transform):
    return ModelConfig(kind="localop", ndim=2, local_operator="waveform", global_operator="waveform",
                       localfno_window=(4, 4), localfno_modes=(3, 3), modes=(3, 3),
                       localfno_base_width=4, localfno_spectral_rank=2,
                       waveform_local_bins=7, waveform_global_bins=9, waveform_transform=transform)


def test_config_environment_and_tied_checkpoint_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORM_TRANSFORM", "separate")
    assert ModelConfig.from_env(ndim=2).waveform_transform == "separate"
    assert ModelConfig.from_dict({}).waveform_transform == "tied"
    spec = config("separate")
    assert ModelConfig.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="transform"):
        config("unknown")
    tied = TrainerModel(build_model(config("tied"), 2)).double()
    separate = TrainerModel(build_model(spec, 2)).double()
    path = tmp_path / "tied.pt"
    torch.save(tied.state_dict(), path)
    report = warm_start(separate, path, strict=True)
    assert not report.missing and not report.unexpected
    x = torch.randn(1, 2, 12, 12, dtype=torch.float64)
    torch.testing.assert_close(tied(x), separate(x), atol=0, rtol=0)
    # Both banks must be restored individually once they have diverged.
    with torch.no_grad():
        separate.fno.encoder0.spectral.synthesis_bank.tables["0"][0] += .1
    restored = TrainerModel(build_model(spec, 2)).double()
    restored.load_state_dict(separate.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), separate(x), atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="WAVEFORM_TRANSFORM=separate"):
        tied.load_state_dict(separate.state_dict(), strict=False)
    partial = copy.deepcopy(separate.state_dict())
    del partial["fno.encoder0.spectral.synthesis_bank.tables.0"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(partial, strict=True)


def test_separate_requires_both_transforms_and_supports_dc_only():
    op = LearnedWaveformOperator(1, 2, (1, 1), transform="separate")
    x = torch.randn(1, 1, 5, 7)
    transform = op.materialize_transform((5, 7), device="cpu", dtype=torch.float32)
    assert isinstance(transform, WaveformTransform)
    with pytest.raises(ValueError, match="analysis and synthesis"):
        op(x, transform=transform.analysis)
    op(x).sum().backward()
    assert all(p.grad is not None for p in op.parameters())
