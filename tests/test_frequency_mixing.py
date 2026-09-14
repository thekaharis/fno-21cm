"""Independent mathematical contracts for fixed Fourier coefficient mixing."""
import copy
import math

import pytest
import torch

from spectral_mixing_operator import (
    CoefficientMixer, FrequencyMixingOperator, RealFourierBasis, contract_basis,
    validate_mixing_shape,
)
from local_fno_3d import QuadrantSpectralConv3d
from models_zre_2d import QuadrantSpectralConv2d
from modeling import ModelConfig, build_model
from operators import build_operator, validate_operator


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def activate(mixer):
    with torch.no_grad():
        if mixer.backend == "dense":
            mixer.weight.normal_(std=.05)
        else:
            net = mixer.synthesis_net if mixer.backend == "factorized" else mixer.pair_net
            net[-1].weight.normal_(std=.05)


@pytest.mark.parametrize("shape,modes", [((7, 8), (2, 3)), ((7, 8, 9), (2, 2, 3))])
def test_basis_and_dense_spatial_oracle_with_gradients(shape, modes):
    op = FrequencyMixingOperator(2, len(shape), modes, mixing_rank=3, hidden_dim=7,
                                 chunk_size=11).double()
    activate(op.mixer)
    bases = op.basis.materialize(shape, device="cpu", dtype=torch.float64)
    u = bases[0]
    for basis in bases[1:]:
        u = torch.kron(u, basis)
    torch.testing.assert_close(u.T @ u, torch.eye(u.shape[1], dtype=u.dtype))
    full_u = torch.kron(torch.eye(2, dtype=u.dtype), u)
    x = torch.randn(2, 2, *shape, dtype=torch.float64, requires_grad=True)
    actual = op.residual(x, bases)
    b = op.mixer.dense_matrix()
    expected = (x.flatten(1) @ (full_u @ b @ full_u.T).T).reshape_as(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    probe = torch.randn_like(x)
    params = (x, *op.mixer.parameters())
    ga = torch.autograd.grad((actual * probe).sum(), params, retain_graph=True)
    gb = torch.autograd.grad((expected * probe).sum(), params)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize("shape,modes", [((7, 8), (2, 3)), ((7, 8, 9), (2, 2, 3))])
def test_zero_residual_exact_legacy_outputs_and_weight_gradients(shape, modes):
    cls = QuadrantSpectralConv2d if len(shape) == 2 else QuadrantSpectralConv3d
    legacy = cls(2, modes)
    for name, p in list(legacy.named_parameters()):
        setattr(legacy, name, torch.nn.Parameter(p.to(torch.complex128)))
    op = FrequencyMixingOperator(2, len(shape), modes, mixing_rank=3, hidden_dim=7).double()
    op.load_fourier_weights(legacy.state_dict())
    x = torch.randn(2, 2, *shape, dtype=torch.float64, requires_grad=True)
    probe = torch.randn_like(x)
    actual, expected = op(x), legacy(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad((actual * probe).sum(), (x, *op.baseline.parameters()))
    gb = torch.autograd.grad((expected * probe).sum(), (x, *legacy.parameters()))
    torch.testing.assert_close(ga[0], gb[0], atol=1e-12, rtol=1e-12)
    for a, b in zip(ga[1:], gb[1:]):
        torch.testing.assert_close(a, torch.view_as_real(b), atol=1e-12, rtol=1e-12)
    bad = dict(legacy.state_dict())
    bad["weights1"] = bad["weights1"][..., :1]
    saved = copy.deepcopy(op.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        op.load_fourier_weights(bad)
    for key, value in op.state_dict().items():
        torch.testing.assert_close(value, saved[key])


def test_prescribed_cross_frequency_and_dc_transfer_on_two_resolutions():
    op = FrequencyMixingOperator(1, 2, (2, 2), backend="dense").double()
    with torch.no_grad():
        # Input sin(2 pi x), output cos(4 pi x) cos(2 pi y) and a constant.
        op.mixer.weight[4 * 3 + 2, 1 * 3] = 1
        op.mixer.weight[0, 1 * 3] = .5
    for shape in ((9, 11), (18, 22)):
        x = torch.arange(shape[0], dtype=torch.float64) / shape[0]
        y = torch.arange(shape[1], dtype=torch.float64) / shape[1]
        field = (math.sqrt(2) * (2 * math.pi * x).sin())[:, None].expand(shape)
        expected = 2 * (4 * math.pi * x).cos()[:, None] * (2 * math.pi * y).cos()[None, :] + .5
        bases = op.basis.materialize(shape, device="cpu", dtype=torch.float64)
        torch.testing.assert_close(op.residual(field[None, None], bases)[0, 0], expected,
                                   atol=1e-12, rtol=1e-12)


def test_dense_kernel_for_spatial_multiplication_matches_projected_product():
    basis = RealFourierBasis((2, 2))
    bases = basis.materialize((9, 9), device="cpu", dtype=torch.float64)
    u = torch.kron(*bases)
    a = torch.randn(81, dtype=torch.float64)
    mixer = CoefficientMixer(1, basis.coordinates(), backend="dense").double()
    with torch.no_grad():
        mixer.weight.copy_(u.T @ (a[:, None] * u))
    c = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    field = contract_basis(c, bases, analysis=False)
    output = contract_basis(mixer(c), bases, analysis=False)
    expected = u @ (u.T @ (a * field.flatten()))
    torch.testing.assert_close(output.flatten(), expected)


@pytest.mark.parametrize("backend", ["factorized", "dense", "pairwise"])
def test_backends_reference_preparation_checkpoint_and_learning(backend):
    op = FrequencyMixingOperator(2, 2, (1, 2), backend=backend, mixing_rank=3,
                                 hidden_dim=7, chunk_size=4).double()
    optimizer = torch.optim.Adam(op.parameters(), lr=.003)
    x = torch.randn(2, 2, 5, 7, dtype=torch.float64)
    target = torch.randn_like(x)
    for step in range(3):
        optimizer.zero_grad()
        (op(x) - target).square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in op.parameters())
        if backend == "factorized":
            if step == 0:
                assert op.mixer.analysis_net[0].weight.grad.norm() == 0
                assert op.mixer.synthesis_net[-1].weight.grad.norm() > 0
            else:
                assert op.mixer.analysis_net[0].weight.grad.norm() > 0
        optimizer.step()
    c = torch.randn(2, 2, *op.basis.counts, dtype=torch.float64)
    expected = (c.flatten(1) @ op.mixer.dense_matrix().T).reshape_as(c)
    torch.testing.assert_close(op.mixer(c), expected)
    prepared = op.materialize_transform(x.shape[2:], device=x.device, dtype=x.dtype)
    torch.testing.assert_close(op(x, transform=prepared), op(x))
    restored = copy.deepcopy(op)
    restored.load_state_dict(op.state_dict())
    torch.testing.assert_close(restored(x), op(x))
    torch.testing.assert_close(op.diagnostics()["residual_frobenius"], op.mixer.dense_matrix().norm())


def test_factorized_autograd_and_chunks():
    basis = RealFourierBasis((1, 1))
    mixer = CoefficientMixer(1, basis.coordinates(), mixing_rank=2, hidden_dim=3, chunk_size=2).double()
    activate(mixer)
    c = torch.randn(1, 1, 3, 1, dtype=torch.float64, requires_grad=True)
    w = mixer.synthesis_net[-1].weight
    assert torch.autograd.gradcheck(
        lambda a, v: torch.func.functional_call(mixer, {"synthesis_net.4.weight": a}, (v,)),
        (w, c), fast_mode=True)
    other = copy.deepcopy(mixer)
    other.chunk_size = 100
    torch.testing.assert_close(other(c), mixer(c))
    torch.testing.assert_close(mixer(c, weights=mixer.factors()), mixer(c))


def test_coordinates_do_not_change_with_cutoff_and_nyquist_is_rejected():
    small, large = RealFourierBasis((1, 2)), RealFourierBasis((2, 3))
    for row in small.coordinates():
        assert (large.coordinates() == row).all(dim=1).any()
    with pytest.raises(ValueError, match="Nyquist"):
        validate_mixing_shape((4, 8), (2, 2))
    with pytest.raises(ValueError, match="limit"):
        FrequencyMixingOperator(16, 3, (16, 16, 16), backend="dense")
    with pytest.raises(ValueError, match="positive"):
        FrequencyMixingOperator(1, 2, (1, 1), mixing_rank=0)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("local", ["fourier", "frequency_mixing"])
def test_registry_model_config_amp_and_windowed_integration(ndim, local, monkeypatch):
    monkeypatch.setenv("FREQUENCY_MIXING_RANK", "3")
    monkeypatch.setenv("FREQUENCY_MIXING_HIDDEN_DIM", "7")
    assert ModelConfig.from_env(ndim).frequency_mixing_rank == 3
    cfg = ModelConfig(kind="localop", ndim=ndim, local_operator=local,
                      global_operator="frequency_mixing", localfno_window=(4,) * ndim,
                      localfno_modes=(1,) * ndim, modes=(1,) * ndim,
                      localfno_base_width=4, localfno_spectral_rank=2,
                      frequency_mixing_rank=3, frequency_mixing_hidden_dim=7,
                      localfno_patch_chunk_size=64)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    model = build_model(cfg, in_channels=2)
    x = torch.randn(1, 2, *((12,) * ndim))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(x)
    y.float().square().mean().backward()
    assert y.shape == (1, 1, *((12,) * ndim))
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_mixing_changes_translation_behavior():
    op = FrequencyMixingOperator(1, 2, (2, 2), backend="dense").double()
    x = torch.randn(1, 1, 9, 9, dtype=torch.float64)
    shift = lambda v: torch.roll(v, 1, dims=2)
    torch.testing.assert_close(op(shift(x)), shift(op(x)))
    with torch.no_grad():
        op.mixer.weight[12, 3] = 1
    assert (op(shift(x)) - shift(op(x))).norm() > .01


@pytest.mark.parametrize("slots", ["global", "both"])
def test_full_checkpoint_migration_is_exact_and_rejects_mismatch(slots):
    from util.frequency_mixing_checkpoint import convert_state
    config = ModelConfig(kind="localfno", ndim=2, modes=(1, 1),
                         localfno_modes=(1, 1), localfno_window=(4, 4),
                         localfno_base_width=4, localfno_spectral_rank=2,
                         frequency_mixing_rank=3, frequency_mixing_hidden_dim=7)
    original = build_model(config, in_channels=2)
    state = original.state_dict()
    migrated, new_config = convert_state(state, config, slots=slots)
    assert set(original.state_dict()) == set(state)
    new = build_model(new_config, in_channels=2)
    new.load_state_dict(migrated, strict=True)
    x = torch.randn(1, 2, 12, 12)
    torch.testing.assert_close(new(x), original(x), atol=1e-6, rtol=1e-6)
    assert new_config.family_tag == ("fno_fmix" if slots == "global" else "fmix_fmix")
    with pytest.raises(ValueError, match="Fourier global"):
        convert_state(migrated, new_config)
    with pytest.raises(ValueError, match="mismatch"):
        convert_state(state, ModelConfig.from_dict(dict(config.to_dict(), modes=[2, 2])))


def test_diagnostic_export_matches_selected_dense_entries(tmp_path):
    import numpy as np
    from viz.frequency_mixing import export_mixer
    op = FrequencyMixingOperator(2, 2, (2, 2), mixing_rank=3, hidden_dim=7).double()
    activate(op.mixer)
    summary = export_mixer(op, tmp_path / "mixer", max_modes=4)
    data = np.load(tmp_path / "mixer.npz")
    indices = torch.from_numpy(data["mode_indices"])
    selected = (torch.arange(2)[:, None] * op.mixer.count + indices).flatten()
    expected = op.mixer.dense_matrix()[selected][:, selected].detach().numpy()
    np.testing.assert_allclose(data["residual_matrix"], expected, atol=1e-12)
    assert summary["sampled_modes"] == 4


def test_checkpoint_loader_requires_complete_mixing_weights(tmp_path):
    from modeling import load_checkpoint
    op = FrequencyMixingOperator(1, 2, (1, 1), mixing_rank=2, hidden_dim=3)
    path = tmp_path / "checkpoint.pt"
    state = op.state_dict()
    torch.save(state, path)
    assert not load_checkpoint(op, path).missing
    broken = dict(state)
    del broken["mixer.analysis_net.0.weight"]
    torch.save(broken, path)
    with pytest.raises(ValueError, match="complete model"):
        load_checkpoint(op, path)
