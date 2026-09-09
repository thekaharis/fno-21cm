"""Phase expressivity, Fourier equivalence, and selectable waveform starts."""

import math

import pytest
import torch

from learned_waveform_operator import (
    LearnedWaveformOperator, WaveformBank, WAVEFORM_INITIALIZATIONS,
)
from local_fno_3d import QuadrantSpectralConv3d
from models_zre_2d import QuadrantSpectralConv2d
from modeling import ModelConfig
from operators import validate_operator


@pytest.fixture
def double_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("shape", [(8, 9), (7, 8, 9)])
def test_phase_blocks_reproduce_existing_fno_outputs_and_gradients(shape, double_default):
    """Copy the actual signed-quadrant FFT operator into all-real phase blocks.

    Use enough real columns to include its asymmetric signed-axis cutoff,
    including the Hermitian completion induced by irFFT on the DC plane.
    Nyquist is not retained by either operator in this comparison.
    """
    torch.manual_seed(21)
    ndim, channels = len(shape), 2
    real_modes = (5,) * (ndim - 1) + (3,)
    cls = QuadrantSpectralConv2d if ndim == 2 else QuadrantSpectralConv3d
    reference = cls(channels, (2,) * ndim)
    for name, value in list(reference.named_parameters()):
        setattr(reference, name, torch.nn.Parameter(value.to(torch.complex128)))
    learned = LearnedWaveformOperator(channels, ndim, real_modes, bins=15, init="sine")
    bases = tuple(u.detach() for u in learned.materialize_transform(
        shape, device="cpu", dtype=torch.float64))
    count = math.prod(real_modes)
    # Responses to coefficient impulses give the exact real representation
    # of the reference, differentiably in all its complex channel weights.
    impulses = torch.eye(channels * count).reshape(channels * count, channels, *real_modes)
    signals = learned.contract(impulses, bases, analysis=False)
    responses = learned.contract(reference(signals), bases, analysis=True)
    dense = responses.reshape(channels, count, channels, count).permute(0, 2, 1, 3)
    diagonal = dense.diagonal(dim1=2, dim2=3).reshape(channels, channels, *real_modes)
    phase = dense[:, :, learned.phase_source, learned.phase_destination]
    assert phase.abs().max() > .01  # this test requires genuine phase mixing
    allowed = torch.eye(count, dtype=torch.bool)
    allowed[learned.phase_source, learned.phase_destination] = True
    torch.testing.assert_close(dense[:, :, ~allowed], torch.zeros_like(dense[:, :, ~allowed]),
                               atol=1e-12, rtol=0)

    x = torch.randn(2, channels, *shape, requires_grad=True)
    actual = torch.func.functional_call(learned, {"weight": diagonal, "phase_weight": phase},
                                         (x,), {"transform": bases})
    expected = reference(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(actual)
    inputs = (x, *reference.parameters())
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("modes", [(4, 5), (3, 4, 1), (3, 3, 3), (1, 2)])
def test_packed_blocks_cover_every_valid_pair_and_only_same_dilation(modes):
    op = LearnedWaveformOperator(2, len(modes), modes)
    coordinates = torch.stack(torch.meshgrid(*(torch.arange(m) for m in modes),
                                              indexing="ij"), dim=-1).reshape(-1, len(modes))
    groups = (coordinates + 1) // 2
    same_group = (groups[:, None] == groups[None, :]).all(dim=-1)
    same_group.fill_diagonal_(False)
    packed = torch.zeros_like(same_group, dtype=torch.int)
    packed[op.phase_source, op.phase_destination] += 1
    torch.testing.assert_close(packed, same_group.int())
    if same_group.any():
        assert op.phase_weight.shape == (2, 2, int(same_group.sum()))
    else:
        assert op.phase_weight is None


def test_cross_axis_phase_rotation_and_no_dc_leakage(double_default):
    op = LearnedWaveformOperator(1, 2, (3, 3), init="sine")
    with torch.no_grad():
        op.weight.zero_()
        # sin(x) sin(y) -> cos(x) cos(y): both axes must change phase.
        edge = (op.phase_source == 4) & (op.phase_destination == 8)
        op.phase_weight[0, 0, edge] = 1
    bases = op.materialize_transform((9, 11), device="cpu", dtype=torch.float64)
    coefficients = torch.zeros(1, 1, 3, 3)
    coefficients[0, 0, 1, 1] = 1
    expected = torch.zeros_like(coefficients)
    expected[0, 0, 2, 2] = 1
    x = op.contract(coefficients, bases, analysis=False)
    torch.testing.assert_close(op(x), op.contract(expected, bases, analysis=False))
    torch.testing.assert_close(op(torch.ones_like(x)), torch.zeros_like(x), atol=1e-12, rtol=0)


def test_nonzero_phase_weights_have_correct_bin_and_weight_gradients(double_default):
    torch.manual_seed(9)
    op = LearnedWaveformOperator(2, 2, (3, 3), bins=7)
    with torch.no_grad():
        op.phase_weight.normal_(std=.2)
    x = torch.randn(1, 2, 7, 9, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, w, v: torch.func.functional_call(
            op, {"bank.tables.0": t, "phase_weight": w}, (v,)),
        (op.bank.tables["0"], op.phase_weight, x), fast_mode=True,
    )


@pytest.mark.parametrize("init", WAVEFORM_INITIALIZATIONS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_starting_waveforms_are_reproducible_orthogonal_and_trainable(init, ndim):
    torch.manual_seed(13)
    op = LearnedWaveformOperator(2, ndim, (5,) * ndim, bins=15, init=init)
    torch.manual_seed(13)
    other = LearnedWaveformOperator(2, ndim, (5,) * ndim, bins=15, init=init)
    for p, q in zip(op.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    bases = op.materialize_transform((16,) * ndim, device="cpu", dtype=torch.float32)
    for u in bases:
        torch.testing.assert_close(u.T @ u, torch.eye(5), atol=1e-6, rtol=1e-6)
    optimizer = torch.optim.Adam(op.parameters(), lr=.001)
    before = [t.detach().clone() for t in op.bank.tables.values()]
    op(torch.randn((1, 2) + (16,) * ndim)).square().mean().backward()
    assert op.phase_weight.grad.norm() > 0
    for table in op.bank.tables.values():
        assert table.grad.isfinite().all() and table.grad.norm() > 0
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, op.bank.tables.values()))


def test_named_profiles_and_random_axes_are_distinct():
    profiles = [WaveformBank(2, (3, 3), init=init).tables["0"].detach()
                for init in WAVEFORM_INITIALIZATIONS]
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            assert not torch.allclose(a / a.norm(), b / b.norm())
    for init in ("random", "smooth_random"):
        bank = WaveformBank(2, (3, 3), init=init)
        assert not torch.equal(bank.tables["0"], bank.tables["1"])


def test_invalid_initialization_fails_at_configuration_and_operator_validation():
    with pytest.raises(ValueError, match="waveform init"):
        ModelConfig(waveform_init="invalid")
    with pytest.raises(ValueError, match="waveform init"):
        validate_operator("waveform", (8, 8), (3, 3), hyperparameters={"init": "invalid"})
    with pytest.raises(ValueError, match="waveform init"):
        WaveformBank(2, (1, 1), init="invalid")


def test_legacy_weight_only_checkpoint_preserves_output_and_new_weights_roundtrip():
    torch.manual_seed(31)
    old = LearnedWaveformOperator(2, 2, (3, 5))
    state = {k: v for k, v in old.state_dict().items() if k != "phase_weight"}
    restored = LearnedWaveformOperator(2, 2, (3, 5), init="square")
    restored.load_state_dict(state, strict=True)
    x = torch.randn(1, 2, 9, 11)
    torch.testing.assert_close(restored(x), old(x), atol=0, rtol=0)
    with torch.no_grad():
        restored.phase_weight.normal_(std=.1)
    other = LearnedWaveformOperator(2, 2, (3, 5))
    other.load_state_dict(restored.state_dict(), strict=True)
    torch.testing.assert_close(other(x), restored(x), atol=0, rtol=0)
    assert ModelConfig.from_dict({}).waveform_init == "random"
