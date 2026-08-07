"""The differentiable bubble-size-spectrum loss.

The property that matters most here is a *negative* one: the spectrum is blind
to where the bubbles are. That is not a bug to fix, it is why the term needs an
anchor, and the trainer refuses to run it without one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from losses import GranulometrySpectrum


def discs(n: int, radius: int, size: int = 96, seed: int = 0,
          shift: int = 0) -> torch.Tensor:
    """Binary field of periodic discs -- a field with a known bubble size."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(n):
        cy, cx = rng.integers(0, size, 2)
        cy, cx = (cy + shift) % size, (cx + shift) % size
        d = (np.minimum(np.abs(yy - cy), size - np.abs(yy - cy)) ** 2
             + np.minimum(np.abs(xx - cx), size - np.abs(xx - cx)) ** 2)
        field[d <= radius ** 2] = 1.0
    return torch.tensor(field)[None, None]


def test_spectrum_is_a_normalised_non_negative_distribution() -> None:
    loss = GranulometrySpectrum(radii=(1, 2, 4, 8))
    spectrum = loss.spectrum(torch.rand(3, 1, 64, 64))
    assert spectrum.shape == (3, 3)
    assert bool((spectrum >= 0).all())
    assert torch.allclose(spectrum.sum(1), torch.ones(3), atol=1e-5)


def test_identical_fields_score_zero() -> None:
    loss = GranulometrySpectrum()
    field = torch.rand(2, 1, 64, 64)
    assert float(loss(field, field)) == pytest.approx(0.0, abs=1e-7)


def test_bigger_bubbles_shift_the_spectrum_to_larger_scales() -> None:
    """The whole point: the statistic must track bubble size."""
    loss = GranulometrySpectrum(radii=(1, 2, 4, 8, 12), downsample=1,
                                max_slices=None)
    scales = torch.tensor([1.0, 2.0, 4.0, 8.0])
    means = []
    for radius in (2, 4, 8):
        spectrum = loss.spectrum(discs(30, radius, seed=radius))[0]
        means.append(float((spectrum * scales).sum()))
    assert means[0] < means[1] < means[2], means


def test_the_spectrum_is_blind_to_bubble_position() -> None:
    """Why this term cannot stand alone.

    Translating every bubble leaves the size distribution untouched, so the
    loss barely moves -- while L2 sees a maximally wrong field. A run with no
    L2/expwall anchor would be free to put the bubbles anywhere, which is how
    the L2-free edge run failed (0.9098, never beat its epoch-0 value).
    """
    loss = GranulometrySpectrum(radii=(1, 2, 4, 8), downsample=1,
                               max_slices=None)
    truth = discs(30, 6, seed=1)
    moved = discs(30, 6, seed=1, shift=37)
    resized = discs(30, 12, seed=1)

    displaced = float(loss(moved, truth))
    wrong_size = float(loss(resized, truth))
    assert displaced < 0.2 * wrong_size, (displaced, wrong_size)
    # ... and the anchor term does see the displacement.
    assert float(((moved - truth) ** 2).mean()) > 10 * displaced


def test_gradient_reaches_the_prediction() -> None:
    loss = GranulometrySpectrum()
    pred = torch.rand(1, 1, 64, 64, requires_grad=True)
    loss(pred, discs(30, 6, size=64, seed=2)).backward()
    assert pred.grad is not None
    assert float(pred.grad.abs().sum()) > 0.0


def test_cubes_are_folded_into_transverse_planes() -> None:
    """3-D bubble sizes are measured per LOS slice, as the evaluator does."""
    loss = GranulometrySpectrum(radii=(1, 2, 4), downsample=1, max_slices=None)
    cube = torch.rand(2, 1, 32, 32, 5)
    assert loss.spectrum(cube).shape == (2 * 5, 2)


def test_slice_subsampling_is_deterministic() -> None:
    """Both calls of one evaluation must see the same slices."""
    loss = GranulometrySpectrum(radii=(1, 2, 4), max_slices=4)
    cube = torch.rand(1, 1, 32, 32, 40)
    assert torch.equal(loss.spectrum(cube), loss.spectrum(cube))
    assert len(loss.spectrum(cube)) == 4


def test_separable_opening_matches_the_square_window_exactly() -> None:
    """The speedup must be an implementation detail, not a change of statistic."""
    import torch.nn.functional as F

    planes = torch.rand(4, 1, 64, 64)
    for radius in (1, 2, 4, 8):
        size = 2 * radius + 1
        eroded = -F.max_pool2d(-planes, size, stride=1, padding=radius)
        square = F.max_pool2d(eroded, size, stride=1, padding=radius)
        assert torch.equal(GranulometrySpectrum._open(planes, radius), square)


def test_fragmentation_is_penalised_harder_than_blur() -> None:
    """The defect this term exists for.

    A field shattered into many small regions and one merely blurred are both
    wrong, but only the first has the wrong bubble sizes. L2 and expwall rank
    them much closer together than the size spectrum does.
    """
    import torch.nn.functional as F

    truth = discs(6, 18, size=128, seed=3)
    blurred = F.avg_pool2d(F.pad(truth, (3,) * 4, mode="circular"), 7, stride=1)
    # Shatter *in place*: punch correlated holes in the same regions and
    # speckle just outside them. Many small structures where a few large ones
    # belong -- not the same as scattering small discs, which merely coalesce.
    rng = np.random.default_rng(11)
    speckle = torch.tensor(rng.random((128, 128), dtype=np.float64)
                           .astype(np.float32))[None, None]
    speckle = F.avg_pool2d(F.pad(speckle, (2,) * 4, mode="circular"), 5, stride=1)
    shattered = (truth * (speckle > speckle.mean()).float()
                 + (1 - truth) * (speckle > speckle.mean() + 0.06).float() * 0.9
                 ).clamp(0, 1)

    loss = GranulometrySpectrum(radii=(1, 2, 4, 8, 12), downsample=1,
                                max_slices=None)
    scales = torch.tensor([1.0, 2.0, 4.0, 8.0])

    def mean_scale(x):
        return float((loss.spectrum(x)[0] * scales).sum())

    assert mean_scale(shattered) < 0.5 * mean_scale(truth)
    assert float(loss(shattered, truth)) > 5 * float(loss(blurred, truth))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_a_featureless_field_is_uniform_and_bounded(value: float) -> None:
    """A collapsed prediction must not explode the gradient.

    With zero mass at every scale, normalising by a clamped denominator gives
    a gradient of order 1/clamp -- measured 3e6 on a constant field. Additive
    smoothing makes the featureless case a uniform spectrum instead. This is
    not hypothetical: prediction collapse is what pred_sat_low/high watches
    for, so the term must survive exactly that state.
    """
    loss = GranulometrySpectrum(radii=(1, 2, 4, 8), downsample=1,
                                max_slices=None)
    flat = torch.full((1, 1, 64, 64), value, requires_grad=True)
    spectrum = loss.spectrum(flat)
    assert torch.allclose(spectrum, torch.full_like(spectrum, 1 / 3), atol=1e-3)

    loss(flat, discs(20, 6, size=64, seed=5)).backward()
    assert torch.isfinite(flat.grad).all()
    assert float(flat.grad.abs().max()) < 1e3


@pytest.mark.parametrize("ndim", [2, 3])
def test_every_architecture_backpropagates_through_the_term(ndim: int) -> None:
    """The term must not care which operator produced the field."""
    from modeling import ModelConfig, build_model

    kinds = [("ufno", {}), ("localfno", {}), ("localwno", {}),
             ("localwhno", {}), ("localsirenfno", {}),
             ("localop", dict(local_operator="hadamard", global_operator="cnn")),
             ("localop", dict(local_operator="cnn", global_operator="cnn"))]
    shape = (32, 32, 32) if ndim == 3 else (48, 48)
    x = torch.rand(1, 2, *shape)
    loss = GranulometrySpectrum(radii=(1, 2, 4), downsample=1,
                                max_slices=8 if ndim == 3 else None)
    for kind, extra in kinds:
        config = ModelConfig(
            kind=kind, ndim=ndim, modes=(4,) * ndim,
            localfno_window=(16,) * ndim, localfno_modes=(4,) * ndim,
            localfno_base_width=8, localfno_spectral_rank=8,
            ufno_width=4, hidden_channels=8, n_layers=2, **extra,
        )
        model = build_model(config, in_channels=2)
        out = model(x)
        loss(out, (torch.rand_like(out) > 0.5).float()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{kind}{extra} produced no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"{kind}{extra}"


def test_an_unbounded_output_still_yields_a_finite_loss() -> None:
    """SirenFNO with output_sigmoid=False is not confined to [0, 1].

    Openings stay monotone in radius for any real field, so the per-scale mass
    is still non-negative and the spectrum still normalises.
    """
    loss = GranulometrySpectrum(radii=(1, 2, 4), downsample=1, max_slices=None)
    wild = torch.randn(2, 1, 48, 48) * 5.0
    wild.requires_grad_(True)
    value = loss(wild, discs(20, 6, size=48, seed=4).expand(2, 1, 48, 48))
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(wild.grad).all()
    assert bool((loss.spectrum(wild.detach()) >= 0).all())


def test_rejects_malformed_configuration() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        GranulometrySpectrum(radii=(4, 2, 1))
    with pytest.raises(ValueError, match="at least two positive"):
        GranulometrySpectrum(radii=(4,))
    with pytest.raises(ValueError, match="downsample"):
        GranulometrySpectrum(downsample=0)


def test_rejects_mismatched_shapes() -> None:
    loss = GranulometrySpectrum()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 16, 16))
