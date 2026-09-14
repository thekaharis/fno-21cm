"""Fixed Fourier analysis with coordinate-generated cross-frequency interactions.

The residual is U A P U.T, of rank at most mixing_rank in channel/coefficient
space. The baseline is the repository's signed-quadrant Fourier multiplier,
evaluated with FFTs instead of materializing its real coefficient matrix.
Mode counts use the existing FFT cutoff convention, not waveform column counts.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn


def real_mode_counts(modes):
    # Signed axes include -m in the legacy FFT, hence complete pairs through m.
    return tuple(2 * m + (1 if a < len(modes) - 1 else -1)
                 for a, m in enumerate(modes))


def validate_mixing_options(backend="factorized", mixing_rank=32, hidden_dim=64,
                            chunk_size=1024, dense_limit=4_000_000):
    if backend not in {"factorized", "dense", "pairwise"}:
        raise ValueError("mixing backend must be factorized, dense, or pairwise")
    for name, value in (("mixing_rank", mixing_rank), ("hidden_dim", hidden_dim),
                        ("chunk_size", chunk_size), ("dense_limit", dense_limit)):
        if int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def validate_mixing_shape(sizes, modes, *, context="frequency_mixing"):
    if len(modes) not in (2, 3) or len(sizes) != len(modes) or any(m < 1 for m in modes):
        raise ValueError("frequency_mixing requires 2-D or 3-D positive modes")
    counts = real_mode_counts(modes)
    limits = tuple(n if n % 2 else n - 1 for n in sizes)
    if any(m > n for m, n in zip(counts, limits)):
        raise ValueError(f"{context}: real Fourier counts {counts} exceed {limits}; Nyquist excluded")


class RealFourierBasis:
    """Deterministic DC, sin(k), cos(k) columns on [0,1), with a bounded cache."""

    def __init__(self, modes: Sequence[int]):
        self.modes = tuple(int(m) for m in modes)
        self.counts = real_mode_counts(self.modes)
        self._cache = {}

    def materialize(self, shape, *, device, dtype):
        validate_mixing_shape(shape, self.modes)
        key = (tuple(shape), str(device), dtype)
        if key not in self._cache:
            bases = []
            for n, m in zip(shape, self.counts):
                x = torch.arange(n, device=device, dtype=dtype) / n
                k = torch.arange(1, (m + 1) // 2, device=device, dtype=dtype)
                angles = 2 * math.pi * x[:, None] * k
                pairs = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
                bases.append(torch.cat((torch.ones(n, 1, device=device, dtype=dtype) / math.sqrt(n),
                                        pairs * math.sqrt(2 / n)), dim=1))
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[key] = tuple(bases)
        return self._cache[key]

    def coordinates(self):
        # The frequency scale and phase encoding never depend on grid/cutoff.
        axes = []
        for count in self.counts:
            j = torch.arange(count)
            k = (j + 1) // 2
            phase = torch.where(j == 0, 0, torch.where(j % 2 == 1, 1, 2))
            axes.append(torch.cat((k[:, None].float(),
                                   torch.nn.functional.one_hot(phase, 3).float()), dim=1))
        indices = torch.cartesian_prod(*(torch.arange(len(a)) for a in axes))
        return torch.cat([a[indices[:, i]] for i, a in enumerate(axes)], dim=1)


def contract_basis(x, bases, *, analysis):
    """Separable contraction; never build the product-grid basis."""
    for axis, basis in enumerate(bases):
        position = 2 + axis
        x = torch.tensordot(x, basis, dims=([position], [0 if analysis else 1]))
        x = x.movedim(-1, position)
    return x


class FourierMultiplier(nn.Module):
    """Legacy quadrant multiplier with real-packed parameters for dtype safety."""

    def __init__(self, channels, modes):
        super().__init__()
        self.channels, self.n_modes = channels, tuple(modes)
        for block in range(2 ** (len(modes) - 1)):
            weight = torch.randn(channels, channels, *modes, dtype=torch.cfloat) / channels**2
            self.register_parameter(f"weights{block + 1}", nn.Parameter(torch.view_as_real(weight)))

    def forward(self, x):
        dims = tuple(range(2, x.ndim))
        ft = torch.fft.rfftn(x, dim=dims)
        out = torch.zeros_like(ft)
        # First signed axis flips fastest, matching weights1..4 in the repo.
        for block in range(2 ** (len(self.n_modes) - 1)):
            slices = [slice(-m, None) if block & (1 << a) else slice(0, m)
                      for a, m in enumerate(self.n_modes[:-1])]
            index = (slice(None), slice(None), *slices, slice(0, self.n_modes[-1]))
            weight = torch.view_as_complex(getattr(self, f"weights{block + 1}").contiguous())
            out[index] = torch.einsum("bi...,io...->bo...", ft[index], weight.to(ft.dtype))
        return torch.fft.irfftn(out, s=x.shape[2:], dim=dims)

    @torch.no_grad()
    def load_legacy(self, state):
        expected = {f"weights{i + 1}" for i in range(2 ** (len(self.n_modes) - 1))}
        if set(state) != expected:
            raise ValueError("expected exactly the legacy Fourier multiplier weights")
        # Validate everything before modifying anything.
        for key in expected:
            if not state[key].is_complex() or state[key].shape != getattr(self, key).shape[:-1]:
                raise ValueError(f"incompatible Fourier weight {key}")
        for key in expected:
            getattr(self, key).copy_(torch.view_as_real(state[key]))


def coordinate_network(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, output_dim))


class CoefficientMixer(nn.Module):
    """Cross-frequency residual, generated from fixed coefficient coordinates.

All forward computations use live parameters. Factors are shared over examples
in a batch; they are not attention weights conditioned on the input.
"""

    def __init__(self, channels, coordinates, *, backend="factorized", mixing_rank=32,
                 hidden_dim=64, chunk_size=1024, dense_limit=4_000_000):
        super().__init__()
        validate_mixing_options(backend, mixing_rank, hidden_dim, chunk_size, dense_limit)
        self.channels, self.count = channels, len(coordinates)
        self.backend, self.mixing_rank = backend, mixing_rank
        self.chunk_size, self.dense_limit = chunk_size, dense_limit
        self.register_buffer("coordinates", coordinates.clone(), persistent=False)
        dim = coordinates.shape[1]
        size = channels * self.count
        if backend == "factorized":
            self.analysis_net = coordinate_network(dim, hidden_dim, channels * mixing_rank)
            self.synthesis_net = coordinate_network(dim, hidden_dim, channels * mixing_rank)
            # Fixed initial scale is stored in the checkpoint, not recomputed
            # per forward or used to average the coefficient sum.
            self.register_buffer("factor_scale", torch.tensor(size ** -.5))
        else:
            if size * size > dense_limit:
                raise ValueError(f"dense mixing requires {size * size} entries, limit={dense_limit}")
            if backend == "dense":
                self.weight = nn.Parameter(torch.zeros(size, size))
            else:
                self.pair_net = coordinate_network(2 * dim, hidden_dim, channels * channels)
        self.reset_residual()

    @torch.no_grad()
    def reset_residual(self):
        if self.backend == "dense":
            self.weight.zero_()
        else:
            last = self.synthesis_net[-1] if self.backend == "factorized" else self.pair_net[-1]
            last.weight.zero_()
            last.bias.zero_()

    def _factor(self, net, coordinates):
        return (net(coordinates).reshape(-1, self.channels, self.mixing_rank)
                .permute(1, 0, 2) * self.factor_scale)

    def factors(self):
        """Materialize O(C M r) factors for diagnostics or a tiny dense oracle."""
        if self.backend != "factorized":
            raise ValueError("factors require factorized mixing")
        p, a = [], []
        for coords in self.coordinates.split(self.chunk_size):
            p.append(self._factor(self.analysis_net, coords))
            a.append(self._factor(self.synthesis_net, coords))
        return torch.cat(a, dim=1), torch.cat(p, dim=1)

    def dense_matrix(self):
        size = self.channels * self.count
        if size * size > self.dense_limit:
            raise ValueError("dense diagnostic exceeds dense_limit")
        if self.backend == "dense":
            return self.weight
        if self.backend == "factorized":
            a, p = self.factors()
            return a.reshape(size, -1) @ p.reshape(size, -1).T
        rows = []
        for coords in self.coordinates.split(self.chunk_size):
            k = coords[:, None, :].expand(-1, self.count, -1)
            q = self.coordinates[None, :, :].expand(len(coords), -1, -1)
            value = self.pair_net(torch.cat((k, q), dim=-1))
            rows.append(value.reshape(len(coords), self.count, self.channels, self.channels))
        return torch.cat(rows).permute(2, 0, 3, 1).reshape(size, size)

    def forward(self, c, *, weights=None):
        flat = c.flatten(2)
        if flat.shape[1:] != (self.channels, self.count):
            raise ValueError("coefficient shape does not match mixer")
        if weights is not None:
            if self.backend != "factorized":
                return (flat.flatten(1) @ weights.T).reshape_as(c)
            a, p = weights
            latent = torch.einsum("bim,imr->br", flat, p)
            return torch.einsum("br,omr->bom", latent, a).reshape_as(c)
        if self.backend != "factorized":
            return (flat.flatten(1) @ self.dense_matrix().T).reshape_as(c)
        latent = None
        for start in range(0, self.count, self.chunk_size):
            end = min(start + self.chunk_size, self.count)
            p = self._factor(self.analysis_net, self.coordinates[start:end])
            value = torch.einsum("bim,imr->br", flat[:, :, start:end], p)
            latent = value if latent is None else latent + value
        pieces = []
        for coords in self.coordinates.split(self.chunk_size):
            a = self._factor(self.synthesis_net, coords)
            pieces.append(torch.einsum("br,omr->bom", latent, a))
        return torch.cat(pieces, dim=2).reshape_as(c)


class PreparedTransform(NamedTuple):
    bases: tuple[torch.Tensor, ...]
    weights: object


class FrequencyMixingOperator(nn.Module):
    """Fourier multiplier plus a general coefficient-space residual."""

    def __init__(self, channels: int, ndim: int, modes: Sequence[int], **options):
        super().__init__()
        self.channels, self.ndim = int(channels), int(ndim)
        self.n_modes = tuple(int(m) for m in modes)
        if self.channels < 1 or self.ndim not in (2, 3) or len(self.n_modes) != self.ndim:
            raise ValueError("expected positive channels and matching 2-D/3-D modes")
        if any(m < 1 for m in self.n_modes):
            raise ValueError("modes must be positive")
        self.basis = RealFourierBasis(self.n_modes)
        self.baseline = FourierMultiplier(self.channels, self.n_modes)
        self.mixer = CoefficientMixer(self.channels, self.basis.coordinates(), **options)

    def _apply(self, fn, recurse=True):
        self.basis._cache.clear()
        return super()._apply(fn, recurse=recurse)

    def materialize_transform(self, shape, *, device, dtype):
        dtype = next(self.parameters()).dtype
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            bases = self.basis.materialize(shape, device=device, dtype=dtype)
            weights = (self.mixer.factors() if self.mixer.backend == "factorized"
                       else self.mixer.dense_matrix())
        return PreparedTransform(bases, weights)

    def residual(self, x, bases, *, weights=None):
        c = contract_basis(x, bases, analysis=True)
        return contract_basis(self.mixer(c, weights=weights), bases, analysis=False)

    def forward(self, x, *, transform=None):
        if x.ndim != self.ndim + 2 or x.shape[1] != self.channels or not x.is_floating_point():
            raise ValueError("expected real (batch, channels, *spatial) input")
        validate_mixing_shape(x.shape[2:], self.n_modes)
        # FFT and contractions run in FP32 under AMP; .double() also correctly
        # promotes the real-packed baseline parameters for derivative checks.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = next(self.parameters()).dtype
            v = x.to(dtype)
            if transform is None:
                transform = self.materialize_transform(v.shape[2:], device=v.device, dtype=dtype)
            if not isinstance(transform, PreparedTransform) or any(
                tuple(u.shape) != (n, m) for u, n, m in
                zip(transform.bases, v.shape[2:], self.basis.counts)
            ) or len(transform.bases) != self.ndim:
                raise ValueError("prepared transform does not match spatial shape")
            return (self.baseline(v) + self.residual(
                v, transform.bases, weights=transform.weights)).to(x.dtype)

    @torch.no_grad()
    def load_fourier_weights(self, state):
        """Explicit weight-only migration; callers must start a fresh optimizer."""
        self.baseline.load_legacy(state)
        self.mixer.reset_residual()

    @torch.no_grad()
    def diagnostics(self):
        """Small factor Gram matrices give the residual norm without M squared."""
        if self.mixer.backend != "factorized":
            matrix = self.mixer.dense_matrix()
            return {"residual_frobenius": matrix.norm(), "coefficient_count": self.mixer.count}
        a, p = self.mixer.factors()
        a, p = a.flatten(0, 1), p.flatten(0, 1)
        norm2 = ((a.T @ a) * (p.T @ p)).sum().clamp_min(0)
        return {"residual_frobenius": norm2.sqrt(), "coefficient_count": self.mixer.count,
                "mixing_rank": self.mixer.mixing_rank}
