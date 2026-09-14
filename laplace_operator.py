"""Laplace Neural Operator as a local/global operator slot.

Port of the pole-residue layer from Cao, Goswami & Karniadakis (2023),
`github.com/qianyingcao/Laplace-Neural-Operator` (`PR2d`/`PR3d`), rewritten so
it fits our grids.

The reference implementation materializes two tensors of shape
``(N..., C, C, M...)`` and ``(C, C, M..., N...)`` -- ``O(C^2 * prod(M) *
prod(N))``. At their published 2-D size (50x50, C=16, M=4x4) that is 0.23 GB; at
ours (140x140, C=32, M=16x16) it is 115 GB for a single layer. Both tensors are
outer products over the mode axes, so the mode sums can be taken *before* the
grid contraction and neither is ever needed. This module does that; the result
is numerically identical to the reference einsums (see
``tests/check_laplace_equivalence.py``), at ``O(C^2 * max(prod(N), prod(M)))``.

Method, per axis ``a``, with ``lambda_a`` the DFT frequencies of that axis:

    alpha              = FFT(x)
    A_a[o,i,k,p]       = 1 / (lambda_a[o] - pole_a[i,k,p])
    transient          = IFFT( sum_i alpha[b,i,o..] * G[i,k,o..] ),
                         G = sum_{p..} residue[i,k,p..] * prod_a A_a[o_a,i,k,p_a]
    steady             = sum_{p..} residue * (alpha contracted with the same A),
                         evaluated on the grid through exp(pole_a * t_a)

`modes` here is the **number of poles per axis**, not a spectral cutoff as in
`fourier`/`frequency_mixing`. Nothing is truncated: every DFT frequency reaches
the output through ``A_a``. Pole count therefore trades capacity against cost
without discarding input frequencies, and the operator is resolution-flexible --
poles live on a normalized domain and carry no grid size.

Poles are stabilized as ``-|Re| + i*Im`` by default (`stable_poles`). The
reference leaves the real part free, which lets ``exp(pole * t)`` grow; on a
normalized domain with their initialization it stays bounded, but nothing
prevents training from pushing it positive. ``stable_poles=False`` reproduces
the published parameterization exactly.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

__all__ = ["LaplaceOperator", "validate_laplace_options", "validate_laplace_shape"]

# Channel/grid/mode index letters for the staged einsums.
_GRID = "rst"
_MODE = "xyz"


def validate_laplace_options(pole_count=None, stable_poles=True, channel_chunk=8):
    if not isinstance(stable_poles, bool):
        raise ValueError("stable_poles must be a bool")
    if int(channel_chunk) != channel_chunk or channel_chunk <= 0:
        raise ValueError("channel_chunk must be a positive integer")
    if pole_count is not None and (int(pole_count) != pole_count or pole_count <= 0):
        raise ValueError("pole_count must be a positive integer")


def validate_laplace_shape(sizes, modes, *, context="laplace"):
    """Poles are grid-independent, so only positivity and rank are required."""
    if len(modes) not in (2, 3) or len(sizes) != len(modes):
        raise ValueError(f"{context}: laplace requires 2-D or 3-D matching modes")
    if any(int(m) < 1 for m in modes) or any(int(n) < 1 for n in sizes):
        raise ValueError(f"{context}: pole counts {tuple(modes)} and sizes {tuple(sizes)} must be positive")


class LaplaceOperator(nn.Module):
    """Pole-residue operator mapping (B, C, *spatial) -> (B, C, *spatial)."""

    def __init__(self, channels: int, ndim: int, modes: Sequence[int], *,
                 pole_count: int | None = None, stable_poles: bool = True,
                 channel_chunk: int = 8):
        super().__init__()
        self.channels, self.ndim = int(channels), int(ndim)
        if self.ndim not in (2, 3):
            raise ValueError("laplace supports 2-D and 3-D")
        counts = tuple(int(pole_count or m) for m in modes)
        if len(counts) != self.ndim:
            raise ValueError("modes must match ndim")
        validate_laplace_options(pole_count, stable_poles, channel_chunk)
        validate_laplace_shape((1,) * self.ndim, counts)
        self.n_modes = counts
        self.stable_poles, self.channel_chunk = stable_poles, int(channel_chunk)
        # Reference initialization: uniform on the unit square, scaled by 1/C^2.
        scale = 1.0 / (self.channels * self.channels)
        self.poles = nn.ParameterList(
            nn.Parameter(scale * torch.rand(self.channels, self.channels, m, dtype=torch.cfloat))
            for m in counts
        )
        self.residue = nn.Parameter(
            scale * torch.rand(self.channels, self.channels, *counts, dtype=torch.cfloat)
        )

    def extra_repr(self):
        return (f"channels={self.channels}, poles={self.n_modes}, "
                f"stable_poles={self.stable_poles}")

    def _pole(self, axis: int) -> torch.Tensor:
        pole = self.poles[axis]
        if not self.stable_poles:
            return pole
        # Decaying transients only; preserves the reference init magnitude,
        # unlike a softplus reparameterization.
        return torch.complex(-pole.real.abs(), pole.imag)

    def _resolvents(self, sizes, *, device, dtype):
        """A_a[o, i, k, p] = 1 / (lambda_a[o] - pole_a[i, k, p])."""
        out = []
        for axis, size in enumerate(sizes):
            # fftfreq defaults to float32; at float64 that would silently cap
            # the frequency axis at single precision.
            real_dtype = torch.empty(0, dtype=dtype).real.dtype
            k = torch.fft.fftfreq(size, d=1.0 / size, device=device, dtype=real_dtype)
            lam = (2j * math.pi) * k.to(dtype)
            out.append(torch.reciprocal(lam.view(-1, 1, 1, 1) - self._pole(axis)))
        return out

    def _transient(self, alpha, resolvents, sizes):
        """sum over poles first, so the (N..., C, C, M...) tensor never forms."""
        grid, mode = _GRID[: self.ndim], _MODE[: self.ndim]
        out = alpha.new_zeros(alpha.shape)
        # Chunk the output channel so G stays O(C * chunk * prod(N)).
        for start in range(0, self.channels, self.channel_chunk):
            stop = min(start + self.channel_chunk, self.channels)
            partial = self.residue[:, start:stop]
            for axis in range(self.ndim):
                a = resolvents[axis][:, :, start:stop]
                # replace mode axis `axis` with grid axis `axis`
                lhs = f"ik{mode}"
                rhs = f"{_GRID[axis]}ik{_MODE[axis]}"
                res = f"ik{mode[:axis]}{_GRID[axis]}{mode[axis + 1:]}"
                partial = torch.einsum(f"{lhs},{rhs}->{res}", partial, a)
                mode = mode[:axis] + _GRID[axis] + mode[axis + 1:]
            mode = _MODE[: self.ndim]
            out[:, start:stop] = torch.einsum(
                f"bi{grid},ik{grid}->bk{grid}", alpha, partial)
        return out

    def _steady_coefficients(self, alpha, resolvents):
        """sum_{o..} alpha * prod_a A_a, contracted one axis at a time."""
        grid, mode = _GRID[: self.ndim], _MODE[: self.ndim]
        # (B, C_in, C_out, *grid) -- broadcast the output-channel axis in.
        work = alpha.unsqueeze(2)
        current = grid
        for axis in range(self.ndim):
            a = resolvents[axis]
            lhs = f"bik{current}"
            rhs = f"{_GRID[axis]}ik{_MODE[axis]}"
            res = f"bik{current[:axis]}{_MODE[axis]}{current[axis + 1:]}"
            work = torch.einsum(f"{lhs},{rhs}->{res}", work, a)
            current = current[:axis] + _MODE[axis] + current[axis + 1:]
        return torch.einsum(f"ik{mode},bik{mode}->bk{mode}", self.residue, work)

    def _steady_field(self, coefficients, sizes, *, device, dtype):
        """Evaluate sum_{p..} coeff * prod_a exp(pole_a * t_a) on the grid.

        Mirrors the reference contraction, whose output-channel axis pairs with
        the poles' *input*-channel axis. The two are equal here by construction.
        """
        grid, mode = _GRID[: self.ndim], _MODE[: self.ndim]
        # Integer arange cast straight to the complex dtype; building it in
        # float32 first would cap the whole operator at single precision.
        coords = [torch.arange(n, device=device).to(dtype) / n for n in sizes]
        # exp_a[i, k, p, n] for each axis
        exps = [torch.exp(torch.einsum("ikp,n->ikpn", self._pole(a), coords[a]))
                for a in range(self.ndim)]
        out = None
        for i in range(self.channels):           # the paired input-channel axis
            work = torch.einsum(
                f"b{mode},k{_MODE[0]}{_GRID[0]}->bk{mode[1:]}{_GRID[0]}",
                coefficients[:, i], exps[0][i])
            for axis in range(1, self.ndim):
                work = torch.einsum(
                    f"bk{_MODE[axis:self.ndim]}{grid[:axis]},k{_MODE[axis]}{_GRID[axis]}"
                    f"->bk{_MODE[axis + 1:self.ndim]}{grid[:axis]}{_GRID[axis]}",
                    work, exps[axis][i])
            out = work if out is None else out + work
        return out

    def forward(self, x, **_):
        if x.ndim != self.ndim + 2 or x.shape[1] != self.channels or not x.is_floating_point():
            raise ValueError("expected real (batch, channels, *spatial) input")
        sizes = tuple(int(n) for n in x.shape[2:])
        validate_laplace_shape(sizes, self.n_modes)
        spatial = tuple(range(2, 2 + self.ndim))
        # FFT and pole algebra stay in full precision under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            real_dtype = self.residue.real.dtype
            alpha = torch.fft.fftn(x.to(real_dtype), dim=spatial)
            complex_dtype = alpha.dtype
            resolvents = self._resolvents(sizes, device=x.device, dtype=complex_dtype)
            transient = torch.fft.ifftn(
                self._transient(alpha, resolvents, sizes), s=sizes, dim=spatial).real
            coefficients = self._steady_coefficients(alpha, resolvents)
            steady = self._steady_field(
                coefficients, sizes, device=x.device, dtype=complex_dtype).real
            steady = steady / float(torch.tensor(sizes).prod())
            return (transient + steady).to(x.dtype)
