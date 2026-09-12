"""Pluggable operator zoo for the local and global slots of the U-Net skeleton.

``LocalFNO2d``/``LocalFNO3d`` are a two-level U-Net whose encoder/decoder
branches ("local" slot, applied to Hann-windowed overlapping patches) and
bottleneck ("global" slot, applied to the whole field) are both built from the
same residual block. This module turns the operator inside that block into a
registry entry so either slot can host any of:

``fourier``
    Truncated rFFT convolution over signed quadrants -- the FNO of the
    ``localfno`` runs.
``siren_fourier``
    The same quadrants with SIREN-generated per-mode weights (``localsirenfno``).
``wavelet``
    Multilevel orthonormal Haar operator, every band retained (``localwno``).
``hadamard``
    Truncated Walsh-Hadamard operator in sequency order (``localwhno``), new
    here.
``cnn``
    A classical U-Net convolution path, the Wen et al. ``U_net`` local branch
    generalized to 2-D/3-D and to configurable depth.
``learned_waveform``
    Random learned bin tables, anti-aliased dilation, and orthonormal real QR
    transforms with per-mode channel mixing.

Every operator maps ``(B, C, *spatial) -> (B, C, *spatial)``. What differs is
declared on :class:`OperatorSpec`: whether the enclosing block should wrap it in
its rank-projection bottleneck, whether it belongs inside the overlap-add
window machinery, and what spatial sizes it accepts (the block pads and crops to
satisfy ``required_size``, which is how a power-of-two-only transform such as
the Walsh-Hadamard one can still run on the 35x35x64 bottleneck).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


_SPATIAL_LETTERS = "xyz"


def _next_multiple(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor)) * int(divisor)


def _next_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and not (value & (value - 1))


def _group_norm(channels: int, requested_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


# =============================================================================
# Walsh-Hadamard operator
# =============================================================================


class WalshHadamardOperator(nn.Module):
    """Truncated Walsh-Hadamard transform with per-mode channel mixing.

    The transform is real and orthonormal, so the inverse is the transpose of
    the same matrices. Coefficients are ordered by *sequency* (the number of
    sign changes of the Walsh function), the Walsh analogue of frequency, so
    retaining the first ``modes`` indices per axis keeps the smoothest
    components -- the direct counterpart of the FNO's mode truncation.

    Rather than the O(N log N) butterfly, each axis is contracted with a dense
    ``(modes, size)`` matrix holding exactly the retained Walsh functions. For
    the window sizes used here (16-64) that is both faster than the butterfly
    (it is one BLAS call instead of log2(N) strided copies) and cheaper than
    the equivalent rFFT, and it never materializes the full-size spectrum.

    Sizes must be powers of two along every transformed axis; the enclosing
    block pads to :meth:`required_size` and crops afterwards.
    """

    def __init__(
        self,
        channels: int,
        ndim: int,
        modes: Sequence[int],
        ordering: str = "sequency",
    ):
        super().__init__()
        self.channels = int(channels)
        self.ndim = int(ndim)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        modes = tuple(int(value) for value in modes)
        if len(modes) != self.ndim or any(value <= 0 for value in modes):
            raise ValueError(
                f"modes must contain {self.ndim} positive integers, got {modes}"
            )
        self.n_modes = modes
        self.ordering = str(ordering).lower()
        if self.ordering not in {"sequency", "natural"}:
            raise ValueError(
                f"ordering must be 'sequency' or 'natural', got {ordering!r}"
            )

        scale = 1.0 / math.sqrt(self.channels)
        self.weight = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, *self.n_modes)
        )
        # Basis matrices are deterministic functions of (size, modes), so they
        # are cached rather than registered: no state-dict keys to keep in sync
        # with a changing spatial shape, and nothing for DDP to broadcast.
        self._basis_cache: dict[tuple, torch.Tensor] = {}

    # -- basis construction --------------------------------------------------

    @staticmethod
    def _hadamard_matrix(
        size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Orthonormal natural-order (Kronecker) Hadamard matrix."""
        matrix = torch.ones(1, 1, device=device, dtype=dtype)
        core = torch.tensor(
            [[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype
        )
        while matrix.shape[0] < size:
            matrix = torch.kron(matrix, core)
        return matrix * size**-0.5

    @staticmethod
    def _sequency_order(size: int) -> list[int]:
        """Row indices sorting natural-order rows by number of sign changes.

        Sequency order is the bit-reversal of the Gray code of the natural
        index; the resulting rows have 0, 1, 2, ... sign changes.
        """
        bits = size.bit_length() - 1
        order = []
        for index in range(size):
            gray = index ^ (index >> 1)
            order.append(
                int(format(gray, f"0{bits}b")[::-1], 2) if bits else 0
            )
        return order

    def _basis(
        self,
        size: int,
        modes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (size, modes, self.ordering, device, dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        if not _is_power_of_two(size):
            raise ValueError(
                f"Walsh-Hadamard axis length must be a power of two, got {size}"
            )
        if modes > size:
            raise ValueError(
                f"modes={modes} exceeds the Walsh-Hadamard axis length {size}"
            )
        matrix = self._hadamard_matrix(size, device=device, dtype=dtype)
        if self.ordering == "sequency":
            order = torch.tensor(
                self._sequency_order(size), device=device, dtype=torch.long
            )
            matrix = matrix.index_select(0, order)
        basis = matrix[:modes].contiguous()
        self._basis_cache[key] = basis
        return basis

    def _bases(self, x: torch.Tensor) -> list[torch.Tensor]:
        sizes = tuple(int(value) for value in x.shape[-self.ndim:])
        return [
            self._basis(size, modes, device=x.device, dtype=x.dtype)
            for size, modes in zip(sizes, self.n_modes)
        ]

    @staticmethod
    def required_size(size: int, hyperparameters: Mapping[str, Any]) -> int:
        return _next_power_of_two(size)

    # -- forward -------------------------------------------------------------

    def _contract(
        self,
        x: torch.Tensor,
        bases: list[torch.Tensor],
        *,
        analysis: bool,
    ) -> torch.Tensor:
        """Apply the per-axis basis (analysis) or its transpose (synthesis)."""
        first_spatial = x.ndim - self.ndim
        for axis, basis in enumerate(bases):
            position = first_spatial + axis
            contracted = 1 if analysis else 0
            x = torch.tensordot(
                x, basis, dims=([position], [contracted])
            ).movedim(-1, position)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.ndim + 2:
            raise ValueError(
                f"expected a {self.ndim + 2}-D channels-first tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} input channels, got {x.shape[1]}"
            )
        bases = self._bases(x)
        coefficients = self._contract(x, bases, analysis=True)
        letters = _SPATIAL_LETTERS[: self.ndim]
        mixed = torch.einsum(
            f"bi{letters},io{letters}->bo{letters}",
            coefficients,
            self._mixing_weight(device=x.device, dtype=x.dtype),
        )
        return self._contract(mixed, bases, analysis=False)

    def _mixing_weight(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """The (C, C, *modes) per-sequency mixing tensor.

        A hook rather than a direct ``self.weight`` read so that
        :class:`SirenWalshHadamardOperator` can generate it instead of storing
        it, without duplicating the transform.
        """
        return self.weight

    def walsh_weight_tensors(self) -> list[torch.Tensor]:
        """Return the per-sequency-mode mixing matrix for diagnostics."""
        parameter = next(self.parameters())
        return [self._mixing_weight(
            device=parameter.device, dtype=parameter.dtype
        )]



class SirenWalshHadamardOperator(WalshHadamardOperator):
    """Walsh-Hadamard transform whose per-sequency mixing is SIREN-generated.

    The Walsh analogue of :class:`local_fno_3d.QuadrantSpectralConv3dSiren`.
    ``WalshHadamardOperator`` stores one dense ``(C, C, *modes)`` parameter, so
    every retained sequency learns its channel mixing independently and the
    parameter count grows with the mode budget.  Here a single
    ``SirenWeightNetwork`` maps the *sequency coordinate* to that mixing
    matrix, making the truncation a smooth learned function of the frequency
    analogue rather than a set of unrelated per-mode parameters -- the same
    change SIREN makes to the FNO, applied to the Walsh basis.

    Two differences from the Fourier case, both simplifications:

    * The Walsh-Hadamard transform is **real**, so there is one trunk rather
      than the Fourier version's real/imaginary pair.
    * Sequency is **non-negative** by construction (it counts sign changes),
      so there are no signed quadrants -- one coordinate block covers the
      retained band, where the Fourier version needs four.

    Coordinates are normalized by the retained band, so the same learned
    function is evaluated at the same coordinates regardless of mode count,
    exactly as ``_quadrant_coordinates`` does for the Fourier operator.
    """

    def __init__(
        self,
        channels: int,
        ndim: int,
        modes: Sequence[int],
        ordering: str = "sequency",
        *,
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        feature_dim: int = 16,
        ff_sigma: float = 128.0,
        learnable_ff: bool = True,
    ):
        super().__init__(channels, ndim, modes, ordering)
        # Drop the dense parameter the base class registered: the SIREN
        # replaces it, and leaving it would train an unused tensor and put a
        # stale key in every checkpoint.
        del self._parameters["weight"]

        kwargs = dict(
            hidden_dim=hidden_dim,
            omega=omega,
            n_hidden=n_hidden,
            feature_dim=feature_dim,
            ff_sigma=ff_sigma,
            learnable_ff=learnable_ff,
        )
        if self.ndim == 3:
            from siren import SirenWeightNetwork as Trunk
        else:
            from models_zre_2d import SirenWeightNetwork2d as Trunk
        self.mixing_weight_net = Trunk(self.channels * self.channels, **kwargs)

    def _sequency_coordinates(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """(*modes, ndim) grid of band-normalized sequency coordinates.

        Sequency 0 is the constant Walsh function -- the DC term -- so the
        coordinate runs 0 -> 1 across the retained band and 0 keeps its
        meaning, unlike the Fourier operator's signed range.
        """
        axes = [
            torch.arange(m, device=device, dtype=dtype) / max(1, m - 1)
            for m in self.n_modes
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    def _mixing_weight(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coordinates = self._sequency_coordinates(device=device, dtype=dtype)
        shape = (*self.n_modes, self.channels, self.channels)
        weight = self.mixing_weight_net(coordinates).reshape(shape)
        # (*modes, C_in, C_out) -> (C_in, C_out, *modes) for the forward einsum
        order = (self.ndim, self.ndim + 1, *range(self.ndim))
        return weight.permute(*order).contiguous()


# =============================================================================
# Convolutional U-Net operator
# =============================================================================


class ConvUNetOperator(nn.Module):
    """Classical U-Net convolution path, shape-preserving.

    ``depth=3`` reproduces the mini U-Net of Wen et al.'s U-FNO (``ufno.U_net``
    and its 2-D twin ``models_zre_2d.UNet2d``): stride-2 convolutions down,
    transposed convolutions up, skip concatenations, and a final convolution on
    the input concatenated with the decoder output. Depth is configurable here
    so the operator can be sized to a 16-cell window as well as to a whole
    field.

    Group normalization is the default rather than U-FNO's batch
    normalization: the 3-D pipeline trains at batch size 1, where batch
    statistics are degenerate.
    """

    def __init__(
        self,
        channels: int,
        ndim: int,
        *,
        depth: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
        norm: str = "groupnorm",
    ):
        super().__init__()
        self.channels = int(channels)
        self.ndim = int(ndim)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.norm = str(norm).lower()
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.kernel_size <= 0 or not self.kernel_size % 2:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.norm not in {"groupnorm", "batchnorm"}:
            raise ValueError(
                f"norm must be 'groupnorm' or 'batchnorm', got {norm!r}"
            )
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = dropout

        self.encoder = nn.ModuleList(
            self._encoder_stage(first=(level == 0)) for level in range(self.depth)
        )
        # The deepest decoder stage sees only the bottom feature map; every
        # shallower one also sees the matching skip, hence the doubled input.
        self.decoder = nn.ModuleList(
            self._deconv(self.channels if level == self.depth - 1
                         else 2 * self.channels)
            for level in range(self.depth)
        )
        self.output_layer = self._conv_class()(
            2 * self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
        )

    # -- construction helpers ------------------------------------------------

    def _conv_class(self):
        return nn.Conv2d if self.ndim == 2 else nn.Conv3d

    def _deconv_class(self):
        return nn.ConvTranspose2d if self.ndim == 2 else nn.ConvTranspose3d

    def _norm_layer(self) -> nn.Module:
        if self.norm == "groupnorm":
            return _group_norm(self.channels)
        return (nn.BatchNorm2d if self.ndim == 2 else nn.BatchNorm3d)(
            self.channels
        )

    def _conv(self, stride: int) -> nn.Sequential:
        return nn.Sequential(
            self._conv_class()(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=(self.kernel_size - 1) // 2,
                bias=False,
            ),
            self._norm_layer(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.dropout),
        )

    def _encoder_stage(self, *, first: bool) -> nn.Sequential:
        # Wen et al. follow every downsampling convolution but the first with a
        # stride-1 refinement convolution.
        if first:
            return nn.Sequential(self._conv(stride=2))
        return nn.Sequential(self._conv(stride=2), self._conv(stride=1))

    def _deconv(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self._deconv_class()(
                in_channels,
                self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def required_size(size: int, hyperparameters: Mapping[str, Any]) -> int:
        return _next_multiple(size, 2 ** int(hyperparameters.get("depth", 3)))

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.ndim + 2:
            raise ValueError(
                f"expected a {self.ndim + 2}-D channels-first tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} input channels, got {x.shape[1]}"
            )
        skips = []
        features = x
        for stage in self.encoder:
            features = stage(features)
            skips.append(features)

        features = self.decoder[-1](skips[-1])
        for level in reversed(range(self.depth - 1)):
            features = self.decoder[level](
                torch.cat((skips[level], features), dim=1)
            )
        return self.output_layer(torch.cat((x, features), dim=1))


# =============================================================================
# Registry
# =============================================================================


@dataclass(frozen=True)
class OperatorSpec:
    """How one operator plugs into a residual block slot."""

    name: str
    #: ``(channels, ndim, modes, hyperparameters) -> nn.Module``
    build: Callable[..., nn.Module]
    #: Defaults for every hyperparameter the operator accepts.
    defaults: Mapping[str, Any] = field(default_factory=dict)
    #: Does the operator consume the block's truncated mode counts?
    uses_modes: bool = False
    #: Should the block wrap it in the rank-projection bottleneck?
    rank_projected: bool = True
    #: Default for "apply through the overlap-add window grid" in the local slot.
    windowed: bool = True
    #: ``(size, hyperparameters) -> padded size`` the operator can process.
    required_size: Callable[[int, Mapping[str, Any]], int] = (
        lambda size, hyperparameters: size
    )
    #: Optional extra validation of a *known* spatial shape at build time.
    validate: Callable[..., None] | None = None


def _build_fourier(channels, ndim, modes, hyperparameters):
    if ndim == 3:
        from local_fno_3d import QuadrantSpectralConv3d

        return QuadrantSpectralConv3d(channels, modes)
    from models_zre_2d import QuadrantSpectralConv2d

    return QuadrantSpectralConv2d(channels, modes)


def _build_siren_fourier(channels, ndim, modes, hyperparameters):
    if ndim == 3:
        from local_fno_3d import QuadrantSpectralConv3dSiren

        return QuadrantSpectralConv3dSiren(channels, modes, **hyperparameters)
    from models_zre_2d import QuadrantSpectralConv2dSiren

    return QuadrantSpectralConv2dSiren(channels, modes, **hyperparameters)


def _build_wavelet(channels, ndim, modes, hyperparameters):
    from wavelet_operator import HaarWaveletOperator

    return HaarWaveletOperator(
        channels, ndim=ndim, levels=int(hyperparameters["levels"])
    )


def _build_hadamard(channels, ndim, modes, hyperparameters):
    return WalshHadamardOperator(
        channels, ndim=ndim, modes=modes,
        ordering=str(hyperparameters["ordering"]),
    )


def _build_siren_hadamard(channels, ndim, modes, hyperparameters):
    values = dict(hyperparameters)
    return SirenWalshHadamardOperator(
        channels, ndim=ndim, modes=modes,
        ordering=str(values.pop("ordering")),
        **values,
    )


class IdentityOperator(nn.Module):
    """Pass the field through untouched: a slot that performs no mixing.

    Purpose is ablation. The local/global shell always applies both slots in
    sequence, so there is otherwise no way to ask what one basis contributes on
    its own -- the windowed local branch sits in front of the global transform
    and can compensate for whatever the basis gets wrong. Putting this in the
    local slot leaves the U-Net skeleton (lifting, down/up sampling, skips,
    projection) intact while removing all local spatial mixing, so
    ``identity/<basis>`` isolates the basis itself.

    It holds no parameters and ignores the mode counts, so the comparison
    between two such models differs only in the other slot.
    """

    def __init__(self, channels: int, ndim: int):
        super().__init__()
        self.channels = int(channels)
        self.ndim = int(ndim)

    def forward(self, x, **_):
        if x.ndim != self.ndim + 2 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected (B, {self.channels}, *{self.ndim} spatial axes), "
                f"got {tuple(x.shape)}"
            )
        return x


def _build_identity(channels, ndim, modes, hyperparameters):
    return IdentityOperator(channels, ndim)


def _build_cnn(channels, ndim, modes, hyperparameters):
    return ConvUNetOperator(
        channels,
        ndim=ndim,
        depth=int(hyperparameters["depth"]),
        kernel_size=int(hyperparameters["kernel_size"]),
        dropout=float(hyperparameters["dropout"]),
        norm=str(hyperparameters["norm"]),
    )


def _build_learned_waveform(channels, ndim, modes, hyperparameters):
    from learned_waveform_operator import LearnedWaveformOperator

    return LearnedWaveformOperator(channels, ndim, modes, **hyperparameters)


def _validate_learned_waveform(sizes, modes, hyperparameters, *, context):
    from learned_waveform_operator import validate_waveform_shape, validate_waveform_init, validate_waveform_transform

    validate_waveform_init(hyperparameters["init"])
    validate_waveform_transform(hyperparameters["transform"])
    bins = int(hyperparameters["bins"])
    limit = float(hyperparameters["condition_limit"])
    if bins < 3 or bins % 2 != 1:
        raise ValueError("waveform bins must be odd and at least 3")
    if not math.isfinite(limit) or limit <= 1:
        raise ValueError("waveform condition_limit must be finite and greater than 1")
    validate_waveform_shape(sizes, modes, context=context)


def _fft_limits(sizes: Sequence[int]) -> tuple[int, ...]:
    """Retained-mode ceilings of the quadrant rFFT convolutions."""
    return tuple(
        size // 2 if axis < len(sizes) - 1 else size // 2 + 1
        for axis, size in enumerate(sizes)
    )


def _validate_fourier(sizes, modes, hyperparameters, *, context):
    limits = _fft_limits(sizes)
    if any(mode > limit for mode, limit in zip(modes, limits)):
        raise ValueError(
            f"modes={tuple(modes)} exceeds the FFT limits {limits} of "
            f"{context}={tuple(sizes)}"
        )


def _validate_wavelet(sizes, modes, hyperparameters, *, context):
    levels = int(hyperparameters["levels"])
    if levels <= 0:
        raise ValueError("wavelet levels must be positive")
    divisor = 2**levels
    if any(size % divisor for size in sizes):
        raise ValueError(
            f"{context}={tuple(sizes)} must be divisible by {divisor} for "
            f"{levels} wavelet levels"
        )


def _validate_hadamard(sizes, modes, hyperparameters, *, context):
    if any(not _is_power_of_two(size) for size in sizes):
        raise ValueError(
            f"{context}={tuple(sizes)} must contain only powers of two for "
            "the Walsh-Hadamard operator"
        )
    if any(mode > size for mode, size in zip(modes, sizes)):
        raise ValueError(
            f"modes={tuple(modes)} exceeds the Walsh-Hadamard sequency "
            f"limits {tuple(sizes)} of {context}"
        )


def _validate_cnn(sizes, modes, hyperparameters, *, context):
    depth = int(hyperparameters["depth"])
    divisor = 2**depth
    if any(size % divisor for size in sizes):
        raise ValueError(
            f"{context}={tuple(sizes)} must be divisible by {divisor} for a "
            f"depth-{depth} CNN operator"
        )


OPERATORS: dict[str, OperatorSpec] = {
    "identity": OperatorSpec(
        name="identity",
        build=_build_identity,
        # No modes, no rank projection and no windowing: every one of those
        # would add parameters or cost to a slot whose whole point is to add
        # nothing. windowed=False also skips the patch loop entirely.
        uses_modes=False,
        rank_projected=False,
        windowed=False,
    ),
    "learned_waveform": OperatorSpec(
        name="learned_waveform",
        build=_build_learned_waveform,
        defaults={"bins": 31, "condition_limit": 1e4, "init": "random", "transform": "tied"},
        uses_modes=True,
        validate=_validate_learned_waveform,
    ),
    "fourier": OperatorSpec(
        name="fourier",
        build=_build_fourier,
        uses_modes=True,
        validate=_validate_fourier,
    ),
    "siren_fourier": OperatorSpec(
        name="siren_fourier",
        build=_build_siren_fourier,
        defaults={
            "hidden_dim": 64,
            "omega": 30.0,
            "n_hidden": 1,
            "feature_dim": 16,
            "ff_sigma": 128.0,
            "learnable_ff": True,
        },
        uses_modes=True,
        validate=_validate_fourier,
    ),
    "wavelet": OperatorSpec(
        name="wavelet",
        build=_build_wavelet,
        defaults={"levels": 2},
        required_size=lambda size, hyperparameters: _next_multiple(
            size, 2 ** int(hyperparameters.get("levels", 2))
        ),
        validate=_validate_wavelet,
    ),
    "hadamard": OperatorSpec(
        name="hadamard",
        build=_build_hadamard,
        defaults={"ordering": "sequency"},
        uses_modes=True,
        required_size=WalshHadamardOperator.required_size,
        validate=_validate_hadamard,
    ),
    "siren_hadamard": OperatorSpec(
        name="siren_hadamard",
        build=_build_siren_hadamard,
        defaults={
            "ordering": "sequency",
            "hidden_dim": 64,
            "omega": 30.0,
            "n_hidden": 1,
            "feature_dim": 16,
            "ff_sigma": 128.0,
            "learnable_ff": True,
        },
        uses_modes=True,
        required_size=WalshHadamardOperator.required_size,
        validate=_validate_hadamard,
    ),
    "cnn": OperatorSpec(
        name="cnn",
        build=_build_cnn,
        defaults={
            "depth": 3,
            "kernel_size": 3,
            "dropout": 0.0,
            "norm": "groupnorm",
        },
        rank_projected=False,
        windowed=False,
        required_size=ConvUNetOperator.required_size,
        validate=_validate_cnn,
    ),
}

#: Friendly spellings accepted wherever an operator name is read.
OPERATOR_ALIASES = {
    "none": "identity",
    "skip": "identity",
    "passthrough": "identity",
    "waveform": "learned_waveform",
    "orthogonal_waveform": "learned_waveform",
    "fno": "fourier",
    "fft": "fourier",
    "siren": "siren_fourier",
    "sirenfno": "siren_fourier",
    "haar": "wavelet",
    "wno": "wavelet",
    "siren_whno": "siren_hadamard",
    "sirenwhno": "siren_hadamard",
    "siren_walsh": "siren_hadamard",
    "walsh": "hadamard",
    "whno": "hadamard",
    "walsh_hadamard": "hadamard",
    "unet": "cnn",
    "conv": "cnn",
}


def operator_names() -> list[str]:
    return sorted(OPERATORS)


def resolve_operator_name(name: str) -> str:
    key = str(name).strip().lower()
    key = OPERATOR_ALIASES.get(key, key)
    if key not in OPERATORS:
        raise ValueError(
            f"unknown operator {name!r}; expected one of {operator_names()}"
        )
    return key


def operator_spec(name: str) -> OperatorSpec:
    return OPERATORS[resolve_operator_name(name)]


def operator_hyperparameters(
    name: str, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Merge user settings onto an operator's defaults, rejecting unknowns."""
    spec = operator_spec(name)
    merged = dict(spec.defaults)
    for key, value in dict(values or {}).items():
        if key not in merged:
            raise ValueError(
                f"operator {spec.name!r} has no hyperparameter {key!r}; "
                f"expected any of {sorted(merged) or ['(none)']}"
            )
        merged[key] = value
    return merged


def required_size(
    name: str, size: int, hyperparameters: Mapping[str, Any] | None = None
) -> int:
    """Smallest size >= ``size`` the operator can process."""
    spec = operator_spec(name)
    return int(spec.required_size(int(size), dict(hyperparameters or {})))


def validate_operator(
    name: str,
    sizes: Sequence[int],
    modes: Sequence[int] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    *,
    context: str = "",
) -> None:
    """Check a known spatial shape against an operator's requirements."""
    spec = operator_spec(name)
    if spec.validate is None:
        return
    merged = operator_hyperparameters(name, hyperparameters)
    spec.validate(
        tuple(int(value) for value in sizes),
        tuple(int(value) for value in (modes or ())),
        merged,
        context=context,
    )


def resolve_slot_operators(
    local_operator: str = "fourier",
    global_operator: str = "fourier",
    local_operator_kwargs: Mapping[str, Any] | None = None,
    global_operator_kwargs: Mapping[str, Any] | None = None,
    *,
    siren: bool = False,
    siren_kwargs: Mapping[str, Any] | None = None,
    wavelet_levels: int | None = None,
) -> tuple[tuple[str, dict], tuple[str, dict]]:
    """Resolve the two slots of a local/global U-Net to (name, hyperparameters).

    Also interprets the pre-registry spellings: ``siren=True`` promotes both
    slots to ``siren_fourier``, and ``wavelet_levels`` supplies the level count
    to whichever slot uses the Haar operator without one of its own.
    """
    local = resolve_operator_name(local_operator)
    global_ = resolve_operator_name(global_operator)
    local_values = dict(local_operator_kwargs or {})
    global_values = dict(global_operator_kwargs or {})

    if siren:
        if local == "wavelet":
            raise ValueError("siren cannot be combined with wavelet local layers")
        for slot, name in (("local", local), ("global", global_)):
            if name not in {"fourier", "siren_fourier"}:
                raise ValueError(
                    f"siren=True is the legacy spelling of "
                    f"siren_fourier and cannot be combined with a {name!r} "
                    f"{slot} operator"
                )
        local = global_ = "siren_fourier"
        local_values = dict(siren_kwargs or {})
        global_values = dict(siren_kwargs or {})

    if wavelet_levels is not None:
        for name, values in ((local, local_values), (global_, global_values)):
            if name == "wavelet":
                values.setdefault("levels", int(wavelet_levels))

    return (
        (local, operator_hyperparameters(local, local_values)),
        (global_, operator_hyperparameters(global_, global_values)),
    )


def build_operator(
    name: str,
    *,
    channels: int,
    ndim: int,
    modes: Sequence[int] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a registered operator for ``channels`` channels."""
    spec = operator_spec(name)
    merged = operator_hyperparameters(name, hyperparameters)
    if spec.uses_modes and modes is None:
        raise ValueError(f"operator {spec.name!r} requires retained mode counts")
    return spec.build(int(channels), int(ndim), modes, merged)


# =============================================================================
# Padding
# =============================================================================


def pad_to_operator_size(
    x: torch.Tensor,
    name: str,
    hyperparameters: Mapping[str, Any] | None,
    pad_modes: Sequence[str],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad the spatial axes up to what ``name`` accepts; report the amounts.

    ``pad_modes`` gives one padding mode per spatial axis so that periodic
    transverse axes wrap (``circular``) while the finite line-of-sight axis
    repeats its edge (``replicate``).
    """
    spec = operator_spec(name)
    merged = dict(hyperparameters or {})
    ndim = len(pad_modes)
    sizes = tuple(int(value) for value in x.shape[-ndim:])
    amounts = tuple(
        int(spec.required_size(size, merged)) - size for size in sizes
    )
    if not any(amounts):
        return x, amounts
    for axis, (amount, mode) in enumerate(zip(amounts, pad_modes)):
        remaining = amount
        while remaining > 0:
            # Circular and replicate padding both refuse to pad by more than
            # the current extent, so grow the axis in passes when the target
            # is more than double the input (e.g. 35 -> 64).
            step = min(remaining, x.shape[-ndim + axis])
            pad = [0] * (2 * ndim)
            # F.pad reads its pad list from the last axis backwards.
            pad[2 * (ndim - 1 - axis) + 1] = step
            x = F.pad(x, pad, mode=mode)
            remaining -= step
    return x, amounts


def crop_to_original(
    x: torch.Tensor, amounts: Sequence[int]
) -> torch.Tensor:
    """Undo :func:`pad_to_operator_size`."""
    if not any(amounts):
        return x
    ndim = len(amounts)
    slices = [slice(None)] * (x.ndim - ndim)
    slices.extend(
        slice(None, x.shape[-ndim + axis] - amount) if amount else slice(None)
        for axis, amount in enumerate(amounts)
    )
    return x[tuple(slices)]
