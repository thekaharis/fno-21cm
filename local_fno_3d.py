"""Windowed local Fourier/wavelet U-Net for 3-D 21cm lightcone cubes."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in values)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain three positive integers")
    return values


def _group_norm(channels: int, requested_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class OverlapAddWindow3d:
    """Extract overlapping windows and reconstruct them by normalized overlap-add.

    X/Y are periodic and therefore gathered/scattered modulo the domain size.
    Z is finite: samples outside the domain use replicated input values but are
    masked during reconstruction. Processing is chunked over patch positions.
    """

    def __init__(
        self,
        window_size: Sequence[int],
        stride: Sequence[int] | None = None,
        offset: Sequence[int] = (0, 0, 0),
        chunk_size: int = 128,
    ):
        self.window_size = _triple(window_size, "window_size")
        self.stride = _triple(
            stride or tuple(value // 2 for value in self.window_size),
            "stride",
        )
        self.offset = tuple(int(value) for value in offset)
        if len(self.offset) != 3 or any(value < 0 for value in self.offset):
            raise ValueError("offset must contain three non-negative integers")
        if any(step > width for step, width in zip(self.stride, self.window_size)):
            raise ValueError("stride cannot exceed window_size")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _axis_starts(size: int, stride: int, offset: int) -> list[int]:
        # One negative-stride halo is always present. It prevents the zero
        # endpoint of the periodic Hann window from underweighting a boundary.
        return list(range(-stride - offset, size, stride))

    def positions(self, spatial_shape: Sequence[int]) -> list[tuple[int, int, int]]:
        shape = _triple(spatial_shape, "spatial_shape")
        starts = [
            self._axis_starts(size, stride, offset)
            for size, stride, offset in zip(shape, self.stride, self.offset)
        ]
        return list(itertools.product(*starts))

    def _window(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        factors = [
            torch.hann_window(
                size,
                periodic=True,
                device=device,
                dtype=dtype,
            ).sqrt()
            for size in self.window_size
        ]
        return (
            factors[0][:, None, None]
            * factors[1][None, :, None]
            * factors[2][None, None, :]
        )

    def apply(
        self,
        x: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected input shape (B, C, X, Y, Z)")
        batch, _, nx, ny, nz = x.shape
        positions = self.positions((nx, ny, nz))
        analysis = self._window(device=x.device, dtype=x.dtype)
        synthesis = analysis
        window_power = analysis * synthesis
        output: torch.Tensor | None = None
        normalization = x.new_zeros(nx * ny * nz)

        local_x = torch.arange(self.window_size[0], device=x.device)
        local_y = torch.arange(self.window_size[1], device=x.device)
        local_z = torch.arange(self.window_size[2], device=x.device)

        for first in range(0, len(positions), self.chunk_size):
            chunk = positions[first:first + self.chunk_size]
            starts = torch.tensor(chunk, device=x.device, dtype=torch.long)
            ix = (starts[:, 0, None] + local_x[None, :]).remainder(nx)
            iy = (starts[:, 1, None] + local_y[None, :]).remainder(ny)
            raw_z = starts[:, 2, None] + local_z[None, :]
            valid_z = (raw_z >= 0) & (raw_z < nz)
            iz = raw_z.clamp(0, nz - 1)

            patches = x[
                :,
                :,
                ix[:, :, None, None],
                iy[:, None, :, None],
                iz[:, None, None, :],
            ]
            # Advanced indexing produces (B, C, P, Wx, Wy, Wz).
            patch_count = patches.shape[2]
            patches = patches.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * patch_count,
                x.shape[1],
                *self.window_size,
            )
            transformed = transform(patches * analysis)
            if transformed.ndim != 5 or transformed.shape[0] != batch * patch_count:
                raise ValueError(
                    "window transform must return (B*patch_count, C, Wx, Wy, Wz)"
                )
            if tuple(transformed.shape[-3:]) != self.window_size:
                raise ValueError("window transform changed the spatial shape")
            out_channels = transformed.shape[1]
            transformed = (
                transformed.reshape(
                    batch,
                    patch_count,
                    out_channels,
                    *self.window_size,
                )
                * synthesis
            )

            if output is None:
                output = x.new_zeros(batch, out_channels, nx * ny * nz)

            linear = (
                (ix[:, :, None, None] * ny + iy[:, None, :, None]) * nz
                + iz[:, None, None, :]
            )
            valid = valid_z[:, None, None, :].expand_as(linear)
            index = linear.reshape(1, 1, -1).expand(batch, out_channels, -1)
            source = transformed.permute(0, 2, 1, 3, 4, 5).reshape(
                batch, out_channels, -1
            )
            source = source * valid.reshape(1, 1, -1)
            output.scatter_add_(2, index, source)

            norm_source = (
                window_power[None, :, :, :]
                .expand(patch_count, -1, -1, -1)
                * valid
            )
            normalization.scatter_add_(
                0,
                linear.reshape(-1),
                norm_source.reshape(-1),
            )

        if output is None:
            raise RuntimeError("window grid produced no patches")
        eps = torch.finfo(x.dtype).eps
        return (
            output / normalization.clamp_min(eps).reshape(1, 1, -1)
        ).reshape(batch, output.shape[1], nx, ny, nz)


class QuadrantSpectralConv3d(nn.Module):
    """Truncated rFFT convolution using four signed X/Y quadrants."""

    def __init__(
        self,
        channels: int,
        modes: Sequence[int],
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_modes = _triple(modes, "modes")
        scale = 1.0 / max(1, self.channels * self.channels)
        shape = (self.channels, self.channels, *self.n_modes)
        self.weights1 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _contract(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = (int(value) for value in x.shape[-3:])
        mx, my, mz = self.n_modes
        limits = (nx // 2, ny // 2, nz // 2 + 1)
        if any(mode > limit for mode, limit in zip(self.n_modes, limits)):
            raise ValueError(
                f"modes={self.n_modes} exceeds FFT limits {limits} "
                f"for spatial shape {(nx, ny, nz)}"
            )
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros_like(x_ft)
        blocks = (
            (slice(0, mx), slice(0, my), self.weights1),
            (slice(-mx, None), slice(0, my), self.weights2),
            (slice(0, mx), slice(-my, None), self.weights3),
            (slice(-mx, None), slice(-my, None), self.weights4),
        )
        for x_slice, y_slice, weight in blocks:
            out_ft[:, :, x_slice, y_slice, :mz] = self._contract(
                x_ft[:, :, x_slice, y_slice, :mz],
                weight,
            )
        return torch.fft.irfftn(
            out_ft,
            s=(nx, ny, nz),
            dim=(-3, -2, -1),
        )


class QuadrantSpectralConv3dSiren(nn.Module):
    """Quadrant spectral convolution with SIREN-generated weights.

    Fills the same four signed X/Y quadrants as ``QuadrantSpectralConv3d``,
    but the per-mode channel-mixing weights are produced by two shared
    ``SirenWeightNetwork`` trunks (real/imaginary) evaluated at signed mode
    coordinates normalized by the retained band, exactly following
    ``siren_fno_3d.SpectralConv3dSiren``. The truncation therefore becomes a
    smooth learned function of the mode coordinate instead of independent
    per-mode parameters. Bias-free: the enclosing residual block's spatial
    convolution carries the bias.

    ``forward`` accepts pre-materialized ``weights`` so that windowed
    overlap-add processing (which calls the transform once per patch chunk)
    evaluates the SIRENs a single time per block forward.
    """

    def __init__(
        self,
        channels: int,
        modes: Sequence[int],
        *,
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        feature_dim: int = 16,
        ff_sigma: float = 128.0,
        learnable_ff: bool = True,
    ):
        super().__init__()
        from siren_fno_3d import SirenWeightNetwork

        self.channels = int(channels)
        self.n_modes = _triple(modes, "modes")
        weight_dim = self.channels * self.channels
        kwargs = dict(
            hidden_dim=hidden_dim,
            omega=omega,
            n_hidden=n_hidden,
            feature_dim=feature_dim,
            ff_sigma=ff_sigma,
            learnable_ff=learnable_ff,
        )
        self.real_weight = SirenWeightNetwork(weight_dim, **kwargs)
        self.imag_weight = SirenWeightNetwork(weight_dim, **kwargs)

    def _quadrant_coordinates(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        mx, my, mz = self.n_modes
        # Normalize by the retained band so the same learned spectral
        # function is evaluated at the same coordinates in every branch
        # regardless of its mode count (cf. SpectralConv3dSiren).
        scale_x = max(1, mx)
        scale_y = max(1, my)
        scale_z = max(1, mz - 1)
        positive_x = torch.arange(mx, device=device, dtype=dtype) / scale_x
        negative_x = torch.arange(-mx, 0, device=device, dtype=dtype) / scale_x
        positive_y = torch.arange(my, device=device, dtype=dtype) / scale_y
        negative_y = torch.arange(-my, 0, device=device, dtype=dtype) / scale_y
        positive_z = torch.arange(mz, device=device, dtype=dtype) / scale_z

        def grid(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
            values = torch.meshgrid(kx, ky, positive_z, indexing="ij")
            return torch.stack(values, dim=-1)

        return [
            grid(positive_x, positive_y),
            grid(negative_x, positive_y),
            grid(positive_x, negative_y),
            grid(negative_x, negative_y),
        ]

    def _make_weight(self, coordinates: torch.Tensor) -> torch.Tensor:
        mx, my, mz = self.n_modes
        shape = (mx, my, mz, self.channels, self.channels)
        real = self.real_weight(coordinates).reshape(shape)
        imag = self.imag_weight(coordinates).reshape(shape)
        real = real.permute(3, 4, 0, 1, 2).contiguous()
        imag = imag.permute(3, 4, 0, 1, 2).contiguous()
        return torch.complex(real, imag)

    def materialize_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Evaluate the SIRENs once, yielding the four quadrant blocks."""
        return [
            self._make_weight(grid)
            for grid in self._quadrant_coordinates(device=device, dtype=dtype)
        ]

    def spectral_weight_tensors(self) -> list[torch.Tensor]:
        """Materialize the retained weight blocks for diagnostics."""
        parameter = next(self.parameters())
        return self.materialize_weights(
            device=parameter.device, dtype=parameter.dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        nx, ny, nz = (int(value) for value in x.shape[-3:])
        mx, my, mz = self.n_modes
        limits = (nx // 2, ny // 2, nz // 2 + 1)
        if any(mode > limit for mode, limit in zip(self.n_modes, limits)):
            raise ValueError(
                f"modes={self.n_modes} exceeds FFT limits {limits} "
                f"for spatial shape {(nx, ny, nz)}"
            )
        if weights is None:
            weights = self.materialize_weights(device=x.device, dtype=x.dtype)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros_like(x_ft)
        blocks = (
            (slice(0, mx), slice(0, my)),
            (slice(-mx, None), slice(0, my)),
            (slice(0, mx), slice(-my, None)),
            (slice(-mx, None), slice(-my, None)),
        )
        for (x_slice, y_slice), weight in zip(blocks, weights, strict=True):
            out_ft[:, :, x_slice, y_slice, :mz] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_ft[:, :, x_slice, y_slice, :mz],
                weight.to(dtype=x_ft.dtype),
            )
        return torch.fft.irfftn(
            out_ft,
            s=(nx, ny, nz),
            dim=(-3, -2, -1),
        )


class SpectralResidualBlock3d(nn.Module):
    """Rank-projected Fourier/wavelet residual block."""

    def __init__(
        self,
        channels: int,
        modes: Sequence[int],
        spectral_rank: int,
        *,
        window_size: Sequence[int] | None = None,
        offset: Sequence[int] = (0, 0, 0),
        patch_chunk_size: int = 128,
        siren: dict | None = None,
        wavelet_levels: int | None = None,
    ):
        super().__init__()
        if siren is not None and wavelet_levels is not None:
            raise ValueError("siren and wavelet operators are mutually exclusive")
        rank = min(int(spectral_rank), int(channels))
        if rank <= 0:
            raise ValueError("spectral_rank must be positive")
        self.in_projection = nn.Conv3d(channels, rank, kernel_size=1, bias=False)
        if wavelet_levels is not None:
            from wavelet_operator import HaarWaveletOperator

            self.spectral = HaarWaveletOperator(rank, ndim=3,
                                                levels=wavelet_levels)
        elif siren is not None:
            self.spectral = QuadrantSpectralConv3dSiren(rank, modes, **siren)
        else:
            self.spectral = QuadrantSpectralConv3d(rank, modes)
        self.out_projection = nn.Conv3d(rank, channels, kernel_size=1, bias=False)
        self.spatial = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = _group_norm(channels)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, 2 * channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(2 * channels, channels, kernel_size=1),
        )
        self.window_grid = (
            OverlapAddWindow3d(
                window_size,
                offset=offset,
                chunk_size=patch_chunk_size,
            )
            if window_size is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.in_projection(x)
        materialize = getattr(self.spectral, "materialize_weights", None)
        if materialize is None:
            transform = self.spectral
        else:
            # SIREN-generated weights depend only on the retained modes, so
            # evaluate the weight networks once per forward instead of once
            # per overlap-add patch chunk.
            weights = materialize(device=projected.device,
                                  dtype=projected.dtype)
            spectral_conv = self.spectral

            def transform(patches: torch.Tensor) -> torch.Tensor:
                return spectral_conv(patches, weights=weights)

        if self.window_grid is None:
            spectral = transform(projected)
        else:
            spectral = self.window_grid.apply(projected, transform)
        y = F.gelu(self.norm(self.out_projection(spectral) + self.spatial(x)))
        return y + self.mlp(y)


class LocalFNO3d(nn.Module):
    """Two-level local-spectral U-Net with a global FNO bottleneck.

    With ``siren=True`` every spectral branch (both windowed levels and the
    global bottleneck) generates its quadrant weights from per-branch SIREN
    networks instead of storing dense per-mode parameters -- the
    "LocalSirenFNO" variant. With ``local_operator="wavelet"``, only the four
    windowed branches use multilevel Haar operators; the whole-volume
    bottleneck remains Fourier, yielding the hybrid "LocalWNO" variant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_width: int = 16,
        local_window: Sequence[int] = (16, 16, 32),
        local_modes: Sequence[int] = (6, 6, 12),
        global_modes: Sequence[int] = (16, 16, 16),
        spectral_rank: int = 16,
        patch_chunk_size: int = 128,
        output_sigmoid: bool = True,
        siren: bool = False,
        siren_hidden_dim: int = 64,
        siren_omega: float = 30.0,
        siren_n_hidden: int = 1,
        siren_feature_dim: int = 16,
        siren_ff_sigma: float = 128.0,
        siren_learnable_ff: bool = True,
        local_operator: str = "fourier",
        wavelet_levels: int = 2,
    ):
        super().__init__()
        width0 = int(base_width)
        if width0 <= 0:
            raise ValueError("base_width must be positive")
        width1, width2 = 2 * width0, 4 * width0
        window = _triple(local_window, "local_window")
        local_modes = _triple(local_modes, "local_modes")
        local_operator = str(local_operator).lower()
        if local_operator not in {"fourier", "wavelet"}:
            raise ValueError("local_operator must be 'fourier' or 'wavelet'")
        if siren and local_operator == "wavelet":
            raise ValueError("siren cannot be combined with wavelet local layers")
        if any(value % 4 for value in window):
            raise ValueError(
                "local_window values must be divisible by four for 50% "
                "overlap and half-stride shifted grids"
            )
        limits = (window[0] // 2, window[1] // 2, window[2] // 2 + 1)
        if local_operator == "fourier" and any(
            mode > limit for mode, limit in zip(local_modes, limits)
        ):
            raise ValueError(
                f"local_modes={local_modes} exceeds window FFT limits {limits}"
            )
        wavelet_levels = int(wavelet_levels)
        wavelet_divisor = 2**wavelet_levels
        if wavelet_levels <= 0:
            raise ValueError("wavelet_levels must be positive")
        if local_operator == "wavelet" and any(
            value % wavelet_divisor for value in window
        ):
            raise ValueError(
                f"local_window values must be divisible by {wavelet_divisor} "
                f"for {wavelet_levels} wavelet levels"
            )
        if int(spectral_rank) > width0:
            raise ValueError("spectral_rank cannot exceed base_width")
        shifted = tuple(value // 4 for value in window)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_width = width0
        self.local_window = window
        self.local_modes = local_modes
        self.global_modes = _triple(global_modes, "global_modes")
        self.spectral_rank = int(spectral_rank)
        self.patch_chunk_size = int(patch_chunk_size)
        self.output_sigmoid = bool(output_sigmoid)
        self.siren = bool(siren)
        self.local_operator = local_operator
        self.wavelet_levels = wavelet_levels
        siren_spec = (
            dict(
                hidden_dim=int(siren_hidden_dim),
                omega=float(siren_omega),
                n_hidden=int(siren_n_hidden),
                feature_dim=int(siren_feature_dim),
                ff_sigma=float(siren_ff_sigma),
                learnable_ff=bool(siren_learnable_ff),
            )
            if self.siren
            else None
        )
        local_wavelet_levels = (
            self.wavelet_levels if self.local_operator == "wavelet" else None
        )

        self.lifting = nn.Conv3d(self.in_channels, width0, kernel_size=1)
        self.encoder0 = SpectralResidualBlock3d(
            width0, local_modes, spectral_rank,
            window_size=window, offset=(0, 0, 0),
            patch_chunk_size=patch_chunk_size, siren=siren_spec,
            wavelet_levels=local_wavelet_levels,
        )
        self.down0 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(width0, width1, kernel_size=1),
        )
        self.encoder1 = SpectralResidualBlock3d(
            width1, local_modes, spectral_rank,
            window_size=window, offset=shifted,
            patch_chunk_size=patch_chunk_size, siren=siren_spec,
            wavelet_levels=local_wavelet_levels,
        )
        self.down1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(width1, width2, kernel_size=1),
        )
        self.bottleneck = nn.Sequential(
            SpectralResidualBlock3d(
                width2, self.global_modes, spectral_rank, siren=siren_spec,
            ),
            SpectralResidualBlock3d(
                width2, self.global_modes, spectral_rank, siren=siren_spec,
            ),
        )
        self.fuse1 = nn.Conv3d(width2 + width1, width1, kernel_size=1)
        self.decoder1 = SpectralResidualBlock3d(
            width1, local_modes, spectral_rank,
            window_size=window, offset=(0, 0, 0),
            patch_chunk_size=patch_chunk_size, siren=siren_spec,
            wavelet_levels=local_wavelet_levels,
        )
        self.fuse0 = nn.Conv3d(width1 + width0, width0, kernel_size=1)
        self.decoder0 = SpectralResidualBlock3d(
            width0, local_modes, spectral_rank,
            window_size=window, offset=shifted,
            patch_chunk_size=patch_chunk_size, siren=siren_spec,
            wavelet_levels=local_wavelet_levels,
        )
        self.projection = nn.Sequential(
            nn.Conv3d(width0, 2 * width0, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(2 * width0, self.out_channels, kernel_size=1),
        )

    @staticmethod
    def _upsample(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=target.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        skip0 = self.encoder0(self.lifting(x))
        skip1 = self.encoder1(self.down0(skip0))
        x = self.bottleneck(self.down1(skip1))
        x = self.fuse1(torch.cat((self._upsample(x, skip1), skip1), dim=1))
        x = self.decoder1(x)
        x = self.fuse0(torch.cat((self._upsample(x, skip0), skip0), dim=1))
        x = self.decoder0(x)
        x = self.projection(x)
        return torch.sigmoid(x) if self.output_sigmoid else x
