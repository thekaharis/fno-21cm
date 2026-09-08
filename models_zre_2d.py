"""2-D U-FNO and LocalFNO for the density -> z_re(x, y) map task.

These mirror the repo's 3-D architectures, reduced to the sky plane:

* :class:`UFNO2d` is Wen et al.'s U-FNO (see the vendored ``ufno.py`` and the
  ``models_ufno.UFNOWrapped`` adapter) with every ``*3d`` op replaced by its
  2-D counterpart: 3 Fourier blocks + 3 U-Fourier blocks (spectral conv +
  1x1 conv + mini U-Net), lifting/projection MLPs, sigmoid output. Both map
  axes are periodic transverse simulation axes, so padding is circular on
  X and Y (the 3-D version's non-periodic replicate padding applied to the
  LOS axis, which does not exist here).
* :class:`LocalFNO2d` is the windowed local-spectral U-Net of
  ``local_fno_3d.LocalFNO3d`` with Hann overlap-add windows, rank-projected
  quadrant spectral convolutions, and a global-FNO bottleneck, on 2-D maps.
  Periodicity again simplifies the window bookkeeping: every axis is gathered
  and scattered modulo the domain, so no validity masking is needed. Its two
  operator slots are filled from the :mod:`operators` registry, exactly as in
  the 3-D model: ``local_operator="wavelet"`` replaces only the windowed
  branches with multilevel Haar operators while the global bottleneck stays
  Fourier, ``"hadamard"`` does the same with truncated Walsh-Hadamard
  operators, and ``"cnn"`` in both slots gives a classical convolutional U-Net.

Both accept channels-first ``(B, C, H, W)`` and return ``(B, 1, H, W)``.
The sigmoid output matches the task: the z_re target is normalized to [0, 1].
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators import (
    build_operator,
    crop_to_original,
    operator_hyperparameters,
    operator_spec,
    pad_to_operator_size,
    resolve_operator_name,
    resolve_slot_operators,
    validate_operator,
)


#: Both map axes are periodic transverse simulation axes.
PAD_MODES_2D = ("circular", "circular")


def _pair(values: Sequence[int], name: str) -> tuple[int, int]:
    values = tuple(int(value) for value in values)
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain two positive integers")
    return values


def _group_norm(channels: int, requested_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _replace_bn_with_groupnorm(module: nn.Module, num_groups: int = 8) -> int:
    """2-D twin of ``models_ufno._replace_bn_with_groupnorm``."""
    n_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _group_norm(child.num_features, num_groups))
            n_replaced += 1
        else:
            n_replaced += _replace_bn_with_groupnorm(child, num_groups)
    return n_replaced


# =============================================================================
# U-FNO (2-D)
# =============================================================================


class SpectralConv2d(nn.Module):
    """2-D Fourier layer: rFFT, truncated per-mode linear transform, irFFT.

    Two signed-frequency blocks along the first axis (the rFFT already halves
    the second), matching the four-quadrant structure of the vendored
    ``SpectralConv3d``.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        self.weights1 = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = (int(value) for value in x.shape[-2:])
        limits = (height // 2, width // 2 + 1)
        if self.modes1 > limits[0] or self.modes2 > limits[1]:
            raise ValueError(
                f"modes=({self.modes1}, {self.modes2}) exceed FFT limits "
                f"{limits} for spatial shape {(height, width)}"
            )
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class UNet2d(nn.Module):
    """2-D twin of Wen et al.'s ``U_net`` local-feature path."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 3, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = self._conv(input_channels, output_channels,
                                kernel_size, 2, dropout_rate)
        self.conv2 = self._conv(input_channels, output_channels,
                                kernel_size, 2, dropout_rate)
        self.conv2_1 = self._conv(input_channels, output_channels,
                                  kernel_size, 1, dropout_rate)
        self.conv3 = self._conv(input_channels, output_channels,
                                kernel_size, 2, dropout_rate)
        self.conv3_1 = self._conv(input_channels, output_channels,
                                  kernel_size, 1, dropout_rate)
        self.deconv2 = self._deconv(input_channels, output_channels)
        self.deconv1 = self._deconv(input_channels * 2, output_channels)
        self.deconv0 = self._deconv(input_channels * 2, output_channels)
        self.output_layer = nn.Conv2d(
            input_channels * 2, output_channels,
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2,
        )

    @staticmethod
    def _conv(in_planes: int, output_channels: int, kernel_size: int,
              stride: int, dropout_rate: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    @staticmethod
    def _deconv(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), dim=1)
        return self.output_layer(concat0)


class UFNO2d(nn.Module):
    """Channels-first 2-D U-FNO: 3 Fourier + 3 U-Fourier blocks + sigmoid.

    Parameters mirror ``models_ufno.UFNOWrapped`` (minus the third mode).
    Both spatial axes are periodic, so the anti-periodicity buffer uses
    circular padding on X and Y.
    """

    MIN_PAD = 8
    MULT_OF = 8   # the U-Net path downsamples 3x by stride 2

    @staticmethod
    def _pad_amount(n: int, mult_of: int = 8, min_pad: int = 8) -> int:
        target = ((n + min_pad + mult_of - 1) // mult_of) * mult_of
        return target - n

    def __init__(self, modes1: int, modes2: int, width: int,
                 in_channels: int, out_channels: int = 1,
                 sigmoid: bool = True,
                 norm: str = "batchnorm",
                 norm_num_groups: int = 8):
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError(
                "UFNO2d only supports out_channels=1 (fc2 = Linear(128, 1), "
                "matching the vendored 3-D body)."
            )
        norm = norm.lower()
        if norm not in ("batchnorm", "groupnorm"):
            raise ValueError(
                f"norm must be 'batchnorm' or 'groupnorm', got {norm!r}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.sigmoid = bool(sigmoid)
        self.norm = norm

        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.convs = nn.ModuleList(
            SpectralConv2d(self.width, self.width, modes1, modes2)
            for _ in range(6)
        )
        self.ws = nn.ModuleList(
            nn.Conv1d(self.width, self.width, 1) for _ in range(6)
        )
        self.unets = nn.ModuleList(
            UNet2d(self.width, self.width, 3, 0.0) for _ in range(3)
        )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        if norm == "groupnorm":
            _replace_bn_with_groupnorm(self, norm_num_groups)

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        batch, _, height, width = x.shape
        pad_x = self._pad_amount(height, self.MULT_OF, self.MIN_PAD)
        pad_y = self._pad_amount(width, self.MULT_OF, self.MIN_PAD)
        x = F.pad(x, (0, pad_y, 0, pad_x), mode="circular")
        size_x, size_y = height + pad_x, width + pad_y

        x = self.fc0(x.permute(0, 2, 3, 1))          # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)                    # (B, width, H, W)

        # Blocks 0-2: Fourier (x1 + x2); blocks 3-5: U-Fourier (x1 + x2 + x3
        # with the U-Net applied to the block input), ReLU after every block,
        # exactly as in the vendored SimpleBlock3d.forward.
        for block in range(6):
            x1 = self.convs[block](x)
            x2 = self.ws[block](
                x.reshape(batch, self.width, -1)
            ).reshape(batch, self.width, size_x, size_y)
            if block >= 3:
                x = x1 + x2 + self.unets[block - 3](x)
            else:
                x = x1 + x2
            x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc2(F.relu(self.fc1(x)))            # (B, H, W, 1)
        x = x.permute(0, 3, 1, 2)[:, :, :height, :width]
        return torch.sigmoid(x) if self.sigmoid else x


# =============================================================================
# LocalFNO (2-D)
# =============================================================================


class OverlapAddWindow2d:
    """Overlapping Hann windows with normalized overlap-add reconstruction.

    Both axes are periodic, so window gather/scatter is modulo the domain
    size and every sample is valid (no finite-axis masking, unlike the 3-D
    version's LOS handling).
    """

    def __init__(
        self,
        window_size: Sequence[int],
        stride: Sequence[int] | None = None,
        offset: Sequence[int] = (0, 0),
        chunk_size: int = 256,
    ):
        self.window_size = _pair(window_size, "window_size")
        self.stride = _pair(
            stride or tuple(value // 2 for value in self.window_size),
            "stride",
        )
        self.offset = tuple(int(value) for value in offset)
        if len(self.offset) != 2 or any(value < 0 for value in self.offset):
            raise ValueError("offset must contain two non-negative integers")
        if any(step > width for step, width in zip(self.stride, self.window_size)):
            raise ValueError("stride cannot exceed window_size")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _axis_starts(size: int, stride: int, offset: int) -> list[int]:
        # One negative-stride halo, as in the 3-D version: it guarantees the
        # zero endpoint of the periodic Hann window never underweights a
        # boundary; the overlap-add normalization absorbs any double count
        # from the periodic wrap.
        return list(range(-stride - offset, size, stride))

    def positions(self, spatial_shape: Sequence[int]) -> list[tuple[int, int]]:
        shape = _pair(spatial_shape, "spatial_shape")
        starts = [
            self._axis_starts(size, stride, offset)
            for size, stride, offset in zip(shape, self.stride, self.offset)
        ]
        return list(itertools.product(*starts))

    def _window(self, *, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
        factors = [
            torch.hann_window(size, periodic=True, device=device,
                              dtype=dtype).sqrt()
            for size in self.window_size
        ]
        return factors[0][:, None] * factors[1][None, :]

    def apply(
        self,
        x: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected input shape (B, C, H, W)")
        batch, _, nx, ny = x.shape
        positions = self.positions((nx, ny))
        analysis = self._window(device=x.device, dtype=x.dtype)
        synthesis = analysis
        window_power = analysis * synthesis
        output: torch.Tensor | None = None
        normalization = x.new_zeros(nx * ny)

        local_x = torch.arange(self.window_size[0], device=x.device)
        local_y = torch.arange(self.window_size[1], device=x.device)

        for first in range(0, len(positions), self.chunk_size):
            chunk = positions[first:first + self.chunk_size]
            starts = torch.tensor(chunk, device=x.device, dtype=torch.long)
            ix = (starts[:, 0, None] + local_x[None, :]).remainder(nx)
            iy = (starts[:, 1, None] + local_y[None, :]).remainder(ny)

            patches = x[:, :, ix[:, :, None], iy[:, None, :]]
            # Advanced indexing produces (B, C, P, Wx, Wy).
            patch_count = patches.shape[2]
            patches = patches.permute(0, 2, 1, 3, 4).reshape(
                batch * patch_count, x.shape[1], *self.window_size
            )
            transformed = transform(patches * analysis)
            if transformed.ndim != 4 or transformed.shape[0] != batch * patch_count:
                raise ValueError(
                    "window transform must return (B*patch_count, C, Wx, Wy)"
                )
            if tuple(transformed.shape[-2:]) != self.window_size:
                raise ValueError("window transform changed the spatial shape")
            out_channels = transformed.shape[1]
            transformed = (
                transformed.reshape(
                    batch, patch_count, out_channels, *self.window_size
                )
                * synthesis
            )

            if output is None:
                output = x.new_zeros(batch, out_channels, nx * ny)

            linear = ix[:, :, None] * ny + iy[:, None, :]
            index = linear.reshape(1, 1, -1).expand(batch, out_channels, -1)
            source = transformed.permute(0, 2, 1, 3, 4).reshape(
                batch, out_channels, -1
            )
            output.scatter_add_(2, index, source)

            norm_source = window_power[None, :, :].expand(patch_count, -1, -1)
            normalization.scatter_add_(
                0, linear.reshape(-1), norm_source.reshape(-1)
            )

        if output is None:
            raise RuntimeError("window grid produced no patches")
        eps = torch.finfo(x.dtype).eps
        return (
            output / normalization.clamp_min(eps).reshape(1, 1, -1)
        ).reshape(batch, output.shape[1], nx, ny)


class QuadrantSpectralConv2d(nn.Module):
    """Truncated rFFT convolution with two signed-X blocks (2-D quadrants)."""

    def __init__(self, channels: int, modes: Sequence[int]):
        super().__init__()
        self.channels = int(channels)
        self.n_modes = _pair(modes, "modes")
        scale = 1.0 / max(1, self.channels * self.channels)
        shape = (self.channels, self.channels, *self.n_modes)
        self.weights1 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _contract(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny = (int(value) for value in x.shape[-2:])
        mx, my = self.n_modes
        limits = (nx // 2, ny // 2 + 1)
        if mx > limits[0] or my > limits[1]:
            raise ValueError(
                f"modes={self.n_modes} exceeds FFT limits {limits} "
                f"for spatial shape {(nx, ny)}"
            )
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :mx, :my] = self._contract(
            x_ft[:, :, :mx, :my], self.weights1
        )
        out_ft[:, :, -mx:, :my] = self._contract(
            x_ft[:, :, -mx:, :my], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class QuadrantSpectralConv2dSiren(nn.Module):
    """Signed-X block spectral convolution with SIREN-generated weights.

    Fills the same positive/negative-X blocks as ``QuadrantSpectralConv2d``,
    but the per-mode channel-mixing weights come from two shared
    ``SirenWeightNetwork2d`` trunks (real/imaginary) evaluated at signed mode
    coordinates normalized by the retained band -- the truncation becomes a
    smooth learned function of the mode coordinate. Bias-free: the enclosing
    residual block's spatial convolution carries the bias.

    Inherits :class:`SpectralConv2dSiren`'s stability lessons for this task:
    ortho-normalized FFTs and a decoupled DC mode (the shared SIREN trunk
    otherwise lets the DC amplitude dominate every mode's gradient -- see the
    comments in ``SpectralConv2dSiren.forward``).

    ``forward`` accepts pre-materialized ``weights`` so the windowed
    overlap-add path evaluates the SIRENs once per block forward instead of
    once per patch chunk.
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
        self.channels = int(channels)
        self.n_modes = _pair(modes, "modes")
        weight_dim = self.channels * self.channels
        kwargs = dict(hidden_dim=hidden_dim, omega=omega, n_hidden=n_hidden,
                      feature_dim=feature_dim, ff_sigma=ff_sigma,
                      learnable_ff=learnable_ff)
        self.real_weight = SirenWeightNetwork2d(weight_dim, **kwargs)
        self.imag_weight = SirenWeightNetwork2d(weight_dim, **kwargs)

    def _block_coordinates(self, *, device, dtype) -> list[torch.Tensor]:
        mx, my = self.n_modes
        # Normalize by the retained band so the learned spectral function is
        # shared across coordinates regardless of the branch's mode count.
        scale_x = max(1, mx)
        scale_y = max(1, my - 1)
        positive_x = torch.arange(mx, device=device, dtype=dtype) / scale_x
        negative_x = torch.arange(-mx, 0, device=device, dtype=dtype) / scale_x
        positive_y = torch.arange(my, device=device, dtype=dtype) / scale_y

        def grid(kx: torch.Tensor) -> torch.Tensor:
            values = torch.meshgrid(kx, positive_y, indexing="ij")
            return torch.stack(values, dim=-1)

        return [grid(positive_x), grid(negative_x)]

    def _make_weight(self, coordinates: torch.Tensor) -> torch.Tensor:
        mx, my = self.n_modes
        shape = (mx, my, self.channels, self.channels)
        real = self.real_weight(coordinates).reshape(shape)
        imag = self.imag_weight(coordinates).reshape(shape)
        real = real.permute(2, 3, 0, 1).contiguous()
        imag = imag.permute(2, 3, 0, 1).contiguous()
        return torch.complex(real, imag)

    def materialize_weights(self, *, device, dtype) -> list[torch.Tensor]:
        """Evaluate the SIRENs once, yielding the two signed-X blocks."""
        return [
            self._make_weight(grid)
            for grid in self._block_coordinates(device=device, dtype=dtype)
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
        nx, ny = (int(value) for value in x.shape[-2:])
        mx, my = self.n_modes
        limits = (nx // 2, ny // 2 + 1)
        if mx > limits[0] or my > limits[1]:
            raise ValueError(
                f"modes={self.n_modes} exceeds FFT limits {limits} "
                f"for spatial shape {(nx, ny)}"
            )
        if weights is None:
            weights = self.materialize_weights(device=x.device, dtype=x.dtype)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        x_ft[:, :, 0, 0] = 0
        out_ft = torch.zeros_like(x_ft)
        blocks = (slice(0, mx), slice(-mx, None))
        for x_slice, weight in zip(blocks, weights, strict=True):
            out_ft[:, :, x_slice, :my] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, x_slice, :my],
                weight.to(dtype=x_ft.dtype),
            )
        return torch.fft.irfft2(out_ft, s=(nx, ny), norm="ortho")


class SpectralResidualBlock2d(nn.Module):
    """2-D twin of ``local_fno_3d.SpectralResidualBlock3d``.

    Hosts one registry operator, rank-projected when the operator declares it,
    and applied through the overlap-add window grid when the slot is windowed.
    ``siren=`` and ``wavelet_levels=`` remain accepted as the pre-registry
    spellings of ``operator="siren_fourier"`` and ``operator="wavelet"``.
    """

    def __init__(
        self,
        channels: int,
        modes: Sequence[int],
        spectral_rank: int,
        *,
        window_size: Sequence[int] | None = None,
        offset: Sequence[int] = (0, 0),
        patch_chunk_size: int = 256,
        operator: str = "fourier",
        operator_kwargs: Mapping | None = None,
        windowed: bool | None = None,
        pad_modes: Sequence[str] = PAD_MODES_2D,
        siren: dict | None = None,
        wavelet_levels: int | None = None,
    ):
        super().__init__()
        if siren is not None and wavelet_levels is not None:
            raise ValueError("siren and wavelet operators are mutually exclusive")
        if siren is not None:
            operator, operator_kwargs = "siren_fourier", dict(siren)
        elif wavelet_levels is not None:
            operator = "wavelet"
            operator_kwargs = {"levels": int(wavelet_levels)}

        self.operator_name = resolve_operator_name(operator)
        self.operator_kwargs = operator_hyperparameters(
            self.operator_name, operator_kwargs
        )
        spec = operator_spec(self.operator_name)
        self.pad_modes = tuple(pad_modes)

        rank = min(int(spectral_rank), int(channels))
        if rank <= 0:
            raise ValueError("spectral_rank must be positive")
        if not spec.rank_projected:
            rank = int(channels)
        # Submodules are registered in the pre-registry order -- projection in,
        # operator, projection out -- because optimizer state dicts are keyed
        # by parameter position, and runs resume from them.
        self.in_projection = (
            nn.Conv2d(channels, rank, kernel_size=1, bias=False)
            if spec.rank_projected
            else nn.Identity()
        )
        self.spectral = build_operator(
            self.operator_name,
            channels=rank,
            ndim=2,
            modes=modes,
            hyperparameters=self.operator_kwargs,
        )
        self.out_projection = (
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            if spec.rank_projected
            else nn.Identity()
        )
        self.spatial = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = _group_norm(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, kernel_size=1),
        )
        use_window = window_size is not None and (
            spec.windowed if windowed is None else bool(windowed)
        )
        self.window_grid = (
            OverlapAddWindow2d(
                window_size, offset=offset, chunk_size=patch_chunk_size
            )
            if use_window
            else None
        )

    def _apply_operator(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the operator on a shape it accepts, then restore the shape."""
        padded, amounts = pad_to_operator_size(
            x, self.operator_name, self.operator_kwargs, self.pad_modes
        )
        return crop_to_original(transform(padded), amounts)

    def forward(self, x: torch.Tensor, *, operator_transform=None) -> torch.Tensor:
        projected = self.in_projection(x)
        prepare = getattr(self.spectral, "materialize_transform", None)
        materialize = getattr(self.spectral, "materialize_weights", None)
        if prepare is not None:
            if operator_transform is None:
                shape = (self.window_grid.window_size if self.window_grid is not None
                         else projected.shape[2:])
                operator_transform = prepare(
                    shape, device=projected.device, dtype=projected.dtype
                )

            def transform(patches: torch.Tensor) -> torch.Tensor:
                return self.spectral(patches, transform=operator_transform)

        elif materialize is None:
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
            spectral = self._apply_operator(transform, projected)
        else:
            spectral = self.window_grid.apply(
                projected,
                lambda patches: self._apply_operator(transform, patches),
            )
        y = F.gelu(self.norm(self.out_projection(spectral) + self.spatial(x)))
        return y + self.mlp(y)


class LocalFNO2d(nn.Module):
    """2-D twin of ``local_fno_3d.LocalFNO3d``: fixed U-Net, pluggable slots.

    The four windowed encoder/decoder branches form the "local" slot and the
    two whole-map bottleneck blocks the "global" slot; each takes any operator
    from the :mod:`operators` registry. ``local_operator="wavelet"`` with the
    default Fourier bottleneck is the LocalWNO variant, ``"hadamard"`` the
    LocalWHNO one, and ``siren=True`` remains the legacy spelling of
    ``siren_fourier`` in both slots.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_width: int = 16,
        local_window: Sequence[int] = (16, 16),
        local_modes: Sequence[int] = (6, 6),
        global_modes: Sequence[int] = (16, 16),
        spectral_rank: int = 16,
        patch_chunk_size: int = 256,
        output_sigmoid: bool = True,
        siren: bool = False,
        siren_hidden_dim: int = 64,
        siren_omega: float = 30.0,
        siren_n_hidden: int = 1,
        siren_feature_dim: int = 16,
        siren_ff_sigma: float = 128.0,
        siren_learnable_ff: bool = True,
        local_operator: str = "fourier",
        global_operator: str = "fourier",
        local_operator_kwargs: Mapping | None = None,
        global_operator_kwargs: Mapping | None = None,
        local_windowed: bool | None = None,
        wavelet_levels: int = 2,
    ):
        super().__init__()
        width0 = int(base_width)
        if width0 <= 0:
            raise ValueError("base_width must be positive")
        width1, width2 = 2 * width0, 4 * width0
        window = _pair(local_window, "local_window")
        local_modes = _pair(local_modes, "local_modes")
        wavelet_levels = int(wavelet_levels)
        if wavelet_levels <= 0:
            raise ValueError("wavelet_levels must be positive")
        siren_spec = dict(
            hidden_dim=int(siren_hidden_dim),
            omega=float(siren_omega),
            n_hidden=int(siren_n_hidden),
            feature_dim=int(siren_feature_dim),
            ff_sigma=float(siren_ff_sigma),
            learnable_ff=bool(siren_learnable_ff),
        )
        (local_name, local_kwargs), (global_name, global_kwargs) = (
            resolve_slot_operators(
                local_operator,
                global_operator,
                local_operator_kwargs,
                global_operator_kwargs,
                siren=bool(siren),
                siren_kwargs=siren_spec,
                wavelet_levels=wavelet_levels,
            )
        )
        if any(value % 4 for value in window):
            raise ValueError(
                "local_window values must be divisible by four for 50% "
                "overlap and half-stride shifted grids"
            )
        local_spec = operator_spec(local_name)
        windowed = (
            local_spec.windowed if local_windowed is None else bool(local_windowed)
        )
        if windowed:
            # A windowed operator only ever sees exactly one window, so its
            # requirements can be checked once, here. The global slot's shape
            # is data-dependent and is padded per forward instead.
            validate_operator(
                local_name, window, local_modes, local_kwargs,
                context="local_window",
            )
        if int(spectral_rank) > width0:
            raise ValueError("spectral_rank cannot exceed base_width")
        shifted = tuple(value // 4 for value in window)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_width = width0
        self.local_window = window
        self.local_modes = local_modes
        self.global_modes = _pair(global_modes, "global_modes")
        self.spectral_rank = int(spectral_rank)
        self.patch_chunk_size = int(patch_chunk_size)
        self.output_sigmoid = bool(output_sigmoid)
        self.siren = bool(siren)
        self.local_operator = local_name
        self.global_operator = global_name
        self.local_operator_kwargs = dict(local_kwargs)
        self.global_operator_kwargs = dict(global_kwargs)
        self.local_windowed = windowed
        self.wavelet_levels = wavelet_levels

        local_block = dict(
            operator=local_name,
            operator_kwargs=local_kwargs,
            windowed=windowed,
            patch_chunk_size=patch_chunk_size,
        )
        global_block = dict(
            operator=global_name,
            operator_kwargs=global_kwargs,
        )

        self.lifting = nn.Conv2d(self.in_channels, width0, kernel_size=1)
        self.encoder0 = SpectralResidualBlock2d(
            width0, local_modes, spectral_rank,
            window_size=window, offset=(0, 0), **local_block,
        )
        self.down0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(width0, width1, kernel_size=1),
        )
        self.encoder1 = SpectralResidualBlock2d(
            width1, local_modes, spectral_rank,
            window_size=window, offset=shifted, **local_block,
        )
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(width1, width2, kernel_size=1),
        )
        self.bottleneck = nn.Sequential(
            SpectralResidualBlock2d(
                width2, self.global_modes, spectral_rank, **global_block,
            ),
            SpectralResidualBlock2d(
                width2, self.global_modes, spectral_rank, **global_block,
            ),
        )
        if global_name == "learned_waveform":
            from learned_waveform_operator import SharedWaveformBottleneck

            self.bottleneck = SharedWaveformBottleneck(*self.bottleneck)
        self.fuse1 = nn.Conv2d(width2 + width1, width1, kernel_size=1)
        self.decoder1 = SpectralResidualBlock2d(
            width1, local_modes, spectral_rank,
            window_size=window, offset=(0, 0), **local_block,
        )
        self.fuse0 = nn.Conv2d(width1 + width0, width0, kernel_size=1)
        self.decoder0 = SpectralResidualBlock2d(
            width0, local_modes, spectral_rank,
            window_size=window, offset=shifted, **local_block,
        )
        self.projection = nn.Sequential(
            nn.Conv2d(width0, 2 * width0, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * width0, self.out_channels, kernel_size=1),
        )

    @staticmethod
    def _upsample(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=target.shape[-2:], mode="bilinear", align_corners=False
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


# ---------------------------------------------------------------------------
# SirenFNO2d: legacy.arch.siren_fno_3d.SirenFNO3d reduced to the sky plane.  The SIREN
# machinery (Fourier-feature mapping -> sine MLP -> dense channel-mixing
# weights per retained mode) is unchanged apart from 2-D mode coordinates;
# rfft2 keeps signed frequencies on X and the non-negative half on Y, so the
# layer fills two signed-X blocks instead of the 3-D version's four X/Y
# quadrants.  Both map axes are periodic, so there is no padding at all.
# ---------------------------------------------------------------------------


class FourierFeatureMapping2d(nn.Module):
    """Encode signed 2-D Fourier-mode coordinates with random Fourier features."""

    def __init__(
        self,
        feature_dim: int = 16,
        sigma: float = 128.0,
        learnable: bool = True,
    ):
        super().__init__()
        if feature_dim <= 0 or feature_dim % 2:
            raise ValueError("feature_dim must be a positive even integer")
        self.feature_dim = int(feature_dim)
        projection = torch.randn(2, self.feature_dim // 2) * float(sigma)
        self.projection = nn.Parameter(projection, requires_grad=bool(learnable))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        phase = torch.matmul(coordinates, self.projection) * math.pi
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class SirenSineLayer(nn.Module):
    """Bias-free SIREN layer with a learnable per-feature frequency scale."""

    def __init__(self, in_features: int, out_features: int, omega: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.omega = nn.Parameter(torch.full((out_features,), float(omega)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x) * self.omega)


class SirenWeightNetwork2d(nn.Module):
    """Map signed 2-D mode coordinates to dense channel-mixing weights."""

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        feature_dim: int = 16,
        ff_sigma: float = 128.0,
        learnable_ff: bool = True,
    ):
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be at least 1")
        self.mapping = FourierFeatureMapping2d(
            feature_dim=feature_dim, sigma=ff_sigma, learnable=learnable_ff,
        )
        self.first = SirenSineLayer(feature_dim, hidden_dim, omega)
        self.hidden = nn.ModuleList(
            SirenSineLayer(hidden_dim, hidden_dim, omega)
            for _ in range(n_hidden - 1)
        )
        self.last = nn.Linear(hidden_dim, out_dim, bias=False)
        with torch.no_grad():
            self.first.linear.weight.uniform_(-1.0 / feature_dim,
                                              1.0 / feature_dim)
            bound = math.sqrt(6.0 / hidden_dim) / float(omega)
            for layer in self.hidden:
                layer.linear.weight.uniform_(-bound, bound)
            self.last.weight.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = self.first(self.mapping(coordinates))
        for layer in self.hidden:
            x = layer(x)
        return self.last(x)


class SpectralConv2dSiren(nn.Module):
    """Truncated 2-D Fourier convolution with SIREN-generated weights.

    ``rfft2`` stores signed frequencies on X and the non-negative half on Y,
    so this layer fills a positive-X and a negative-X block over the retained
    non-negative Y range.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Sequence[int],
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        feature_dim: int = 16,
        ff_sigma: float = 128.0,
        learnable_ff: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(n_modes) != 2:
            raise ValueError("n_modes must contain exactly two values")
        if any(int(mode) <= 0 for mode in n_modes):
            raise ValueError("all retained mode counts must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)
        self.dropout = float(dropout)
        weight_dim = self.in_channels * self.out_channels
        kwargs = dict(hidden_dim=hidden_dim, omega=omega, n_hidden=n_hidden,
                      feature_dim=feature_dim, ff_sigma=ff_sigma,
                      learnable_ff=learnable_ff)
        self.real_weight = SirenWeightNetwork2d(weight_dim, **kwargs)
        self.imag_weight = SirenWeightNetwork2d(weight_dim, **kwargs)
        self.bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1))

    def _validate_modes(self, spatial_shape: Sequence[int]) -> None:
        nx, ny = (int(size) for size in spatial_shape)
        mx, my = self.n_modes
        limits = (nx // 2, ny // 2 + 1)
        if mx > limits[0] or my > limits[1]:
            raise ValueError(
                f"n_modes={self.n_modes} exceeds FFT limits {limits} "
                f"for spatial shape {(nx, ny)}"
            )

    def _block_coordinates(self, *, device, dtype) -> list[torch.Tensor]:
        mx, my = self.n_modes
        # Normalize by the retained band, not the input resolution, so the
        # learned spectral function is resolution-independent.
        scale_x = max(1, mx)
        scale_y = max(1, my - 1)
        positive_x = torch.arange(mx, device=device, dtype=dtype) / scale_x
        negative_x = torch.arange(-mx, 0, device=device, dtype=dtype) / scale_x
        positive_y = torch.arange(my, device=device, dtype=dtype) / scale_y

        def grid(kx: torch.Tensor) -> torch.Tensor:
            values = torch.meshgrid(kx, positive_y, indexing="ij")
            return torch.stack(values, dim=-1)

        return [grid(positive_x), grid(negative_x)]

    def _make_weight(self, coordinates: torch.Tensor) -> torch.Tensor:
        mx, my = self.n_modes
        shape = (mx, my, self.in_channels, self.out_channels)
        real = self.real_weight(coordinates).reshape(shape)
        imag = self.imag_weight(coordinates).reshape(shape)
        real = real.permute(2, 3, 0, 1).contiguous()
        imag = imag.permute(2, 3, 0, 1).contiguous()
        return torch.complex(real, imag)

    def spectral_weight_tensors(self) -> list[torch.Tensor]:
        """Materialize the two retained weight blocks for diagnostics."""
        parameter = next(self.parameters())
        return [
            self._make_weight(grid)
            for grid in self._block_coordinates(
                device=parameter.device, dtype=parameter.dtype
            )
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B,C,H,W), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        spatial_shape = tuple(int(size) for size in x.shape[-2:])
        self._validate_modes(spatial_shape)
        mx, my = self.n_modes
        # ortho normalization keeps every mode's amplitude O(pixel scale);
        # unnormalized FFT gives the DC mode ~Nx*Ny amplitude, which acts as
        # a ~2e4x effective LR on the output mean and destabilizes training
        # against the mostly-zero z_re target (saturation collapse -> NaN).
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        # Decouple the DC mode from the SIREN-generated weights: its ~Nx*Ny
        # amplitude otherwise dominates the shared weight-network's gradient
        # (every mode's backward sums into one trunk, unlike a plain FNO's
        # independent per-mode parameters), dragging the whole spectral map
        # toward mean-shifting directions until the sigmoid saturates. The
        # mean signal still reaches the output via the residual stream,
        # channel MLPs, and this layer's bias.
        x_ft[:, :, 0, 0] = 0
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            spatial_shape[0],
            spatial_shape[1] // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        blocks = (slice(0, mx), slice(-mx, None))
        for coordinate_grid, x_slice in zip(
            self._block_coordinates(device=x.device, dtype=x.dtype),
            blocks,
            strict=True,
        ):
            weight = self._make_weight(coordinate_grid).to(dtype=x_ft.dtype)
            out_ft[:, :, x_slice, :my] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, x_slice, :my], weight
            )
        out = torch.fft.irfft2(out_ft, s=spatial_shape, dim=(-2, -1),
                               norm="ortho")
        if self.dropout and self.training:
            out = F.dropout(out, p=self.dropout)
        return out + self.bias


class PointwiseMLP2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.first = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.last = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.first(x))
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)
        return self.last(x)


class SirenFNO2d(nn.Module):
    """Residual 2-D FNO with SIREN-generated truncated spectral weights.

    ``legacy.arch.siren_fno_3d.SirenFNO3d`` on the sky plane: both axes are periodic
    transverse simulation axes, so the 3-D version's LOS padding is dropped
    entirely.
    """

    def __init__(
        self,
        n_modes: Sequence[int],
        hidden_channels: int,
        in_channels: int,
        out_channels: int = 1,
        n_layers: int = 4,
        add_grid: bool = True,
        siren_hidden_dim: int = 64,
        siren_omega: float = 30.0,
        siren_n_hidden: int = 1,
        siren_feature_dim: int = 16,
        siren_ff_sigma: float = 128.0,
        siren_learnable_ff: bool = True,
        mlp_dropout: float = 0.0,
        output_sigmoid: bool = True,
        sigmoid_temperature: float = 2.0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if sigmoid_temperature <= 0:
            raise ValueError("sigmoid_temperature must be positive")
        self.n_modes = tuple(int(value) for value in n_modes)
        self.hidden_channels = int(hidden_channels)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_layers = int(n_layers)
        self.add_grid = bool(add_grid)
        self.output_sigmoid = bool(output_sigmoid)
        self.sigmoid_temperature = float(sigmoid_temperature)

        lift_channels = self.in_channels + (2 if self.add_grid else 0)
        self.lifting = nn.Conv2d(lift_channels, self.hidden_channels,
                                 kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv2dSiren(
                self.hidden_channels,
                self.hidden_channels,
                self.n_modes,
                hidden_dim=siren_hidden_dim,
                omega=siren_omega,
                n_hidden=siren_n_hidden,
                feature_dim=siren_feature_dim,
                ff_sigma=siren_ff_sigma,
                learnable_ff=siren_learnable_ff,
                dropout=mlp_dropout,
            )
            for _ in range(self.n_layers)
        )
        self.channel_mlps = nn.ModuleList(
            PointwiseMLP2d(
                self.hidden_channels,
                self.hidden_channels,
                4 * self.hidden_channels,
                dropout=mlp_dropout,
            )
            for _ in range(self.n_layers)
        )
        self.projection = PointwiseMLP2d(
            self.hidden_channels,
            self.out_channels,
            4 * self.hidden_channels,
        )

    @staticmethod
    def _make_grid(batch: int, nx: int, ny: int, *, device, dtype):
        axes = [
            torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
            for size in (nx, ny)
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, x: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if x is None:
            x = kwargs.get("x")
        if x is None:
            raise TypeError("SirenFNO2d.forward expected argument 'x'")
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B,C,H,W), got {tuple(x.shape)}"
            )
        batch, channels, nx, ny = x.shape
        if self.add_grid:
            if channels == self.in_channels:
                grid = self._make_grid(batch, nx, ny,
                                       device=x.device, dtype=x.dtype)
                x = torch.cat((x, grid), dim=1)
            elif channels != self.in_channels + 2:
                raise ValueError(
                    f"Expected {self.in_channels} or {self.in_channels + 2} "
                    f"channels, got {channels}"
                )
        elif channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}"
            )

        x = self.lifting(x)
        for index, (spectral, channel_mlp) in enumerate(
            zip(self.spectral_layers, self.channel_mlps, strict=True)
        ):
            x = x + channel_mlp(spectral(x))
            if index + 1 < self.n_layers:
                x = F.gelu(x)
        x = self.projection(x)
        if self.output_sigmoid:
            x = torch.sigmoid(x / self.sigmoid_temperature)
        return x
