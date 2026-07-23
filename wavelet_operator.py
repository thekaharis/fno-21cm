"""Dependency-free multilevel Haar wavelet operators for local model blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HaarWaveletOperator(nn.Module):
    """Mix channels independently in every wavelet scale and orientation.

    The transform is orthonormal and critically sampled. At each level the
    approximation band is decomposed again, while every detail orientation is
    retained and receives its own learned channel-mixing matrix. The inverse
    transform therefore returns the original spatial shape without padding.
    """

    def __init__(self, channels: int, ndim: int, levels: int = 2):
        super().__init__()
        self.channels = int(channels)
        self.ndim = int(ndim)
        self.levels = int(levels)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        if self.levels <= 0:
            raise ValueError("levels must be positive")

        detail_count = 2**self.ndim - 1
        scale = 1.0 / math.sqrt(self.channels)
        self.detail_weights = nn.ParameterList(
            nn.Parameter(
                scale * torch.randn(detail_count, self.channels, self.channels)
            )
            for _ in range(self.levels)
        )
        self.lowpass_weight = nn.Parameter(
            scale * torch.randn(self.channels, self.channels)
        )

    @staticmethod
    def _split_axis(
        x: torch.Tensor, axis: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        even_slice = [slice(None)] * x.ndim
        odd_slice = [slice(None)] * x.ndim
        even_slice[axis] = slice(0, None, 2)
        odd_slice[axis] = slice(1, None, 2)
        even = x[tuple(even_slice)]
        odd = x[tuple(odd_slice)]
        factor = math.sqrt(0.5)
        return (even + odd) * factor, (even - odd) * factor

    @staticmethod
    def _merge_axis(
        low: torch.Tensor, high: torch.Tensor, axis: int
    ) -> torch.Tensor:
        even = (low + high) * math.sqrt(0.5)
        odd = (low - high) * math.sqrt(0.5)
        shape = list(low.shape)
        shape[axis] *= 2
        output = low.new_empty(shape)
        even_slice = [slice(None)] * low.ndim
        odd_slice = [slice(None)] * low.ndim
        even_slice[axis] = slice(0, None, 2)
        odd_slice[axis] = slice(1, None, 2)
        output[tuple(even_slice)] = even
        output[tuple(odd_slice)] = odd
        return output

    def _decompose_once(self, x: torch.Tensor) -> list[torch.Tensor]:
        bands = [x]
        for axis in range(2, 2 + self.ndim):
            next_bands = []
            for band in bands:
                next_bands.extend(self._split_axis(band, axis))
            bands = next_bands
        return bands

    def _reconstruct_once(self, bands: list[torch.Tensor]) -> torch.Tensor:
        for axis in reversed(range(2, 2 + self.ndim)):
            bands = [
                self._merge_axis(bands[index], bands[index + 1], axis)
                for index in range(0, len(bands), 2)
            ]
        if len(bands) != 1:
            raise RuntimeError("invalid Haar subband structure")
        return bands[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.ndim + 2:
            raise ValueError(
                f"expected a {self.ndim + 2}-D channels-first tensor, "
                f"got shape {tuple(x.shape)}"
            )
        divisor = 2**self.levels
        if any(size % divisor for size in x.shape[-self.ndim:]):
            raise ValueError(
                f"spatial dimensions must be divisible by {divisor} for "
                f"{self.levels} Haar levels, got {tuple(x.shape[-self.ndim:])}"
            )

        low = x
        details = []
        for _ in range(self.levels):
            bands = self._decompose_once(low)
            low, level_details = bands[0], bands[1:]
            details.append(level_details)

        low = torch.einsum("bi...,io->bo...", low, self.lowpass_weight)
        for level in reversed(range(self.levels)):
            stacked = torch.stack(details[level], dim=1)
            mixed = torch.einsum(
                "bsi...,sio->bso...", stacked, self.detail_weights[level]
            )
            low = self._reconstruct_once(
                [low, *mixed.unbind(dim=1)]
            )
        return low

    def wavelet_weight_tensors(self) -> list[torch.Tensor]:
        """Return low-pass and per-level detail matrices for diagnostics."""
        return [self.lowpass_weight, *self.detail_weights]
