"""SIREN weight generation: coordinate MLPs that emit spectral weights.

Used by the ``siren_fourier`` slot of the local/global operator registry
(``operators.py``), which asks for a weight per Fourier mode instead of
storing one. The standalone SirenFNO architectures built on top of this live
in ``legacy/arch/``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatureMapping(nn.Module):
    """Encode signed Fourier-mode coordinates with random Fourier features."""

    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 16,
        sigma: float = 128.0,
        learnable: bool = True,
    ):
        super().__init__()
        if feature_dim <= 0 or feature_dim % 2:
            raise ValueError("feature_dim must be a positive even integer")
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        projection = torch.randn(self.input_dim, self.feature_dim // 2) * float(sigma)
        self.projection = nn.Parameter(projection, requires_grad=bool(learnable))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected coordinate dimension {self.input_dim}, "
                f"got {coordinates.shape[-1]}"
            )
        phase = torch.matmul(coordinates, self.projection) * math.pi
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class SineLayer(nn.Module):
    """Bias-free SIREN layer with a learnable per-feature frequency scale."""

    def __init__(self, in_features: int, out_features: int, omega: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.omega = nn.Parameter(torch.full((out_features,), float(omega)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x) * self.omega)


class SirenWeightNetwork(nn.Module):
    """Map signed 3-D mode coordinates to dense channel-mixing weights."""

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
        self.mapping = FourierFeatureMapping(
            input_dim=3,
            feature_dim=feature_dim,
            sigma=ff_sigma,
            learnable=learnable_ff,
        )
        self.first = SineLayer(feature_dim, hidden_dim, omega)
        self.hidden = nn.ModuleList(
            SineLayer(hidden_dim, hidden_dim, omega)
            for _ in range(n_hidden - 1)
        )
        self.last = nn.Linear(hidden_dim, out_dim, bias=False)
        self._initialize(feature_dim=feature_dim, hidden_dim=hidden_dim, omega=omega)

    def _initialize(self, feature_dim: int, hidden_dim: int, omega: float) -> None:
        with torch.no_grad():
            self.first.linear.weight.uniform_(-1.0 / feature_dim, 1.0 / feature_dim)
            bound = math.sqrt(6.0 / hidden_dim) / float(omega)
            for layer in self.hidden:
                layer.linear.weight.uniform_(-bound, bound)
            self.last.weight.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = self.first(self.mapping(coordinates))
        for layer in self.hidden:
            x = layer(x)
        return self.last(x)

