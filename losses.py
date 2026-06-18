"""Small loss adapters used by the neuralop Trainer."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


class AbsoluteLoss:
    """Call a neuralop loss through its absolute-norm implementation."""

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **_):
        return self.loss.abs(out, y)


class RelativeLoss:
    """Call a neuralop loss through its relative-norm implementation."""

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **_):
        return self.loss.rel(out, y)


class WeightedLoss:
    """Combine ``(weight, loss)`` terms while preserving Trainer kwargs."""

    def __init__(self, *terms: tuple[float, Callable]):
        self.terms = tuple((float(weight), loss) for weight, loss in terms)

    def __call__(self, out, y, **kwargs):
        return sum(
            weight * loss(out, y, **kwargs)
            for weight, loss in self.terms
            if weight != 0.0
        )


class BinaryCrossEntropyTerm:
    """Voxel-mean BCE for neutral-fraction targets in ``[0, 1]``."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, out, y, **_):
        prediction = out.clamp(self.eps, 1.0 - self.eps)
        return torch.nn.functional.binary_cross_entropy(prediction, y)


class LightconeH1Loss:
    """Absolute H1 norm with periodic X/Y and interior-only LOS differences.

    The lightcone endpoints are physically unrelated, so Z must not wrap.
    One-sided endpoint stencils can nevertheless dominate the loss because
    derivatives are scaled by the inverse grid spacing. This implementation
    evaluates the Z derivative only where a centered stencil is available,
    while retaining the value term and periodic X/Y derivatives on the full
    cube.
    """

    def __init__(
        self,
        measure: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reduction: str = "sum",
    ):
        if len(measure) != 3:
            raise ValueError("LightconeH1Loss requires three domain measures")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.d = 3
        self.measure = tuple(float(value) for value in measure)
        self.reduction = reduction
        self.periodic_in_x = True
        self.periodic_in_y = True
        self.periodic_in_z = False

    def uniform_quadrature(self, x: torch.Tensor) -> tuple[float, float, float]:
        return tuple(
            self.measure[axis] / x.size(axis - 3)
            for axis in range(3)
        )

    def _reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return value.sum()
        return value.mean()

    def compute_terms(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Return flattened value/derivative terms used by :meth:`abs`.

        Terms 0, 1, and 2 cover the full cube. Term 3 contains only
        ``Z[1:-1]`` because centered LOS differences are undefined at the
        physical endpoints.
        """
        if x.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {x.shape} != {y.shape}"
            )
        if x.ndim < 3 or x.size(-1) < 3:
            raise ValueError(
                "LightconeH1Loss requires at least three spatial dimensions "
                "and three LOS cells"
            )
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        hx, hy, hz = (float(value) for value in quadrature)

        def terms(field: torch.Tensor) -> dict[int, torch.Tensor]:
            dx = (
                torch.roll(field, shifts=-1, dims=-3)
                - torch.roll(field, shifts=1, dims=-3)
            ) / (2.0 * hx)
            dy = (
                torch.roll(field, shifts=-1, dims=-2)
                - torch.roll(field, shifts=1, dims=-2)
            ) / (2.0 * hy)
            dz = (
                field[..., 2:] - field[..., :-2]
            ) / (2.0 * hz)
            return {
                0: torch.flatten(field, start_dim=-3),
                1: torch.flatten(dx, start_dim=-3),
                2: torch.flatten(dy, start_dim=-3),
                3: torch.flatten(dz, start_dim=-3),
            }

        return terms(x), terms(y)

    def abs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None = None,
        take_root: bool = True,
    ) -> torch.Tensor:
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        quadrature = tuple(float(value) for value in quadrature)
        terms_x, terms_y = self.compute_terms(x, y, quadrature)
        scale = math.prod(quadrature)
        error = sum(
            scale * torch.sum(
                (terms_x[index] - terms_y[index]).square(),
                dim=-1,
            )
            for index in range(4)
        )
        if take_root:
            error = error.sqrt()
        return self._reduce(error).squeeze()
