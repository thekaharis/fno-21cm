"""Small loss adapters used by the neuralop Trainer."""

from __future__ import annotations

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
