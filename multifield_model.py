"""Field-aware output activations and objectives on the shared architectures."""
from __future__ import annotations

import torch
from torch import nn

from dataset.fields import FieldMapping, FieldRegistry
from modeling import build_model


class MultiFieldModel(nn.Module):
    """The final projection has an independent output row for each scalar field.

    All preceding layers remain shared. This is equivalent to separate linear
    one-channel heads on the projection's shared hidden features, without an
    extra bottleneck or architecture-specific feature-extraction hooks.
    """
    def __init__(self, config, in_channels, mapping: FieldMapping, registry=None):
        super().__init__()
        registry = registry or FieldRegistry()
        mapping = FieldMapping.create(mapping.inputs, mapping.targets, mapping.conditioning, registry)
        self.mapping = mapping
        self.bounded = tuple(registry[name].bounded for name in mapping.targets)
        self.backbone = build_model(config, in_channels,
                                    out_channels=len(mapping.targets), output_sigmoid=False)

    def forward(self, x):
        raw = self.backbone(x)
        return torch.cat([torch.sigmoid(raw[:, i:i+1]) if bounded else raw[:, i:i+1]
                          for i, bounded in enumerate(self.bounded)], dim=1)


def field_mse(prediction, target):
    """Per-field normalized MSE, averaging samples and spatial dimensions."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("expected matching (batch, fields, X, Y, Z) tensors")
    return (prediction - target).square().mean(dim=(0, 2, 3, 4))


def weighted_objective(prediction, target, weights):
    errors = field_mse(prediction, target)
    return (errors * weights).sum() / weights.sum()
