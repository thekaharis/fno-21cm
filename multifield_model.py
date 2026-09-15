"""Field-aware output activations and objectives on the shared architectures."""
from __future__ import annotations

from copy import copy

import torch
from torch import nn
import torch.nn.functional as F

from dataset.fields import FieldMapping, FieldRegistry
from modeling import build_model


class MultiFieldModel(nn.Module):
    """The final projection has an independent output row for each scalar field.

    All preceding layers remain shared. This is equivalent to separate linear
    one-channel heads on the projection's shared hidden features, without an
    extra bottleneck or architecture-specific feature-extraction hooks.
    """
    def __init__(self, config, in_channels, mapping: FieldMapping, registry=None, window_config=None):
        super().__init__()
        registry = registry or FieldRegistry()
        mapping = FieldMapping.create(mapping.inputs, mapping.targets, mapping.conditioning, registry)
        self.mapping = mapping
        self.bounded = tuple(registry[name].bounded for name in mapping.targets)
        self.context_encoder = None
        context_channels = 0
        if window_config is not None and window_config.mode == "coarse_context":
            context_channels = 2*window_config.context_features
            self.context_encoder = SurroundingEncoder(in_channels, window_config.context_features,
                                                       window_config.context_factor)
        self.backbone = build_model(config, in_channels + context_channels,
                                    out_channels=len(mapping.targets), output_sigmoid=False)

    def forward(self, x, context=None):
        if self.context_encoder is not None:
            if context is None:
                raise ValueError("coarse_context model requires the surrounding input")
            x = torch.cat((x, self.context_encoder(context, x.shape[-3:])), dim=1)
        elif context is not None:
            raise ValueError("this model was configured without coarse context")
        raw = self.backbone(x)
        return torch.cat([torch.sigmoid(raw[:, i:i+1]) if bounded else raw[:, i:i+1]
                          for i, bounded in enumerate(self.bounded)], dim=1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # NeuralOperator's FNO inserts constructor metadata as a literal root
        # key even when nested inside this wrapper. PyTorch's recursive loader
        # bypasses the child's public loader, so remove that non-weight entry
        # here. Our checkpoint already records the complete ModelConfig.
        state_dict = copy(state_dict)
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class SurroundingEncoder(nn.Module):
    """Spatially aligned coarse features plus a pooled surrounding summary.

    Pooling the full surrounding region lets even distant context affect the
    fine output. The second branch samples features at fine voxel centers,
    preserving their position instead of broadcasting only one global vector.
    """
    def __init__(self, in_channels, features, factor):
        super().__init__()
        self.factor = factor
        self.lift = nn.Conv3d(in_channels, features, 1)
        self.mix = nn.Conv3d(features, features, 3)

    def forward(self, context, shape):
        value = F.gelu(self.lift(context))
        # Full transverse planes are periodic; LOS boundaries are not.
        padded = F.pad(value, (0, 0, 1, 1, 1, 1), mode="circular")
        padded = F.pad(padded, (1, 1, 0, 0, 0, 0), mode="replicate")
        value = F.gelu(self.mix(padded))
        axes = [2*(torch.arange(n, device=value.device, dtype=value.dtype)+0.5)/n-1
                for n in shape]
        axes[-1] = axes[-1]/self.factor
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((gz, gy, gx), dim=-1)[None].expand(value.shape[0], -1, -1, -1, -1)
        aligned = F.grid_sample(value, grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
        summary = value.mean(dim=(-3, -2, -1), keepdim=True).expand(-1, -1, *shape)
        return torch.cat((aligned, summary), dim=1)


def field_mse(prediction, target, mask=None):
    """Per-field normalized MSE, averaging samples and spatial dimensions."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("expected matching (batch, fields, X, Y, Z) tensors")
    errors = (prediction - target).square()
    if mask is None:
        return errors.mean(dim=(0, 2, 3, 4))
    mask = torch.broadcast_to(mask.to(device=errors.device, dtype=errors.dtype), errors.shape)
    count = mask.sum(dim=(0, 2, 3, 4))
    if torch.any(count <= 0):
        raise ValueError("each target must have valid supervised voxels")
    return (errors*mask).sum(dim=(0, 2, 3, 4))/count


def weighted_objective(prediction, target, weights, mask=None):
    errors = field_mse(prediction, target, mask)
    return (errors * weights).sum() / weights.sum()
