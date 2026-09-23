"""Field-aware output activations and objectives on the shared architectures."""
from __future__ import annotations

from copy import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataset.fields import FieldMapping, FieldRegistry
from dataset.lightcone_params import PARAM_NAMES
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
        self.excursion_set = None
        es_channels = 0
        if config.excursion_set != "none":
            self.excursion_set = ExcursionSetFeatures.for_native_windows(config, mapping, window_config)
            es_channels = self.excursion_set.out_channels
        self.backbone = build_model(config, in_channels + context_channels + es_channels,
                                    out_channels=len(mapping.targets), output_sigmoid=False)

    def forward(self, x, context=None):
        if self.excursion_set is not None:
            x = torch.cat((x, self.excursion_set(x)), dim=1)
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


class ExcursionSetFeatures(nn.Module):
    """Soft sharp-k excursion set on the density window, as extra input channels.

    21cmFAST (hii_filter = sharp-k) ionizes a cell when the density smoothed by
    a sharp-k filter of radius R exceeds a barrier at ANY R. With the extended
    Press-Schechter collapsed fraction that barrier is B(R) = a - b s_R,
    s_R = sqrt(sigma^2_min - sigma^2_R), so two numbers per redshift fix it
    (tools_excursion_set_diagnostic.py: this shape reaches ~97% of a free
    per-radius barrier, and a varies smoothly with z within a cone).

    ``es``   a small MLP maps (1/(1+z), parameters) of every LOS slice to
             (a, b); margins m_R = (delta_R + b s_R - a)/T are combined by a
             log-sum-exp (a soft OR over radii) into a score. Channels:
             sigmoid(score) and tanh(score/8). T is learned.
    ``bank`` the raw filter bank delta_R / 0.3 (ablation: multi-scale inputs
             without the barrier or the OR).

    Radii are fixed and geometric in comoving Mpc. The cell size comes from the
    ``relative_los_Mpc/1000`` channel, the transverse planes are periodic, and
    the LOS is reflect-padded before the FFT so window ends do not wrap onto
    each other (the 32-cell halos absorb what remains of the edge effect).
    """
    # dataset.multifield: the "density" normalization rule is delta/10.
    DENSITY_SCALE = 10.0
    BANK_SCALE = 0.3          # ~ cell-scale std of the overdensity

    def __init__(self, mode, radii, los_pad, hidden, density_index, condition_indices,
                 relative_index):
        super().__init__()
        self.mode, self.los_pad = mode, int(los_pad)
        self.density_index, self.relative_index = int(density_index), int(relative_index)
        self.condition_indices = tuple(int(i) for i in condition_indices)
        self.register_buffer("radii", torch.as_tensor(radii, dtype=torch.float32), persistent=False)
        self._k_cache = {}
        if mode == "es":
            self.barrier = nn.Sequential(nn.Linear(len(self.condition_indices), hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
            # Start near the diagnostic's typical fit (physical delta units):
            # a ~ 0.3, b ~ 0.2, and a moderately sharp T.
            with torch.no_grad():
                self.barrier[-1].weight.mul_(0.01)
                self.barrier[-1].bias.copy_(torch.tensor([0.3, float(np.log(np.expm1(0.2)))]))
            self.log_temperature = nn.Parameter(torch.tensor(float(np.log(0.05))))
            self.out_channels = 2
        elif mode == "bank":
            self.out_channels = len(radii)
        else:
            raise ValueError(f"unknown excursion-set mode {mode!r}")

    @classmethod
    def for_native_windows(cls, config, mapping, window_config):
        if window_config is None:
            raise ValueError("excursion-set features need native LOS windows")
        if "density" not in mapping.inputs:
            raise ValueError("excursion-set features need density as an input")
        # Channel layout of dataset.los_windows.NativeLightconeDataset.
        n_in = len(mapping.inputs)
        n_params = len(PARAM_NAMES) if mapping.use_params else 0
        radii = np.geomspace(config.excursion_set_rmin, config.excursion_set_rmax,
                             config.excursion_set_radii)
        return cls(config.excursion_set, radii, config.excursion_set_los_pad,
                   config.excursion_set_hidden, mapping.inputs.index("density"),
                   range(n_in, n_in + 1 + n_params), n_in + 1 + n_params)

    def _wavenumber(self, shape, cell, device):
        key = (shape, round(cell, 9), device)
        if key not in self._k_cache:
            nx, ny, nz = shape
            f = [torch.fft.fftfreq(nx, device=device), torch.fft.fftfreq(ny, device=device),
                 torch.fft.rfftfreq(nz, device=device)]
            k = 2*np.pi/cell*torch.sqrt(f[0][:, None, None]**2 + f[1][None, :, None]**2
                                        + f[2][None, None, :]**2)
            self._k_cache = {key: k}
        return self._k_cache[key]

    def forward(self, x):
        rel = x[0, self.relative_index, 0, 0, :2]
        cell = float((rel[1]-rel[0]))*1000.0
        density = x[:, self.density_index].float()*self.DENSITY_SCALE
        n = density.shape[-1]
        pad = min(self.los_pad, n-1)
        padded = F.pad(density[:, None], (pad, pad, 0, 0, 0, 0), mode="reflect")[:, 0]
        spectrum = torch.fft.rfftn(padded, dim=(-3, -2, -1))
        k = self._wavenumber(tuple(padded.shape[-3:]), cell, x.device)
        bank = []
        for R in self.radii:
            smoothed = torch.fft.irfftn(spectrum*(k*R <= 1.0), s=padded.shape[-3:], dim=(-3, -2, -1))
            bank.append(smoothed[..., pad:pad+n])
        if self.mode == "bank":
            return torch.stack(bank, dim=1).to(x.dtype)/self.BANK_SCALE
        variance = torch.stack([b.var(dim=(-3, -2, -1)) for b in bank], dim=1)      # (B, N)
        s = torch.sqrt((variance[:, :1]-variance).clamp_min(0) + 1e-12)
        condition = x[:, self.condition_indices, 0, 0, :].transpose(1, 2).float()   # (B, Z, C)
        raw = self.barrier(condition)
        a = raw[..., 0][:, None, None, :]
        b = F.softplus(raw[..., 1])[:, None, None, :]
        temperature = self.log_temperature.exp()
        score = None
        for i, smoothed in enumerate(bank):
            margin = (smoothed + b*s[:, i, None, None, None] - a)/temperature
            score = margin if score is None else torch.logaddexp(score, margin)
        return torch.stack((torch.sigmoid(score), torch.tanh(score/8)), dim=1).to(x.dtype)


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
