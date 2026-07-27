"""Field statistics for 2-D x_HI maps, shared by the analysis probes.

All functions take a batch-first ``(N, H, W)`` tensor and return one value per
sample (or a batch of maps, for :func:`lowpass`).

A note on measuring sharpness, because the obvious choice is wrong: mean
``|grad|`` does *not* distinguish a sharp edge from a blurred one.  Blurring a
monotonic step spreads the same total variation over more pixels, so the mean
barely moves.  Use :func:`peak_grad` (the steepness actually attained) or
:func:`width_px` (ramp pixels per unit of edge crossed) instead.
"""

from __future__ import annotations

import torch

# Values outside this band count as "settled"; inside it, as transition.
BAND_LO, BAND_HI = 0.1, 0.9


def _grad(field: torch.Tensor) -> torch.Tensor:
    dx = torch.roll(field, -1, dims=-2) - field
    dy = torch.roll(field, -1, dims=-1) - field
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def blur_frac(field: torch.Tensor) -> torch.Tensor:
    """Fraction of pixels in the intermediate band. Blurring inflates it."""
    return ((field > BAND_LO) & (field < BAND_HI)).flatten(1).float().mean(1)


def total_variation(field: torch.Tensor) -> torch.Tensor:
    return _grad(field).flatten(1).sum(1)


def edge_density(field: torch.Tensor) -> torch.Tensor:
    """Total variation per pixel -- how much boundary the field contains."""
    return total_variation(field) / field[0].numel()


def peak_grad(field: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """High quantile of |grad|: the steepness actually reached at boundaries."""
    return torch.quantile(_grad(field).flatten(1).float(), q, dim=1)


def width_px(field: torch.Tensor) -> torch.Tensor:
    """Transition width in pixels: ramp area divided by total variation."""
    band = ((field > BAND_LO) & (field < BAND_HI)).flatten(1).sum(1).float()
    return band / total_variation(field).clamp_min(1e-9)


def lowpass(field: torch.Tensor, k_cut: float, clamp: bool = True) -> torch.Tensor:
    """Gaussian low-pass at ``k_cut`` (cycles/pixel), optionally clamped to [0,1].

    Gaussian rather than a brick wall: an ideal cutoff rings, and the overshoot
    would dominate the very sharpness statistics being measured.  The clamp
    mimics a model's sigmoid output stage.
    """
    h, w = field.shape[-2:]
    ky = torch.fft.fftfreq(h, device=field.device)[:, None]
    kx = torch.fft.rfftfreq(w, device=field.device)[None, :]
    win = torch.exp(-0.5 * (ky**2 + kx**2) / max(k_cut, 1e-6) ** 2)
    out = torch.fft.irfft2(torch.fft.rfft2(field) * win, s=(h, w))
    return out.clamp(0.0, 1.0) if clamp else out


def rmse(a: torch.Tensor, b: torch.Tensor, per_sample: bool = False):
    d = (a - b) ** 2
    return (d.flatten(1).mean(1) if per_sample else d.mean()).sqrt()


def summary(field: torch.Tensor) -> dict[str, float]:
    """Mean sharpness descriptors, for printing next to another field's."""
    return {
        "blur_frac": float(blur_frac(field).mean()),
        "peak_grad": float(peak_grad(field).mean()),
        "width_px": float(width_px(field).mean()),
        "edge_density": float(edge_density(field).mean()),
    }
