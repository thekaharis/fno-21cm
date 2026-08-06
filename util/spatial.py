"""Spatial boundary helpers shared by 3-D operator architectures."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_lightcone_spatial(
    x: torch.Tensor,
    pad_x: int,
    pad_y: int,
    pad_z: int,
) -> torch.Tensor:
    """Pad channels-first cubes with periodic X/Y and non-periodic Z."""
    if pad_x or pad_y:
        x = F.pad(
            x,
            (0, 0, 0, int(pad_y), 0, int(pad_x)),
            mode="circular",
        )
    if pad_z:
        x = F.pad(x, (0, int(pad_z), 0, 0, 0, 0), mode="replicate")
    return x
