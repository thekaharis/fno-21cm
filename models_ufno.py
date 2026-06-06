"""Adapter around Wen et al.'s U-FNO (Net3d / SimpleBlock3d in ``ufno.py``).

Why a wrapper
-------------
The upstream ``ufno.py`` (vendored as-is from
https://github.com/gegewen/ufno) was written for a specific CO2 multiphase
flow problem with hardcoded 12 input channels and a channels-last tensor
convention.  Two small footguns also bite us:

  * ``SimpleBlock3d`` lifts via ``nn.Linear(12, width)`` -- assumes 12 input
    channels.  We have 13 (density / 10 + 1/(1+z) + 11 LHS params).
  * ``Net3d.forward`` returns ``x.squeeze()`` which silently collapses every
    size-1 dim.  At ``batch_size == 1`` and ``out_channels == 1`` it drops
    both, breaking downstream loss code that expects ``(B, 1, X, Y, Z)``.

This wrapper bridges the conventions used by the rest of our codebase
(channels-first ``(B, C, X, Y, Z)``, in_channels parameterised, sigmoid on
output so predictions live in ``[0, 1]``) without modifying ``ufno.py`` --
that file stays as a clean vendored copy of the upstream.

Architecture (unchanged from the paper):

    SimpleBlock3d
      fc0:  Linear(in_channels -> width)
      Block 0:  SpectralConv3d + 1x1Conv                         <- FNO block
      Block 1:  SpectralConv3d + 1x1Conv                         <- FNO block
      Block 2:  SpectralConv3d + 1x1Conv                         <- FNO block
      Block 3:  SpectralConv3d + 1x1Conv + U_net (mini 3-D U-Net) <- U-Fourier
      Block 4:  SpectralConv3d + 1x1Conv + U_net                  <- U-Fourier
      Block 5:  SpectralConv3d + 1x1Conv + U_net                  <- U-Fourier
      fc1:  Linear(width -> 128)
      fc2:  Linear(128 -> 1)
    sigmoid                                                        <- our addition
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ufno import SimpleBlock3d


class UFNOWrapped(nn.Module):
    """Channels-first U-FNO wrapper for the density -> x_HI lightcone task.

    Parameters
    ----------
    modes1, modes2, modes3 : int
        Number of Fourier modes per spatial axis.  Equivalent to neuralop's
        ``n_modes=(modes1, modes2, modes3)``.
    width : int
        Hidden channel width inside the U-FNO blocks (analog of neuralop's
        ``hidden_channels``).  All 6 spectral blocks share this width.
    in_channels : int
        Number of input channels (e.g. 13 with parameter conditioning,
        2 without).  Overrides the upstream ``SimpleBlock3d``'s hardcoded
        ``Linear(12, width)`` lifting layer.
    out_channels : int, default 1
        Number of output channels.  Hardcoded to 1 in the upstream
        ``SimpleBlock3d.fc2``; values other than 1 are not supported here
        (the patch surface would grow significantly).
    sigmoid : bool, default True
        If True, apply a sigmoid to the output so predictions live in
        ``[0, 1]``.  Physically motivated for x_HI; disable for fair
        comparison against a non-sigmoid baseline.

    Notes
    -----
    Forward pass conventions:
      Input:  ``(B, C=in_channels, X, Y, Z)`` -- channels-first
      Output: ``(B, 1, X, Y, Z)`` -- channels-first, optionally sigmoid

    Internally we permute to ``(B, X, Y, Z, C)`` for the U-FNO body, then
    permute back.  Boundary padding (replicate by 8 in Y/Z, zero by 8 in X)
    is preserved from the upstream ``Net3d.forward`` -- this mitigates the
    FFT periodicity assumption at the cube boundaries.
    """

    # The U-Net inside each U-Fourier block does 3 stride-2 downsamples
    # before its decoder, so every spatial dim must be divisible by 8 or
    # the encoder/decoder skip-connection cat throws a shape error
    # (e.g. 148 // 2 // 2 // 2 -> 18 then * 2 * 2 * 2 -> 144 != 148).
    # MIN_PAD reproduces the upstream's anti-periodicity buffer (8 cells on
    # each spatial axis); MULT_OF is the U-Net's downsample factor.
    MIN_PAD = 8
    MULT_OF = 8

    @staticmethod
    def _pad_amount(n: int) -> int:
        """Smallest p such that (n + p) is a multiple of MULT_OF and p >= MIN_PAD."""
        target = ((n + UFNOWrapped.MIN_PAD + UFNOWrapped.MULT_OF - 1)
                  // UFNOWrapped.MULT_OF) * UFNOWrapped.MULT_OF
        return target - n

    def __init__(self, modes1: int, modes2: int, modes3: int,
                 width: int, in_channels: int, out_channels: int = 1,
                 sigmoid: bool = True):
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError(
                "UFNOWrapped only supports out_channels=1 because the "
                "upstream SimpleBlock3d hardcodes fc2 = Linear(128, 1). "
                "Patch fc2 manually if you need a different output.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.sigmoid = bool(sigmoid)

        # Construct the upstream body and override the hardcoded input
        # lifting layer.  Everything else (spectral convs, U-Net blocks,
        # output MLPs) keeps the paper's defaults.
        self.body = SimpleBlock3d(modes1, modes2, modes3, width)
        self.body.fc0 = nn.Linear(self.in_channels, width)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Convert (B, C, X, Y, Z) -> (B, X, Y, Z, C) for the U-FNO body
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        B, X, Y, Z, _ = x.shape

        # Per-axis padding so each spatial dim ends up a multiple of 8 (U-Net
        # downsample requirement) AND has at least MIN_PAD cells of buffer
        # (anti-periodicity for the FFT, mirrors the upstream).  For our
        # production shape (140, 140, 256) this gives pad (12, 12, 8) ->
        # (152, 152, 264), all divisible by 8.
        pad_x = self._pad_amount(X)
        pad_y = self._pad_amount(Y)
        pad_z = self._pad_amount(Z)

        # First pass: replicate-pad Y and Z on the right.  F.pad order is
        # (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...).
        x = F.pad(x, (0, 0, 0, pad_z, 0, pad_y), mode="replicate")
        # Second pass: zero-pad X on the right.
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_x), mode="constant", value=0.0)

        # Run the U-FNO body.  SimpleBlock3d returns (B, X+pad_x, Y+pad_y,
        # Z+pad_z, 1) -- the final fc2 already projects to 1 output channel.
        x = self.body(x)

        # Trim the padding back to the original spatial extent.  We replace
        # the upstream Net3d's footgun:
        #     return x.view(...)[..., :-8, :-8, :-8, :].squeeze()
        # (which collapses both batch and channel dims at batch=1).
        x = x[:, :X, :Y, :Z, :]

        if self.sigmoid:
            x = torch.sigmoid(x)

        # Back to channels-first (B, 1, X, Y, Z) for the rest of the pipeline.
        return x.permute(0, 4, 1, 2, 3).contiguous()

    # ----------------------------------------------------------- checkpoint
    def save_checkpoint(self, save_folder, save_name: str) -> None:
        """Match neuralop.models.BaseModel.save_checkpoint's contract.

        The neuralop Trainer calls this in single-process mode (DDP path
        saves ``model.module.state_dict()`` directly).  We only need the
        state-dict file -- the metadata file is used by ``from_checkpoint``
        which we don't call from anywhere.
        """
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        path = save_folder / f"{save_name}_state_dict.pt"
        torch.save(self.state_dict(), path.as_posix())

    def load_checkpoint(self, save_folder, save_name: str,
                        map_location=None) -> None:
        """Mirror of BaseModel.load_checkpoint."""
        save_folder = Path(save_folder)
        path = save_folder / f"{save_name}_state_dict.pt"
        self.load_state_dict(torch.load(path.as_posix(),
                                        map_location=map_location,
                                        weights_only=False))
