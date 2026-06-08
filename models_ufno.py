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


def _replace_bn_with_groupnorm(module: nn.Module, num_groups: int = 8) -> int:
    """Walk *module* in-place, replacing every ``BatchNorm3d`` with a matching
    ``GroupNorm``.

    Returns the number of layers replaced (useful for sanity checking that
    the conversion did something on a model you expect to have BN in it).

    For a layer with ``num_channels`` channels we use
    ``min(num_groups, largest divisor of num_channels <= num_groups)`` groups.
    At the U-FNO's ``width=32`` and ``num_groups=8`` this picks 8 groups of 4
    channels each, which is the standard GroupNorm configuration.

    Cross-DDP behaviour: GroupNorm has no batch statistics, so
    ``SyncBatchNorm.convert_sync_batchnorm`` (called downstream of this in
    fno_21cm_3d.py) is a no-op on GroupNorm modules -- the v2 code path
    keeps the SyncBN call so the v1 path stays unchanged.
    """
    n_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            num_channels = child.num_features
            # Find the largest divisor of num_channels that's <= num_groups.
            ng = min(num_groups, num_channels)
            while ng > 1 and num_channels % ng != 0:
                ng -= 1
            new_norm = nn.GroupNorm(num_groups=ng, num_channels=num_channels)
            setattr(module, name, new_norm)
            n_replaced += 1
        else:
            n_replaced += _replace_bn_with_groupnorm(child, num_groups)
    return n_replaced


# =============================================================================
# Tier 2 U-Net path variants (D, E, F from the architectural-improvements menu)
# =============================================================================
# These replace the upstream Wen et al. ``U_net`` inside each U-Fourier block.
# Selection is via the ``unet_variant`` and ``global_residual`` kwargs of
# UFNOWrapped, themselves driven by env vars UFNO_UNET_VARIANT and
# UFNO_GLOBAL_RESIDUAL in the training script.  All variants:
#   * use BatchNorm3d by default so the existing _replace_bn_with_groupnorm
#     + SyncBatchNorm.convert_sync_batchnorm pipeline transparently handles
#     the v2 norm-swap when ``norm="groupnorm"``
#   * accept channels-first (B, C, X, Y, Z) input (the convention inside
#     SimpleBlock3d's forward, after the channels-last -> channels-first
#     permute by the caller)
#   * return the same shape as input -- they're a residual contribution
#     summed into ``x1 + x2 + x3`` inside each U-Fourier block
# =============================================================================


def _conv_block(in_c: int, out_c: int, kernel_size, stride, dropout_rate: float):
    """3-D conv + BN + LeakyReLU + Dropout block (matches upstream U_net.conv).

    Extracted to a shared helper so the variants don't drift from the paper
    recipe in their own conv stacks.
    """
    if isinstance(kernel_size, int):
        padding = (kernel_size - 1) // 2
    else:
        padding = tuple((k - 1) // 2 for k in kernel_size)
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate),
    )


class AnisotropicZUNet(nn.Module):
    """Option D: Wen et al. U-Net topology with anisotropic Z downsampling.

    Identical encoder/decoder structure to the upstream ``U_net`` except
    the outermost stage uses ``stride=(2, 2, 4)`` instead of ``(2, 2, 2)``.
    This doubles the effective LOS receptive field at the bottleneck
    (~16 cells ≈ 2.3 redshift units) without adding parameters or layers.
    The decoder's outermost ``ConvTranspose3d`` uses matching
    ``stride=(2, 2, 4)`` to invert the asymmetric downsample.

    Encoder spatial shapes (input ``X, Y, Z``, all assumed divisible by
    ``(8, 8, 16)`` after the UFNOWrapped padding):

      conv1:   (X, Y, Z)       -> (X/2, Y/2, Z/4)    stride (2, 2, 4)
      conv2_1∘conv2: (X/4, Y/4, Z/8)                 stride (2, 2, 2)∘(1, 1, 1)
      conv3_1∘conv3: (X/8, Y/8, Z/16)                stride (2, 2, 2)∘(1, 1, 1)

    Decoder mirrors this with ``ConvTranspose3d`` strides
    ``(2, 2, 2)``, ``(2, 2, 2)``, ``(2, 2, 4)``.
    """

    # Padding requirement -- exposed so UFNOWrapped knows how much to pad
    # each axis when this variant is selected.
    REQUIRED_MULT_OF_Z = 16
    REQUIRED_MULT_OF_XY = 8

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 3, dropout_rate: float = 0.0):
        super().__init__()
        self.input_channels = input_channels
        s2 = (2, 2, 2)
        s224 = (2, 2, 4)

        # Encoder
        self.conv1 = _conv_block(input_channels, output_channels,
                                 kernel_size, stride=s224,
                                 dropout_rate=dropout_rate)
        self.conv2 = _conv_block(input_channels, output_channels,
                                 kernel_size, stride=s2,
                                 dropout_rate=dropout_rate)
        self.conv2_1 = _conv_block(input_channels, output_channels,
                                   kernel_size, stride=1,
                                   dropout_rate=dropout_rate)
        self.conv3 = _conv_block(input_channels, output_channels,
                                 kernel_size, stride=s2,
                                 dropout_rate=dropout_rate)
        self.conv3_1 = _conv_block(input_channels, output_channels,
                                   kernel_size, stride=1,
                                   dropout_rate=dropout_rate)

        # Decoder.  ConvTranspose3d with kernel_size=4, stride=2, padding=1
        # exactly doubles the spatial dim along each axis.  For the
        # asymmetric outermost stage we use kernel_size=(4, 4, 8),
        # stride=(2, 2, 4), padding=(1, 1, 2) to invert the (2, 2, 4)
        # encoder stride.
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels,
                               kernel_size=4, stride=s2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(input_channels * 2, output_channels,
                               kernel_size=4, stride=s2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose3d(input_channels * 2, output_channels,
                               kernel_size=(4, 4, 8), stride=s224,
                               padding=(1, 1, 2)),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.output_layer = nn.Conv3d(
            input_channels * 2, output_channels,
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(x)                            # X/2, Y/2, Z/4
        out_conv2 = self.conv2_1(self.conv2(out_conv1))      # X/4, Y/4, Z/8
        out_conv3 = self.conv3_1(self.conv3(out_conv2))      # X/8, Y/8, Z/16
        out_deconv2 = self.deconv2(out_conv3)                # X/4, Y/4, Z/8
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)
        out_deconv1 = self.deconv1(concat2)                  # X/2, Y/2, Z/4
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)
        out_deconv0 = self.deconv0(concat1)                  # X, Y, Z
        concat0 = torch.cat((x, out_deconv0), dim=1)
        return self.output_layer(concat0)


class LOSConv1DPath(nn.Module):
    """Option F: 1-D LOS convolution stack replacing the 3-D U-Net entirely.

    Stack of ``Conv3d`` layers with ``kernel_size=(1, 1, kz)`` -- effectively
    1-D convolutions along the LOS axis only, leaving the transverse plane
    untouched.  Receptive field along Z is ``1 + n_layers * (kz - 1)``; at
    the default ``kz=7, n_layers=4`` that's 25 cells (~1.9 redshift units),
    much wider than the U-Net's ~8 cells of LOS receptive field.

    Inductive bias: the cone's reionization history is dominantly a 1-D
    structure along LOS (redshift evolution); the spectral path already
    handles transverse coupling, so the local-feature path can specialise
    in LOS context.  No spatial downsample -> no padding requirement.
    """

    REQUIRED_MULT_OF_Z = 1
    REQUIRED_MULT_OF_XY = 1

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size_z: int = 7, n_layers: int = 4,
                 dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        pad_z = (kernel_size_z - 1) // 2
        for i in range(n_layers):
            in_c = input_channels if i == 0 else output_channels
            layers.append(
                nn.Conv3d(in_c, output_channels,
                          kernel_size=(1, 1, kernel_size_z),
                          padding=(0, 0, pad_z), bias=False)
            )
            layers.append(nn.BatchNorm3d(output_channels))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalResidualWrapper(nn.Module):
    """Option E: wrap any U-Net path with a global-pooling residual input.

    Computes per-channel global mean over the spatial dims, projects it
    through a small MLP, broadcasts back to all voxels, and adds to the
    wrapped path's input.  Gives the local-feature path cone-level context
    (e.g. "this cone is heavily reionized" / "this cone is fully neutral")
    that its small receptive field can't see from local patches alone.

    Composable with any base U-Net variant (default, anisotropic_z, los1d).
    Strictly residual -- the wrapped path can in principle ignore the
    broadcast term, so this should never hurt baseline performance.
    """

    def __init__(self, base_path: nn.Module, channels: int):
        super().__init__()
        self.base_path = base_path
        self.global_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels * 2, channels),
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X, Y, Z)
        global_feat = x.mean(dim=(-3, -2, -1))            # (B, C)
        global_proj = self.global_mlp(global_feat)        # (B, C)
        # Broadcast to spatial dims via .view(B, C, 1, 1, 1).
        bias = global_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.base_path(x + bias)


# Map from env-var string to U-Net class for UFNOWrapped construction.
_UNET_VARIANT_CLASSES = {
    "default": None,                # keep upstream U_net (no replacement)
    "anisotropic_z": AnisotropicZUNet,
    "los1d": LOSConv1DPath,
}


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
    def _pad_amount(n: int, mult_of: int = 8, min_pad: int = 8) -> int:
        """Smallest p such that ``(n + p)`` is a multiple of ``mult_of``
        and ``p >= min_pad``.

        Per-axis padding lets the U-Net variants tighten the Z multiple
        (e.g. ``mult_of=16`` for the anisotropic-Z variant) without
        affecting X/Y, or relax it entirely (``mult_of=1`` for the
        no-downsample LOS-1D variant) while keeping the same anti-FFT-
        periodicity buffer ``min_pad`` everywhere.
        """
        target = ((n + min_pad + mult_of - 1) // mult_of) * mult_of
        return target - n

    def __init__(self, modes1: int, modes2: int, modes3: int,
                 width: int, in_channels: int, out_channels: int = 1,
                 sigmoid: bool = True,
                 norm: str = "batchnorm",
                 norm_num_groups: int = 8,
                 unet_variant: str = "default",
                 global_residual: bool = False):
        """
        norm : "batchnorm" (paper default) or "groupnorm".
            With "groupnorm" every ``BatchNorm3d`` in the U-Net path is
            replaced by a ``GroupNorm(num_groups, num_channels)`` of matching
            channel count.  GroupNorm is batch-independent, which sidesteps
            the small-batch + DDP foot-gun that motivated wrapping the model
            in ``SyncBatchNorm`` for the v1 U-FNO run.  ``SyncBatchNorm``
            still helps at large batch even with BN, but at ``batch_size=1``
            per rank GroupNorm tends to train cleaner from epoch 0 because
            it doesn't depend on batch statistics at all.

        norm_num_groups : int, default 8
            Target group count when ``norm="groupnorm"``.  If a layer's
            channel count isn't divisible by this, the group count is
            reduced for that layer to the largest divisor <= num_groups.
            ``num_groups=8`` is the GroupNorm-paper recommendation; at
            ``width=32`` it splits channels into 4-channel groups (matches
            standard image-classification GroupNorm configs).

        unet_variant : "default" | "anisotropic_z" | "los1d"
            Picks the local-feature path inside each U-Fourier block.
              * "default" -- Wen et al.'s original 3-D U-Net (v1/v2).
              * "anisotropic_z" -- (option D) U-Net with extra Z
                downsampling.  Doubles the LOS receptive field at the
                bottleneck (~16 cells); pads Z to multiple of 16
                (vs default 8).  Targets the cone-61-style high-z
                artefact root cause directly.
              * "los1d" -- (option F) replaces the 3-D U-Net with a
                stack of 1-D Z-only convolutions.  No spatial downsample
                (no padding constraint).  Much wider LOS receptive field
                (~25 cells at default kz=7, n=4) at much fewer params.

        global_residual : bool, default False
            If True, wrap the chosen U-Net path with a
            ``GlobalResidualWrapper`` -- (option E) adds a per-channel
            global-pooling residual that broadcasts cone-level context
            back into the U-Net's input.  Composable with any variant.
        """
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError(
                "UFNOWrapped only supports out_channels=1 because the "
                "upstream SimpleBlock3d hardcodes fc2 = Linear(128, 1). "
                "Patch fc2 manually if you need a different output.")
        norm = norm.lower()
        if norm not in ("batchnorm", "groupnorm"):
            raise ValueError(
                f"norm must be 'batchnorm' or 'groupnorm', got {norm!r}")
        unet_variant = unet_variant.lower()
        if unet_variant not in _UNET_VARIANT_CLASSES:
            raise ValueError(
                f"unet_variant must be one of {sorted(_UNET_VARIANT_CLASSES)}, "
                f"got {unet_variant!r}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.sigmoid = bool(sigmoid)
        self.norm = norm
        self.unet_variant = unet_variant
        self.global_residual = bool(global_residual)

        # Construct the upstream body and override the hardcoded input
        # lifting layer.  Everything else (spectral convs, U-Net blocks,
        # output MLPs) keeps the paper's defaults.
        self.body = SimpleBlock3d(modes1, modes2, modes3, width)
        self.body.fc0 = nn.Linear(self.in_channels, width)

        # Tier 2 (D, F): replace each U-Fourier block's U-Net with the
        # selected variant.  "default" leaves Wen et al.'s upstream U_net
        # in place (v1/v2 behavior preserved exactly).  Padding-multiple
        # requirements get tightened on Z if the variant has more
        # downsamples there.
        variant_cls = _UNET_VARIANT_CLASSES[unet_variant]
        self._z_mult_of = self.MULT_OF
        self._xy_mult_of = self.MULT_OF
        if variant_cls is not None:
            for attr in ("unet3", "unet4", "unet5"):
                setattr(self.body, attr, variant_cls(width, width))
            if hasattr(variant_cls, "REQUIRED_MULT_OF_Z"):
                self._z_mult_of = int(variant_cls.REQUIRED_MULT_OF_Z)
            if hasattr(variant_cls, "REQUIRED_MULT_OF_XY"):
                self._xy_mult_of = int(variant_cls.REQUIRED_MULT_OF_XY)

        # Tier 2 (E): wrap each U-Net path with a global-pooling residual.
        # Applied after variant selection so it composes with any variant.
        if self.global_residual:
            for attr in ("unet3", "unet4", "unet5"):
                base = getattr(self.body, attr)
                setattr(self.body, attr,
                        GlobalResidualWrapper(base, channels=width))

        # B: post-construction patch of every BatchNorm3d in the U-Net
        # path to a matching GroupNorm.  Applied LAST so it also catches
        # BN layers inside the Tier 2 variant classes (AnisotropicZUNet
        # and LOSConv1DPath also use BN by default).
        if norm == "groupnorm":
            _replace_bn_with_groupnorm(self.body, norm_num_groups)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Convert (B, C, X, Y, Z) -> (B, X, Y, Z, C) for the U-FNO body
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        B, X, Y, Z, _ = x.shape

        # Per-axis padding so each spatial dim ends up a multiple of 8 (U-Net
        # downsample requirement) AND has at least MIN_PAD cells of buffer
        # (anti-periodicity for the FFT, mirrors the upstream).  For our
        # production shape (140, 140, 256) this gives pad (12, 12, 8) ->
        # (152, 152, 264), all divisible by 8.
        pad_x = self._pad_amount(X, mult_of=self._xy_mult_of,
                                 min_pad=self.MIN_PAD)
        pad_y = self._pad_amount(Y, mult_of=self._xy_mult_of,
                                 min_pad=self.MIN_PAD)
        pad_z = self._pad_amount(Z, mult_of=self._z_mult_of,
                                 min_pad=self.MIN_PAD)

        # X/Y are equivalent periodic simulation axes; Z is the finite
        # lightcone/redshift direction. Pad channels-first so PyTorch can
        # apply circular padding to both transverse dimensions identically.
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = pad_ufno_spatial(x, pad_x=pad_x, pad_y=pad_y, pad_z=pad_z)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

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


def pad_ufno_spatial(
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
