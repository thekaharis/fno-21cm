"""Retired 3-D architectures, kept loadable for inference on old checkpoints.

``modeling.build_model`` dispatches here for kinds that are no longer
trained. Nothing in the main tree imports this at module level, so the retired
code costs nothing until an old ``run_metadata.json`` asks for it.
"""

from __future__ import annotations

import torch.nn as nn

KINDS = ("fno", "sirenfno")


def build(config, in_channels: int) -> nn.Module:
    """Build a retired architecture from its recorded ``ModelConfig``."""
    if config.kind == "sirenfno":
        from legacy.arch.siren_fno_3d import SirenFNO3d

        return SirenFNO3d(
            n_modes=config.modes,
            hidden_channels=config.hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            n_layers=config.n_layers,
            padding=config.siren_padding,
            add_grid=True,
            siren_hidden_dim=config.siren_hidden_dim,
            siren_omega=config.siren_omega,
            siren_n_hidden=config.siren_n_hidden,
            siren_feature_dim=config.siren_feature_dim,
            siren_ff_sigma=config.siren_ff_sigma,
            siren_learnable_ff=config.siren_learnable_ff,
            mlp_dropout=config.siren_mlp_dropout,
            output_sigmoid=config.siren_output_sigmoid,
            sigmoid_temperature=config.siren_sigmoid_temperature,
        )
    if config.kind == "fno":
        from util.neuralop_setup import prefer_local_neuralop

        prefer_local_neuralop()
        from neuralop.models import FNO

        return FNO(
            n_modes=config.modes,
            hidden_channels=config.hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            n_layers=config.n_layers,
            projection_channel_ratio=2,
            positional_embedding="grid",
        )
    raise ValueError(f"{config.kind!r} is not a retired architecture")
