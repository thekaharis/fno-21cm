"""Shared model construction and checkpoint handling for training and plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

from neuralop.models import FNO  # noqa: E402


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by 3-D training and visualization."""

    kind: str = "fno"
    modes: tuple[int, int, int] = (16, 16, 16)
    hidden_channels: int = 32
    n_layers: int = 4
    ufno_width: int = 32
    ufno_norm: str = "batchnorm"
    ufno_unet_variant: str = "default"
    ufno_global_residual: bool = False
    siren_hidden_dim: int = 64
    siren_omega: float = 30.0
    siren_n_hidden: int = 1
    siren_feature_dim: int = 16
    siren_ff_sigma: float = 128.0
    siren_learnable_ff: bool = True
    siren_padding: tuple[int, int, int] = (0, 0, 8)
    siren_mlp_dropout: float = 0.0
    # False preserves the behavior of legacy metadata that predates this
    # option. Fresh environment-driven SirenFNO runs default to True below.
    siren_output_sigmoid: bool = False
    siren_sigmoid_temperature: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in {"fno", "ufno", "sirenfno"}:
            raise ValueError(
                "kind must be 'fno', 'ufno', or 'sirenfno', "
                f"got {self.kind!r}"
            )
        if len(self.modes) != 3 or any(int(value) <= 0 for value in self.modes):
            raise ValueError("modes must contain three positive values")
        if self.ufno_norm not in {"batchnorm", "groupnorm"}:
            raise ValueError(
                "ufno_norm must be 'batchnorm' or 'groupnorm', "
                f"got {self.ufno_norm!r}"
            )
        if self.ufno_unet_variant not in {"default", "anisotropic_z", "los1d"}:
            raise ValueError(
                "ufno_unet_variant must be 'default', 'anisotropic_z', or "
                f"'los1d', got {self.ufno_unet_variant!r}"
            )
        if len(self.siren_padding) != 3 or any(
            int(value) < 0 for value in self.siren_padding
        ):
            raise ValueError("siren_padding must contain three non-negative values")
        if self.siren_feature_dim <= 0 or self.siren_feature_dim % 2:
            raise ValueError("siren_feature_dim must be a positive even integer")
        if self.siren_n_hidden < 1:
            raise ValueError("siren_n_hidden must be at least 1")
        if self.siren_sigmoid_temperature <= 0:
            raise ValueError("siren_sigmoid_temperature must be positive")

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Read the experiment switches used by the SLURM scripts."""
        return cls(
            kind=os.environ.get("MODEL_KIND", "fno").lower(),
            modes=(
                int(os.environ.get("N_MODES_X", "16")),
                int(os.environ.get("N_MODES_Y", "16")),
                int(os.environ.get("N_MODES_Z", "16")),
            ),
            ufno_norm=os.environ.get("UFNO_NORM", "batchnorm").lower(),
            ufno_unet_variant=os.environ.get(
                "UFNO_UNET_VARIANT", "default"
            ).lower(),
            ufno_global_residual=_env_bool("UFNO_GLOBAL_RESIDUAL"),
            siren_hidden_dim=int(os.environ.get("SIREN_HIDDEN_DIM", "64")),
            siren_omega=float(os.environ.get("SIREN_OMEGA", "30.0")),
            siren_n_hidden=int(os.environ.get("SIREN_N_HIDDEN", "1")),
            siren_feature_dim=int(os.environ.get("SIREN_FEATURE_DIM", "16")),
            siren_ff_sigma=float(os.environ.get("SIREN_FF_SIGMA", "128.0")),
            siren_learnable_ff=_env_bool("SIREN_LEARNABLE_FF", True),
            siren_padding=(
                int(os.environ.get("SIREN_PADDING_X", "0")),
                int(os.environ.get("SIREN_PADDING_Y", "0")),
                int(os.environ.get("SIREN_PADDING_Z", "8")),
            ),
            siren_mlp_dropout=float(os.environ.get("SIREN_MLP_DROPOUT", "0.0")),
            siren_output_sigmoid=_env_bool("SIREN_OUTPUT_SIGMOID", True),
            siren_sigmoid_temperature=float(
                os.environ.get("SIREN_SIGMOID_TEMPERATURE", "2.0")
            ),
        )

    @classmethod
    def from_dict(cls, values: Mapping) -> "ModelConfig":
        values = dict(values)
        if "modes" in values:
            values["modes"] = tuple(int(value) for value in values["modes"])
        if "siren_padding" in values:
            values["siren_padding"] = tuple(
                int(value) for value in values["siren_padding"]
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "modes": list(self.modes),
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "ufno_width": self.ufno_width,
            "ufno_norm": self.ufno_norm,
            "ufno_unet_variant": self.ufno_unet_variant,
            "ufno_global_residual": self.ufno_global_residual,
            "siren_hidden_dim": self.siren_hidden_dim,
            "siren_omega": self.siren_omega,
            "siren_n_hidden": self.siren_n_hidden,
            "siren_feature_dim": self.siren_feature_dim,
            "siren_ff_sigma": self.siren_ff_sigma,
            "siren_learnable_ff": self.siren_learnable_ff,
            "siren_padding": list(self.siren_padding),
            "siren_mlp_dropout": self.siren_mlp_dropout,
            "siren_output_sigmoid": self.siren_output_sigmoid,
            "siren_sigmoid_temperature": self.siren_sigmoid_temperature,
        }

    @property
    def default_checkpoint_dir(self) -> Path:
        suffix = {
            "fno": "",
            "ufno": "_ufno",
            "sirenfno": "_sirenfno",
        }[self.kind]
        return Path(f"checkpoints_3d{suffix}")

    def describe(self) -> str:
        if self.kind == "fno":
            return (
                f"FNO modes={self.modes} hidden={self.hidden_channels} "
                f"layers={self.n_layers} pos-emb=grid"
            )
        if self.kind == "sirenfno":
            return (
                f"SirenFNO modes={self.modes} hidden={self.hidden_channels} "
                f"layers={self.n_layers} siren-hidden={self.siren_hidden_dim} "
                f"features={self.siren_feature_dim} padding={self.siren_padding} "
                f"sigmoid={self.siren_output_sigmoid} "
                f"temperature={self.siren_sigmoid_temperature:g}"
            )
        residual = "+global_residual" if self.ufno_global_residual else ""
        return (
            f"U-FNO modes={self.modes} width={self.ufno_width} "
            f"norm={self.ufno_norm} "
            f"unet={self.ufno_unet_variant}{residual} sigmoid-output"
        )


class TrainerModel(nn.Module):
    """Adapt an ``x``-only model to Trainer samples containing extra fields.

    The wrapped model intentionally remains under the attribute ``fno`` so
    existing checkpoints retain their ``fno.*`` state-dict keys.
    """

    def __init__(self, fno: nn.Module):
        super().__init__()
        self.fno = fno

    def forward(self, x, **_):
        return self.fno(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["fno"], name)


def build_3d_model(config: ModelConfig, in_channels: int) -> nn.Module:
    """Construct the configured 3-D architecture."""
    if config.kind == "ufno":
        from models_ufno import UFNOWrapped

        return UFNOWrapped(
            modes1=config.modes[0],
            modes2=config.modes[1],
            modes3=config.modes[2],
            width=config.ufno_width,
            in_channels=in_channels,
            out_channels=1,
            sigmoid=True,
            norm=config.ufno_norm,
            unet_variant=config.ufno_unet_variant,
            global_residual=config.ufno_global_residual,
        )
    if config.kind == "sirenfno":
        from siren_fno_3d import SirenFNO3d

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
    return FNO(
        n_modes=config.modes,
        hidden_channels=config.hidden_channels,
        in_channels=in_channels,
        out_channels=1,
        n_layers=config.n_layers,
        projection_channel_ratio=2,
        positional_embedding="grid",
    )


@dataclass(frozen=True)
class CheckpointLoadReport:
    transform: str
    matched: int
    total: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def load_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
) -> CheckpointLoadReport:
    """Load raw, trainer-wrapped, or DDP-wrapped state dicts safely."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a state dict in {checkpoint}, got {type(raw)}")
    raw = {key: value for key, value in raw.items() if key != "_metadata"}

    def strip_module(key: str) -> str:
        return key.removeprefix("module.")

    candidates = (
        ("as-is", raw),
        ("add fno.", {f"fno.{key}": value for key, value in raw.items()}),
        (
            "strip module.",
            {strip_module(key): value for key, value in raw.items()},
        ),
        (
            "strip module. + add fno.",
            {
                f"fno.{strip_module(key)}": value
                for key, value in raw.items()
            },
        ),
    )
    target = model.state_dict()

    def match_count(state_dict) -> int:
        return sum(
            key in target and target[key].shape == value.shape
            for key, value in state_dict.items()
        )

    transform, state_dict = max(candidates, key=lambda item: match_count(item[1]))
    matched = match_count(state_dict)
    if matched == 0:
        raw_key = next(iter(raw), "<empty>")
        target_key = next(iter(target), "<empty>")
        raise RuntimeError(
            "No checkpoint parameters match the configured model. "
            f"Sample checkpoint key: {raw_key!r}; model key: {target_key!r}."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return CheckpointLoadReport(
        transform=transform,
        matched=matched,
        total=len(target),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
    )
