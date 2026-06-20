"""Shared model construction and checkpoint handling for training and plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from neuralop_setup import prefer_local_neuralop

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

    def __post_init__(self) -> None:
        if self.kind not in {"fno", "ufno"}:
            raise ValueError(f"kind must be 'fno' or 'ufno', got {self.kind!r}")
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

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Read the experiment switches used by the SLURM scripts."""
        return cls(
            kind=os.environ.get("MODEL_KIND", "fno").lower(),
            modes=(16, 16, int(os.environ.get("N_MODES_Z", "16"))),
            ufno_norm=os.environ.get("UFNO_NORM", "batchnorm").lower(),
            ufno_unet_variant=os.environ.get(
                "UFNO_UNET_VARIANT", "default"
            ).lower(),
            ufno_global_residual=_env_bool("UFNO_GLOBAL_RESIDUAL"),
        )

    @classmethod
    def from_dict(cls, values: Mapping) -> "ModelConfig":
        values = dict(values)
        if "modes" in values:
            values["modes"] = tuple(int(value) for value in values["modes"])
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
        }

    @property
    def default_checkpoint_dir(self) -> Path:
        suffix = "" if self.kind == "fno" else "_ufno"
        return Path(f"checkpoints_3d{suffix}")

    def describe(self) -> str:
        if self.kind == "fno":
            return (
                f"FNO modes={self.modes} hidden={self.hidden_channels} "
                f"layers={self.n_layers} pos-emb=grid"
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
