"""Shared model construction and checkpoint handling for training and plots."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from operators import resolve_operator_name, resolve_slot_operators, validate_operator


#: Model kinds built from the pluggable local/global U-Net skeleton, and the
#: (local, global) operator pair each one is shorthand for. ``localop`` takes
#: both operators from the configuration instead.
LOCAL_GLOBAL_KINDS = {
    "localfno": ("fourier", "fourier"),
    "localwno": ("wavelet", "fourier"),
    "localwhno": ("hadamard", "fourier"),
    "localsirenfno": ("siren_fourier", "siren_fourier"),
}

#: Short tags used to name checkpoint directories of explicit ``localop`` runs.
OPERATOR_TAGS = {
    "fourier": "fno",
    "siren_fourier": "sirenfno",
    "wavelet": "wno",
    "hadamard": "whno",
    "cnn": "cnn",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_optional_bool(name: str) -> bool | None:
    value = os.environ.get(name, "auto").strip().lower()
    if value in {"", "auto", "default", "none"}:
        return None
    return _env_bool(name)


def operator_env_settings() -> dict:
    """Operator-slot switches, shared by the 2-D and 3-D entry points."""
    return {
        "local_operator": os.environ.get("LOCAL_OPERATOR", "fourier"),
        "global_operator": os.environ.get("GLOBAL_OPERATOR", "fourier"),
        "local_windowed": _env_optional_bool("LOCAL_WINDOWED"),
        "whno_ordering": os.environ.get("WHNO_ORDERING", "sequency").lower(),
        "cnn_depth": int(os.environ.get("CNN_DEPTH", "3")),
        "cnn_kernel_size": int(os.environ.get("CNN_KERNEL_SIZE", "3")),
        "cnn_dropout": float(os.environ.get("CNN_DROPOUT", "0.0")),
        "cnn_norm": os.environ.get("CNN_NORM", "groupnorm").lower(),
        "localwno_levels": int(os.environ.get("LOCALWNO_LEVELS", "2")),
        "siren_hidden_dim": int(os.environ.get("SIREN_HIDDEN_DIM", "64")),
        "siren_omega": float(os.environ.get("SIREN_OMEGA", "30.0")),
        "siren_n_hidden": int(os.environ.get("SIREN_N_HIDDEN", "1")),
        "siren_feature_dim": int(os.environ.get("SIREN_FEATURE_DIM", "16")),
        "siren_ff_sigma": float(os.environ.get("SIREN_FF_SIGMA", "128.0")),
        "siren_learnable_ff": _env_bool("SIREN_LEARNABLE_FF", True),
    }


def slot_hyperparameters(operator: str, settings: Mapping) -> dict:
    """Pick the hyperparameters one operator reads out of a settings mapping."""
    if operator == "wavelet":
        return {"levels": int(settings["localwno_levels"])}
    if operator == "hadamard":
        return {"ordering": str(settings["whno_ordering"])}
    if operator == "cnn":
        return {
            "depth": int(settings["cnn_depth"]),
            "kernel_size": int(settings["cnn_kernel_size"]),
            "dropout": float(settings["cnn_dropout"]),
            "norm": str(settings["cnn_norm"]),
        }
    if operator in {"siren_fourier", "siren_hadamard"}:
        siren = {
            "hidden_dim": int(settings["siren_hidden_dim"]),
            "omega": float(settings["siren_omega"]),
            "n_hidden": int(settings["siren_n_hidden"]),
            "feature_dim": int(settings["siren_feature_dim"]),
            "ff_sigma": float(settings["siren_ff_sigma"]),
            "learnable_ff": bool(settings["siren_learnable_ff"]),
        }
        if operator == "siren_hadamard":
            # The SIREN generates the mixing weights; the Walsh basis it mixes
            # still needs its ordering, so this slot reads both groups.
            siren["ordering"] = str(settings["whno_ordering"])
        return siren
    return {}


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by every training entry point and its plots.

    ``ndim`` selects the 2-D or 3-D twin of an architecture; the axis tuples
    (``modes``, ``localfno_window``, ``localfno_modes``, ``siren_padding``) must
    match it. All three tasks -- 3-D x_HI cubes, 2-D x_HI slices, 2-D z_re maps
    -- read the same environment variables and record the same metadata, so a
    run is described the same way whatever its dimensionality.
    """

    kind: str = "fno"
    ndim: int = 3
    modes: tuple[int, ...] | None = None
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
    siren_padding: tuple[int, ...] | None = None
    siren_mlp_dropout: float = 0.0
    # False preserves the behavior of legacy metadata that predates this
    # option. Fresh environment-driven SirenFNO runs default to True below.
    siren_output_sigmoid: bool = False
    siren_sigmoid_temperature: float = 2.0
    localfno_window: tuple[int, ...] | None = None
    localfno_modes: tuple[int, ...] | None = None
    localfno_base_width: int = 16
    localfno_spectral_rank: int = 16
    localfno_patch_chunk_size: int = 128
    localwno_levels: int = 2
    # Operator slots of the local/global U-Net. Legacy kinds fill these in
    # automatically; kind="localop" reads them as given.
    local_operator: str = "fourier"
    global_operator: str = "fourier"
    local_windowed: bool | None = None
    whno_ordering: str = "sequency"
    cnn_depth: int = 3
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.0
    cnn_norm: str = "groupnorm"

    def __post_init__(self) -> None:
        if self.kind not in {
            "fno", "ufno", "sirenfno", "localfno", "localsirenfno",
            "localwno", "localwhno", "localop",
        }:
            raise ValueError(
                "kind must be 'fno', 'ufno', 'sirenfno', 'localfno', "
                "'localsirenfno', 'localwno', 'localwhno', or 'localop', "
                f"got {self.kind!r}"
            )
        object.__setattr__(
            self, "local_operator", resolve_operator_name(self.local_operator)
        )
        object.__setattr__(
            self, "global_operator", resolve_operator_name(self.global_operator)
        )
        if self.kind in LOCAL_GLOBAL_KINDS:
            slots = LOCAL_GLOBAL_KINDS[self.kind]
            explicit = (self.local_operator, self.global_operator)
            if explicit not in (slots, ("fourier", "fourier")):
                raise ValueError(
                    f"kind={self.kind!r} implies operators {slots}, but "
                    f"{explicit} were configured; use kind='localop' to pair "
                    "operators freely"
                )
            object.__setattr__(self, "local_operator", slots[0])
            object.__setattr__(self, "global_operator", slots[1])
        if self.ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {self.ndim!r}")
        # 3-D defaults truncated to ndim: the dropped entry is always the LOS
        # axis, which a 2-D sky-plane map does not have.
        for name, default in (("modes", (16, 16, 16)),
                              ("siren_padding", (0, 0, 8)),
                              ("localfno_window", (16, 16, 32)),
                              ("localfno_modes", (6, 6, 12))):
            value = getattr(self, name)
            object.__setattr__(
                self, name,
                tuple(default[:self.ndim]) if value is None
                else tuple(int(v) for v in value),
            )
        for name in ("modes", "localfno_window", "localfno_modes"):
            value = getattr(self, name)
            if len(value) != self.ndim or any(int(v) <= 0 for v in value):
                raise ValueError(
                    f"{name} must contain {self.ndim} positive values for "
                    f"ndim={self.ndim}, got {tuple(value)}"
                )
        if any(int(value) % 4 for value in self.localfno_window):
            raise ValueError("localfno_window values must be divisible by four")
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
        if len(self.siren_padding) != self.ndim or any(
            int(value) < 0 for value in self.siren_padding
        ):
            raise ValueError(
                f"siren_padding must contain {self.ndim} non-negative values"
            )
        if self.siren_feature_dim <= 0 or self.siren_feature_dim % 2:
            raise ValueError("siren_feature_dim must be a positive even integer")
        if self.siren_n_hidden < 1:
            raise ValueError("siren_n_hidden must be at least 1")
        if self.siren_sigmoid_temperature <= 0:
            raise ValueError("siren_sigmoid_temperature must be positive")
        if self.localfno_base_width <= 0:
            raise ValueError("localfno_base_width must be positive")
        if self.localfno_spectral_rank <= 0:
            raise ValueError("localfno_spectral_rank must be positive")
        if self.localfno_spectral_rank > self.localfno_base_width:
            raise ValueError(
                "localfno_spectral_rank cannot exceed localfno_base_width"
            )
        if self.localfno_patch_chunk_size <= 0:
            raise ValueError("localfno_patch_chunk_size must be positive")
        if self.localwno_levels <= 0:
            raise ValueError("localwno_levels must be positive")
        if self.cnn_depth <= 0:
            raise ValueError("cnn_depth must be positive")
        if self.cnn_kernel_size <= 0 or not self.cnn_kernel_size % 2:
            raise ValueError("cnn_kernel_size must be a positive odd integer")
        if not 0.0 <= self.cnn_dropout < 1.0:
            raise ValueError("cnn_dropout must be in [0, 1)")
        if self.cnn_norm not in {"groupnorm", "batchnorm"}:
            raise ValueError(
                f"cnn_norm must be 'groupnorm' or 'batchnorm', "
                f"got {self.cnn_norm!r}"
            )
        if self.whno_ordering not in {"sequency", "natural"}:
            raise ValueError(
                f"whno_ordering must be 'sequency' or 'natural', "
                f"got {self.whno_ordering!r}"
            )
        if self.is_local_global:
            # Reject an unbuildable local slot here, before any data is read.
            # The global slot's shape is data-dependent, so the model pads to
            # the operator's requirements at run time instead.
            local, _ = self.operator_slots()
            if self.local_slot_is_windowed:
                validate_operator(
                    local[0],
                    self.localfno_window,
                    self.localfno_modes,
                    local[1],
                    context="localfno_window",
                )

    @property
    def is_local_global(self) -> bool:
        """Is this kind built from the pluggable local/global U-Net?"""
        return self.kind in LOCAL_GLOBAL_KINDS or self.kind == "localop"

    @property
    def local_slot_is_windowed(self) -> bool:
        from operators import operator_spec

        if self.local_windowed is not None:
            return bool(self.local_windowed)
        return operator_spec(self.local_operator).windowed

    def _slot_kwargs(self, operator: str) -> dict:
        """Hyperparameters this configuration supplies to one operator."""
        return slot_hyperparameters(
            operator,
            {
                "localwno_levels": self.localwno_levels,
                "whno_ordering": self.whno_ordering,
                "cnn_depth": self.cnn_depth,
                "cnn_kernel_size": self.cnn_kernel_size,
                "cnn_dropout": self.cnn_dropout,
                "cnn_norm": self.cnn_norm,
                "siren_hidden_dim": self.siren_hidden_dim,
                "siren_omega": self.siren_omega,
                "siren_n_hidden": self.siren_n_hidden,
                "siren_feature_dim": self.siren_feature_dim,
                "siren_ff_sigma": self.siren_ff_sigma,
                "siren_learnable_ff": self.siren_learnable_ff,
            },
        )

    def operator_slots(
        self,
    ) -> tuple[tuple[str, dict], tuple[str, dict]]:
        """Resolve both slots to ``(operator name, hyperparameters)``."""
        return resolve_slot_operators(
            self.local_operator,
            self.global_operator,
            self._slot_kwargs(self.local_operator),
            self._slot_kwargs(self.global_operator),
        )

    @classmethod
    def from_env(cls, ndim: int = 3) -> "ModelConfig":
        """Read the experiment switches used by the SLURM scripts.

        Axis variables are read per axis, so a 2-D run uses ``*_X``/``*_Y`` and
        ignores ``*_Z``. Every other switch is shared verbatim with 3-D.
        """
        def axes(*prefixes: str, defaults: tuple[int, ...]) -> tuple[int, ...]:
            """First prefix that is set wins, per axis.

            ``LOCALFNO_GLOBAL_MODES_*`` is the 2-D pipeline's name for the
            bottleneck modes that 3-D calls ``N_MODES_*``; both spell the same
            field, so both sets of sbatch keep working.
            """
            names = ("X", "Y", "Z")[:ndim]
            out = []
            for axis, default in zip(names, defaults):
                for prefix in prefixes:
                    value = os.environ.get(f"{prefix}_{axis}")
                    if value is not None:
                        break
                out.append(int(value if value is not None else default))
            return tuple(out)

        return cls(
            ndim=ndim,
            # "fno" and "sirenfno" remain valid (legacy.arch builds them for
            # old checkpoints) but are no longer the default for a new run.
            kind=os.environ.get("MODEL_KIND", "localfno").lower(),
            modes=axes("LOCALFNO_GLOBAL_MODES", "N_MODES",
                       defaults=(16, 16, 16)),
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
            siren_padding=axes("SIREN_PADDING", defaults=(0, 0, 8)),
            siren_mlp_dropout=float(os.environ.get("SIREN_MLP_DROPOUT", "0.0")),
            siren_output_sigmoid=_env_bool("SIREN_OUTPUT_SIGMOID", True),
            siren_sigmoid_temperature=float(
                os.environ.get("SIREN_SIGMOID_TEMPERATURE", "2.0")
            ),
            localfno_window=axes("LOCALFNO_WINDOW", defaults=(16, 16, 32)),
            localfno_modes=axes("LOCALFNO_MODES", defaults=(6, 6, 12)),
            localfno_base_width=int(
                os.environ.get("LOCALFNO_BASE_WIDTH", "16")
            ),
            localfno_spectral_rank=int(
                os.environ.get("LOCALFNO_SPECTRAL_RANK", "16")
            ),
            localfno_patch_chunk_size=int(
                os.environ.get("LOCALFNO_PATCH_CHUNK_SIZE", "128")
            ),
            **{
                key: value
                for key, value in operator_env_settings().items()
                if not key.startswith("siren_")
            },
        )

    @classmethod
    def from_dict(cls, values: Mapping) -> "ModelConfig":
        """Rebuild from recorded metadata, including pre-unification shapes.

        The 2-D entry points used to write their own dict shape: global modes
        under ``localfno_global_modes`` rather than ``modes``, no ``ndim``, and
        bookkeeping keys (``in_channels``, ``localwno_wavelet``) that are not
        configuration. Normalising here is what lets every old checkpoint --
        2-D, 3-D or z_re -- rebuild through the one factory.
        """
        values = dict(values)
        # 2-D runs recorded the bottleneck modes under their own name, and
        # `n_modes` meant whatever the architecture's main mode count was.
        for alias in ("localfno_global_modes", "n_modes"):
            if alias in values:
                values.setdefault("modes", values.pop(alias))
            values.pop(alias, None)
        for key in ("modes", "siren_padding", "localfno_window", "localfno_modes"):
            if key in values:
                values[key] = tuple(int(value) for value in values[key])
        # Dimensionality was implicit in the tuple length before `ndim` existed.
        if "ndim" not in values and "modes" in values:
            values["ndim"] = len(values["modes"])
        # Recorded for the reader, not accepted by the constructor.
        for noise in ("in_channels", "out_channels", "localwno_wavelet",
                      "local_operator_kwargs", "global_operator_kwargs"):
            values.pop(noise, None)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model_config keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ndim": self.ndim,
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
            "localfno_window": list(self.localfno_window),
            "localfno_modes": list(self.localfno_modes),
            "localfno_base_width": self.localfno_base_width,
            "localfno_spectral_rank": self.localfno_spectral_rank,
            "localfno_patch_chunk_size": self.localfno_patch_chunk_size,
            "localwno_levels": self.localwno_levels,
            "local_operator": self.local_operator,
            "global_operator": self.global_operator,
            "local_windowed": self.local_windowed,
            "whno_ordering": self.whno_ordering,
            "cnn_depth": self.cnn_depth,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_dropout": self.cnn_dropout,
            "cnn_norm": self.cnn_norm,
        }

    @property
    def checkpoint_tag(self) -> str:
        """Short architecture name used in checkpoint directory names."""
        if self.kind == "localop":
            return (
                f"local_{OPERATOR_TAGS[self.local_operator]}"
                f"_{OPERATOR_TAGS[self.global_operator]}"
            )
        return self.kind

    @property
    def model_name(self) -> str:
        """Human-readable architecture name for the run banner."""
        if not self.is_local_global:
            return self.kind.upper() if self.kind == "fno" else self.kind
        return {
            "localfno": "LocalFNO",
            "localsirenfno": "LocalSirenFNO",
            "localwno": "LocalWNO",
            "localwhno": "LocalWHNO",
        }.get(
            self.kind,
            f"Local[{OPERATOR_TAGS[self.local_operator]}"
            f"/{OPERATOR_TAGS[self.global_operator]}]",
        )

    @property
    def uses_local_modes(self) -> bool:
        from operators import operator_spec

        return operator_spec(self.local_operator).uses_modes

    @property
    def default_checkpoint_dir(self) -> Path:
        suffix = "" if self.kind == "fno" else f"_{self.checkpoint_tag}"
        return Path("checkpoints") / f"checkpoints_3d{suffix}"

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
        if self.is_local_global:
            from operators import operator_spec

            name = {
                "localfno": "LocalFNO",
                "localsirenfno": "LocalSirenFNO",
                "localwno": "LocalWNO",
                "localwhno": "LocalWHNO",
            }.get(
                self.kind,
                f"Local[{OPERATOR_TAGS[self.local_operator]}/"
                f"{OPERATOR_TAGS[self.global_operator]}]",
            )
            slots = {self.local_operator, self.global_operator}
            siren = (
                (
                    f" siren={self.siren_hidden_dim}x{self.siren_n_hidden} "
                    f"ff={self.siren_feature_dim}@{self.siren_ff_sigma:g}"
                )
                if "siren_fourier" in slots
                else ""
            )
            wavelet = (
                f" wavelet=haar levels={self.localwno_levels}"
                if "wavelet" in slots
                else ""
            )
            walsh = (
                f" walsh=hadamard order={self.whno_ordering}"
                if "hadamard" in slots
                else ""
            )
            cnn = (
                f" cnn=depth{self.cnn_depth}k{self.cnn_kernel_size}"
                f"/{self.cnn_norm}"
                if "cnn" in slots
                else ""
            )
            # Only operators that truncate modes report them.
            local_modes = (
                f"local-modes={self.localfno_modes} "
                if operator_spec(self.local_operator).uses_modes
                else ""
            )
            windowed = "" if self.local_slot_is_windowed else " unwindowed-local"
            return (
                f"{name} local={self.local_operator} "
                f"global={self.global_operator} "
                f"window={self.localfno_window} "
                f"{local_modes}"
                f"global-modes={self.modes} widths="
                f"{self.localfno_base_width}/"
                f"{2 * self.localfno_base_width}/"
                f"{4 * self.localfno_base_width} "
                f"rank={self.localfno_spectral_rank} "
                f"chunk={self.localfno_patch_chunk_size}"
                f"{siren}{wavelet}{walsh}{cnn}{windowed} "
                "sigmoid-output"
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

    def save_checkpoint(self, save_folder, save_name: str) -> None:
        """Save with ``fno.``-prefixed keys, matching DDP-branch checkpoints.

        ``neuralop.training.save_training_state`` calls this for non-DDP
        models. Implementing it here (rather than per architecture) keeps
        single-GPU checkpoints byte-compatible with the multi-GPU runs, which
        save ``model.module.state_dict()`` of this same wrapper.
        """
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.state_dict(),
            (save_folder / f"{save_name}_state_dict.pt").as_posix(),
        )

    def load_checkpoint(self, save_folder, save_name: str, map_location=None) -> None:
        path = Path(save_folder) / f"{save_name}_state_dict.pt"
        self.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=False)
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["fno"], name)


def build_model(config: ModelConfig, in_channels: int) -> nn.Module:
    """Construct the configured architecture at ``config.ndim`` dimensions.

    One factory for all three tasks. Each architecture family has a 2-D and a
    3-D twin that take the same arguments, so the only thing that varies is
    which class is imported.

    Kinds no longer trained ("fno", "sirenfno" in 3-D) are still accepted and
    built from ``legacy.arch``, so a checkpoint written before the cleanup
    rebuilds from its own ``run_metadata.json`` with no edits.
    """
    from legacy.arch import KINDS as LEGACY_KINDS

    two_d = config.ndim == 2
    if config.kind in LEGACY_KINDS and not two_d:
        from legacy import arch

        return arch.build(config, in_channels)
    if config.kind == "ufno":
        if two_d:
            from models_zre_2d import UFNO2d

            return UFNO2d(
                modes1=config.modes[0], modes2=config.modes[1],
                width=config.ufno_width, in_channels=in_channels,
                out_channels=1, sigmoid=True, norm=config.ufno_norm,
            )
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
    if config.kind == "sirenfno" and two_d:
        from models_zre_2d import SirenFNO2d

        return SirenFNO2d(
            n_modes=config.modes,
            hidden_channels=config.hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            n_layers=config.n_layers,
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
    if config.is_local_global:
        (local_name, local_kwargs), (global_name, global_kwargs) = (
            config.operator_slots()
        )
        if two_d:
            from models_zre_2d import LocalFNO2d as LocalFNO
        else:
            from local_fno_3d import LocalFNO3d as LocalFNO
        return LocalFNO(
            in_channels=in_channels,
            out_channels=1,
            base_width=config.localfno_base_width,
            local_window=config.localfno_window,
            local_modes=config.localfno_modes,
            global_modes=config.modes,
            spectral_rank=config.localfno_spectral_rank,
            patch_chunk_size=config.localfno_patch_chunk_size,
            output_sigmoid=True,
            local_operator=local_name,
            global_operator=global_name,
            local_operator_kwargs=local_kwargs,
            global_operator_kwargs=global_kwargs,
            local_windowed=config.local_windowed,
            wavelet_levels=config.localwno_levels,
        )
    if config.kind == "fno" and two_d:
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
    raise ValueError(f"no {config.ndim}-D builder for kind {config.kind!r}")


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
