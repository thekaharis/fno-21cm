#!/usr/bin/env python3
"""Train a 2-D operator on density -> x_HI lightcone slices.

The cache contains a small number of slices from many independent cones.
Splits are made by cone before parameter normalization, preventing correlated
slices or validation parameters from leaking into training.

Environment overrides (defaults in parentheses):
  CACHE_FILE (trainset.h5), INPUT_FEATURES (density_z_params)
  MODEL_KIND fno | ufno | localfno | localwno | localwhno | localop (localwno)
  N_EPOCHS (50), BATCH_SIZE (16), LEARNING_RATE (1e-4)
  LOSS_L2_WEIGHT (1.0), LOSS_H1_WEIGHT (0.0), LOSS_BCE_WEIGHT (0.0)
  LOSS_SWD_WEIGHT (0.0), LOSS_HIGHK_WEIGHT (0.0),
  LOSS_EDGE_WARMUP_EPOCHS (5), SWD_DIRECTIONS (48), HIGHK_MIN (0.2)
  CHECKPOINT_DIR (checkpoints/checkpoints_2d_xhi_<kind>)
  LOCALFNO_BASE_WIDTH (32), LOCALFNO_WINDOW_X/Y (16),
  LOCALFNO_GLOBAL_MODES_X/Y (16), LOCALFNO_SPECTRAL_RANK (16),
  LOCALWNO_LEVELS (2)

With MODEL_KIND=localop the two operator slots of the local-global U-Net are
chosen freely with LOCAL_OPERATOR / GLOBAL_OPERATOR (fourier | siren_fourier |
wavelet | hadamard | cnn); the other kinds are shorthand for a fixed pair. See
``operators.py`` for each operator's hyperparameters: LOCALWNO_LEVELS,
WHNO_ORDERING, CNN_DEPTH / CNN_KERNEL_SIZE / CNN_DROPOUT / CNN_NORM.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralop import H1Loss, LpLoss, Trainer
from neuralop.models import FNO
from neuralop.utils import count_model_params

import neuralop as _neuralop

from dataset.dataset import SliceCache, split_by_cone
from dataset import paths
from losses import (
    AbsoluteLoss,
    BinaryCrossEntropyTerm,
    H1Seminorm,
    ExponentialWallDistance,
    HighKPowerRatio,
    ScheduledWeightedLoss,
    SlicedWassersteinEdges,
    WallPlacementLoss,
)
from contrast import ContrastComposed
from modeling import LOCAL_GLOBAL_KINDS, OperatorSlots, TrainerModel
from util.run_metadata import write_run_metadata

print(f"[fno_21cm] using neuralop from {_neuralop.__file__}")


CACHE_FILE = Path(os.environ.get("CACHE_FILE", paths.TRAINSET))
INPUT_FEATURES = os.environ.get("INPUT_FEATURES", "density_z_params").lower()
MODEL_KIND = os.environ.get("MODEL_KIND", "localwno").lower()

N_MODES = (
    int(os.environ.get("N_MODES_X", "32")),
    int(os.environ.get("N_MODES_Y", "32")),
)
HIDDEN_CHANNELS = int(os.environ.get("HIDDEN_CHANNELS", "64"))
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))
UFNO_WIDTH = int(os.environ.get("UFNO_WIDTH", "32"))
UFNO_NORM = os.environ.get("UFNO_NORM", "batchnorm").lower()
LOCALFNO_BASE_WIDTH = int(os.environ.get("LOCALFNO_BASE_WIDTH", "32"))
LOCALFNO_WINDOW = (
    int(os.environ.get("LOCALFNO_WINDOW_X", "16")),
    int(os.environ.get("LOCALFNO_WINDOW_Y", "16")),
)
LOCALFNO_MODES = (
    int(os.environ.get("LOCALFNO_MODES_X", "6")),
    int(os.environ.get("LOCALFNO_MODES_Y", "6")),
)
LOCALFNO_GLOBAL_MODES = (
    int(os.environ.get("LOCALFNO_GLOBAL_MODES_X", "16")),
    int(os.environ.get("LOCALFNO_GLOBAL_MODES_Y", "16")),
)
LOCALFNO_SPECTRAL_RANK = int(
    os.environ.get("LOCALFNO_SPECTRAL_RANK", "16")
)
LOCALFNO_PATCH_CHUNK_SIZE = int(
    os.environ.get("LOCALFNO_PATCH_CHUNK_SIZE", "32")
)
LOCALWNO_LEVELS = int(os.environ.get("LOCALWNO_LEVELS", "2"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "50"))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "5"))
LOSS_L2_WEIGHT = float(os.environ.get("LOSS_L2_WEIGHT", "1.0"))
LOSS_H1_WEIGHT = float(os.environ.get("LOSS_H1_WEIGHT", "0.0"))
# BCE is a confidence regulariser on the [0, 1] x_HI target (see
# losses.BinaryCrossEntropyTerm / the 3-D pipeline's LOSS_BCE_WEIGHT);
# 0 by default so existing L2/H1 runs are unaffected.
LOSS_BCE_WEIGHT = float(os.environ.get("LOSS_BCE_WEIGHT", "0.0"))
# Edge-sharpness pair: SWD scores *where* the transitions are (optimal
# transport on |grad|, so it does not reward the positional hedging that
# makes L2 and H1 blur), HIGHK scores *how much* small-scale power the
# prediction carries. Both 0 by default. They are ramped in over
# LOSS_EDGE_WARMUP_EPOCHS -- their gradients are weak and noisy before L2
# has established rough structure.
LOSS_SWD_WEIGHT = float(os.environ.get("LOSS_SWD_WEIGHT", "0.0"))
LOSS_HIGHK_WEIGHT = float(os.environ.get("LOSS_HIGHK_WEIGHT", "0.0"))
LOSS_EDGE_WARMUP_EPOCHS = int(os.environ.get("LOSS_EDGE_WARMUP_EPOCHS", "5"))
SWD_DIRECTIONS = int(os.environ.get("SWD_DIRECTIONS", "48"))
HIGHK_MIN = float(os.environ.get("HIGHK_MIN", "0.2"))
# Wall-placement loss (losses.WallPlacementLoss): penalises every pixel by its
# distance from the true wall, so misplacement costs grow with distance instead
# of saturating. WALL_CAP bounds the per-pixel weight and hence the gradient.
LOSS_WALL_WEIGHT = float(os.environ.get("LOSS_WALL_WEIGHT", "0.0"))
WALL_CAP = int(os.environ.get("WALL_CAP", "32"))
# Gradient-only Sobolev term. neuralop's H1Loss is NOT L2-free; this one is,
# and is exactly blind to misplacement (measured 1.00x from 4 px to 48 px).
LOSS_H1SEMI_WEIGHT = float(os.environ.get("LOSS_H1SEMI_WEIGHT", "0.0"))
# Exponential-in-distance absolute error (losses.ExponentialWallDistance).
# Median-seeking, so it does not hedge the way L2/BCE do; EXPWALL_SCALE sets
# how fast the penalty grows with distance from the true wall.
LOSS_EXPWALL_WEIGHT = float(os.environ.get("LOSS_EXPWALL_WEIGHT", "0.0"))
EXPWALL_SCALE = float(os.environ.get("EXPWALL_SCALE", "8.0"))
H1SEMI_CAP = os.environ.get("H1SEMI_CAP", "")
# Output contrast map (contrast.py): off | global | head | xhi.
# Initialised at the identity, so every mode starts from the baseline.
CONTRAST_MODE = os.environ.get("CONTRAST_MODE", "off").lower()
# xhi mode: JSON of theta-schedule floats from tests/fit_theta_schedule.py, or
# "theta=<v>" for a constant map. CONTRAST_FREEZE stops the map being learned --
# a learnable map is simply neutralised (a learned global theta went to 4.43,
# i.e. the identity), so freezing is the only way to make the network face it.
CONTRAST_SCHEDULE = os.environ.get("CONTRAST_SCHEDULE", "")
CONTRAST_FREEZE = os.environ.get("CONTRAST_FREEZE", "0") not in ("0", "", "false")
# Alternating refit (util/contrast_refit.py): epoch 0 trains with no map, then
# each epoch refits theta(mean_pred) from the previous epoch's predictions and
# trains through the frozen result.
CONTRAST_REFIT = os.environ.get("CONTRAST_REFIT", "0") not in ("0", "", "false")
CONTRAST_REFIT_SAMPLES = int(os.environ.get("CONTRAST_REFIT_SAMPLES", "2048"))
CONTRAST_REFIT_STEPS = int(os.environ.get("CONTRAST_REFIT_STEPS", "400"))
# sigmoid = 4-parameter curve; stepped = one learned theta per x_HI bin. The
# curve can only express a single monotone step, so it cannot represent the
# band the data wants (identity below 0.005, sharpen 0.005-0.05, identity above
# 0.1); the table can, and can be non-monotonic.
CONTRAST_SCHEDULE_KIND = os.environ.get("CONTRAST_SCHEDULE_KIND", "sigmoid").lower()
CONTRAST_BINS = int(os.environ.get("CONTRAST_BINS", "14"))
# Slice cache the refit draws from, if not the training loader. The training
# sampler down-weights the x_HI tails ~50x, so a refit over it leaves the low
# bins empty -- the first stepped run populated 4 of 14, none below x_HI = 0.1,
# which is exactly the band where sharpening was measured to help. Must be
# built from TRAIN cones: the schedule is part of the model, so fitting it on
# val/test slices would leak.
CONTRAST_REFIT_CACHE = os.environ.get("CONTRAST_REFIT_CACHE", "")
# Floor on an *installed* theta. The refit optimises theta for a frozen
# prediction and is blind to what it does to the next epoch's gradients: the
# map amplifies them by ~1/(2*theta), so a fitted theta of 0.031 amplified by
# 16x and took the model to NaN inside one epoch. 0.25 caps that at ~2x.
CONTRAST_THETA_FLOOR = float(os.environ.get("CONTRAST_THETA_FLOOR", "0.25"))
# The 2-D trainer had no clipping at all, unlike fno_zre.py.
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "0.0"))

SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
RUN_SEED = int(os.environ.get("RUN_SEED", "0"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.1"))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.1"))

_LOCAL_KINDS = tuple(LOCAL_GLOBAL_KINDS) + ("localop",)
_KIND_SUFFIX = {
    "fno": "fno",
    "ufno": "ufno",
    **{kind: kind for kind in _LOCAL_KINDS},
}
OPERATOR_SLOTS = (
    OperatorSlots.from_env(MODEL_KIND) if MODEL_KIND in _LOCAL_KINDS else None
)
_CHECKPOINT_TAG = (
    OPERATOR_SLOTS.checkpoint_tag if OPERATOR_SLOTS is not None
    else _KIND_SUFFIX.get(MODEL_KIND, MODEL_KIND)
)
CHECKPOINT_DIR = Path(os.environ.get(
    "CHECKPOINT_DIR",
    f"checkpoints/checkpoints_2d_xhi_{_CHECKPOINT_TAG}",
))
RESUME_DIR = os.environ.get("RESUME_DIR") or None
DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu",
)


class SliceLoggingTrainer(Trainer):
    """Single-process trainer with dashboard-compatible JSONL metrics."""

    def __init__(self, *args, metrics_path=None, append=False,
                 contrast_refit=False, refit_loader=None,
                 refit_samples=2048, refit_steps=400,
                 refit_objective=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_refit = contrast_refit
        self.refit_loader = refit_loader
        self.refit_samples = refit_samples
        self.refit_steps = refit_steps
        # Fit the map under the loss actually being trained on.
        self.refit_objective = refit_objective
        self._schedule_stats: dict | None = None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.metrics_path.unlink(missing_ok=True)
        self._last_train: dict | None = None

    def train_one_epoch(self, epoch, train_loader, training_loss):
        # Drives ScheduledWeightedLoss' warmup ramp (matches the 3-D trainer).
        if hasattr(training_loss, "set_epoch"):
            training_loss.set_epoch(int(epoch))
        if self.contrast_refit:
            from util import contrast_refit as _refit
            if int(epoch) == 0:
                # No prediction exists yet to fit a schedule against.
                _refit.disable(self.model)
                self._schedule_stats = None
                print("[contrast] epoch 0: map disabled", flush=True)
            else:
                st = _refit.refit_and_install(
                    self.model, self.refit_loader or train_loader,
                    DEVICE, self.refit_samples, self.refit_steps,
                    objective=self.refit_objective,
                    theta_floor=CONTRAST_THETA_FLOOR)
                self._schedule_stats = st
                print(f"[contrast] epoch {int(epoch)}: "
                      f"{_refit.summary_line(st)}", flush=True)
        out = super().train_one_epoch(epoch, train_loader, training_loss)
        train_err, avg_loss, _avg_lasso, elapsed = out
        row = {
            "epoch": int(epoch),
            "train_err": float(train_err),
            "avg_loss": float(avg_loss),
            "epoch_train_time": float(elapsed),
            "train_samples_per_second": (
                len(train_loader.dataset) / float(elapsed)
                if elapsed > 0 else 0.0
            ),
        }
        if self._schedule_stats:
            row.update({f"contrast_{k}": float(v)
                        for k, v in self._schedule_stats.items()
                        if isinstance(v, (int, float))})
            if "thetas" in self._schedule_stats:
                row["contrast_thetas"] = list(self._schedule_stats["thetas"])
                row["contrast_bin_counts"] = list(
                    self._schedule_stats.get("bin_counts", []))
        if hasattr(training_loss, "pop_term_means"):
            row.update({
                f"train_{name}_term": float(value)
                for name, value in training_loss.pop_term_means().items()
            })
        self._last_train = row
        if self.eval_interval and epoch % self.eval_interval != 0:
            self._flush_row({})
        return out

    def evaluate_all(self, *args, **kwargs):
        metrics = super().evaluate_all(*args, **kwargs)
        self._flush_row({key: float(value) for key, value in metrics.items()})
        return metrics

    def resume_state_from_dir(self, save_dir):
        super().resume_state_from_dir(save_dir)
        # neuralop manifests store the epoch that just completed.
        self.start_epoch += 1
        if self.verbose:
            print(f"Continuing with epoch {self.start_epoch}")

    def _flush_row(self, eval_metrics: dict) -> None:
        if self.metrics_path is None or self._last_train is None:
            return
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps({**self._last_train, **eval_metrics}) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_2d_model(kind: str, in_channels: int):
    """Build a 2-D x_HI model and its serialized configuration."""
    if kind not in _KIND_SUFFIX:
        raise ValueError(f"MODEL_KIND must be one of {sorted(_KIND_SUFFIX)}")
    if kind == "ufno":
        from models_zre_2d import UFNO2d

        model = UFNO2d(
            modes1=N_MODES[0], modes2=N_MODES[1], width=UFNO_WIDTH,
            in_channels=in_channels, out_channels=1, sigmoid=True,
            norm=UFNO_NORM,
        )
        config = {
            "kind": kind, "n_modes": list(N_MODES),
            "in_channels": in_channels, "out_channels": 1,
            "ufno_width": UFNO_WIDTH, "ufno_norm": UFNO_NORM,
        }
        description = (
            f"U-FNO2d modes={N_MODES} width={UFNO_WIDTH} norm={UFNO_NORM}"
        )
        return model, config, description
    if kind in _LOCAL_KINDS:
        from models_zre_2d import LocalFNO2d

        slots = (
            OPERATOR_SLOTS if kind == MODEL_KIND
            else OperatorSlots.from_env(kind)
        )
        model = LocalFNO2d(
            in_channels=in_channels,
            out_channels=1,
            base_width=LOCALFNO_BASE_WIDTH,
            local_window=LOCALFNO_WINDOW,
            local_modes=LOCALFNO_MODES,
            global_modes=LOCALFNO_GLOBAL_MODES,
            spectral_rank=LOCALFNO_SPECTRAL_RANK,
            patch_chunk_size=LOCALFNO_PATCH_CHUNK_SIZE,
            output_sigmoid=True,
            **slots.model_kwargs(),
        )
        config = {
            "kind": kind,
            "in_channels": in_channels,
            "out_channels": 1,
            "localfno_base_width": LOCALFNO_BASE_WIDTH,
            "localfno_window": list(LOCALFNO_WINDOW),
            "localfno_global_modes": list(LOCALFNO_GLOBAL_MODES),
            "localfno_spectral_rank": LOCALFNO_SPECTRAL_RANK,
            "localfno_patch_chunk_size": LOCALFNO_PATCH_CHUNK_SIZE,
            **slots.metadata(),
        }
        if slots.uses_local_modes():
            config["localfno_modes"] = list(LOCALFNO_MODES)
        if "wavelet" in {slots.local, slots.global_}:
            # Retained for readers of older metadata that predate the registry.
            config.update(localwno_levels=LOCALWNO_LEVELS,
                          localwno_wavelet="haar")
        local_modes = (
            f"local-modes={LOCALFNO_MODES} " if slots.uses_local_modes() else ""
        )
        description = (
            f"{slots.model_name}2d window={LOCALFNO_WINDOW} {local_modes}"
            f"{slots.describe()} "
            f"global-modes={LOCALFNO_GLOBAL_MODES} "
            f"width={LOCALFNO_BASE_WIDTH} rank={LOCALFNO_SPECTRAL_RANK}"
        )
        return model, config, description

    model = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=in_channels,
        out_channels=1,
        n_layers=N_LAYERS,
        projection_channel_ratio=2,
        positional_embedding="grid",
    )
    config = {
        "kind": kind,
        "in_channels": in_channels,
        "out_channels": 1,
        "n_modes": list(N_MODES),
        "hidden_channels": HIDDEN_CHANNELS,
        "n_layers": N_LAYERS,
    }
    description = (
        f"FNO2d modes={N_MODES} hidden={HIDDEN_CHANNELS} layers={N_LAYERS}"
    )
    return model, config, description


def build_losses():
    weights = (
        LOSS_L2_WEIGHT, LOSS_H1_WEIGHT, LOSS_BCE_WEIGHT,
        LOSS_SWD_WEIGHT, LOSS_HIGHK_WEIGHT,
        LOSS_WALL_WEIGHT, LOSS_H1SEMI_WEIGHT, LOSS_EXPWALL_WEIGHT,
    )
    if all(weight <= 0 for weight in weights):
        raise ValueError("at least one loss weight must be positive")
    l2 = AbsoluteLoss(LpLoss(d=2, p=2))
    h1 = AbsoluteLoss(H1Loss(d=2))
    bce = BinaryCrossEntropyTerm()
    swd = SlicedWassersteinEdges(n_directions=SWD_DIRECTIONS, seed=RUN_SEED)
    highk = HighKPowerRatio(k_min=HIGHK_MIN)
    wall = WallPlacementLoss(cap=WALL_CAP)
    h1semi = H1Seminorm(cap=float(H1SEMI_CAP) if H1SEMI_CAP else None)
    expwall = ExponentialWallDistance(scale=EXPWALL_SCALE, cap=WALL_CAP)
    training = ScheduledWeightedLoss(
        (LOSS_L2_WEIGHT, l2),
        (LOSS_H1_WEIGHT, h1),
        (LOSS_BCE_WEIGHT, bce),
        (LOSS_SWD_WEIGHT, swd),
        (LOSS_HIGHK_WEIGHT, highk),
        (LOSS_WALL_WEIGHT, wall),
        (LOSS_H1SEMI_WEIGHT, h1semi),
        (LOSS_EXPWALL_WEIGHT, expwall),
        warmup_terms=(3, 4),
        warmup_epochs=LOSS_EDGE_WARMUP_EPOCHS,
        term_names=("l2", "h1", "bce", "swd", "highk", "wall", "h1semi",
                    "expwall"),
    )
    return training, {
        "l2": l2, "h1": h1, "bce": bce, "swd": swd, "highk": highk,
        "wall": wall, "h1semi": h1semi, "expwall": expwall,
    }


def _gradient_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = prediction - target
    dx = torch.roll(error, -1, dims=-2) - error
    dy = torch.roll(error, -1, dims=-1) - error
    return dx.square().sum() + dy.square().sum()


@torch.no_grad()
def final_report(model, loaders, device) -> dict[str, float]:
    """Pixel and high-frequency metrics for resolving transverse detail."""
    model.eval()
    report: dict[str, float] = {}
    for split, loader in loaders.items():
        squared_error = gradient_error = mean_error = 0.0
        pixel_count = gradient_count = sample_count = 0
        pred_power = truth_power = cross_power = 0.0
        for sample in loader:
            x = sample["x"].to(device)
            target = sample["y"].to(device)
            prediction = model(x)
            error = prediction - target
            squared_error += float(error.square().sum())
            pixel_count += error.numel()
            gradient_error += float(_gradient_error(prediction, target))
            gradient_count += 2 * error.numel()
            mean_error += float(torch.abs(
                prediction.mean(dim=(-2, -1)) - target.mean(dim=(-2, -1))
            ).sum())
            sample_count += prediction.shape[0]

            height, width = prediction.shape[-2:]
            fx = torch.fft.fftfreq(height, device=prediction.device)[:, None]
            fy = torch.fft.rfftfreq(width, device=prediction.device)[None, :]
            high_k = torch.sqrt(fx.square() + fy.square()) >= 0.25
            pred_fft = torch.fft.rfft2(prediction, norm="ortho")
            truth_fft = torch.fft.rfft2(target, norm="ortho")
            pred_high = pred_fft[..., high_k]
            truth_high = truth_fft[..., high_k]
            pred_power += float(pred_high.abs().square().sum())
            truth_power += float(truth_high.abs().square().sum())
            cross_power += float(
                (pred_high * truth_high.conj()).real.sum()
            )

        report[f"{split}_rmse"] = float(np.sqrt(squared_error / pixel_count))
        report[f"{split}_gradient_rmse"] = float(
            np.sqrt(gradient_error / gradient_count)
        )
        report[f"{split}_mean_xhi_mae"] = mean_error / sample_count
        report[f"{split}_high_k_power_ratio"] = pred_power / max(
            truth_power, 1e-12
        )
        report[f"{split}_high_k_cross_correlation"] = cross_power / max(
            np.sqrt(pred_power * truth_power), 1e-12
        )
    return report


@torch.no_grad()
def save_test_figure(model, dataset, test_ds, device, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(6, len(test_ds.indices))
    picker = np.random.default_rng(SPLIT_SEED)
    indices = picker.choice(test_ds.indices, size=count, replace=False)
    fig, axes = plt.subplots(count, 3, figsize=(10, 3.1 * count), squeeze=False)
    model.eval()
    for row, index in enumerate(indices):
        sample = dataset[int(index)]
        prediction = model(sample["x"][None].to(device))[0, 0].cpu().numpy()
        truth = sample["y"][0].numpy()
        error = prediction - truth
        panels = (
            (truth, "viridis", 0.0, 1.0, "truth"),
            (prediction, "viridis", 0.0, 1.0, "prediction"),
            (error, "RdBu_r", -max(abs(error.min()), abs(error.max())),
             max(abs(error.min()), abs(error.max())), "error"),
        )
        for col, (image, cmap, vmin, vmax, title) in enumerate(panels):
            axis = axes[row, col]
            rendered = axis.imshow(image, origin="lower", cmap=cmap,
                                   vmin=vmin, vmax=vmax)
            prefix = (
                f"cone {int(sample['cone_id'])}, z={float(sample['z']):.2f}\n"
                if col == 0 else ""
            )
            axis.set_title(prefix + title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(rendered, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _cone_ids(cache: SliceCache, subset) -> list[int]:
    return sorted(np.unique(cache.cone_id[subset.indices]).astype(int).tolist())


def main() -> None:
    if not CACHE_FILE.is_file():
        print(
            f"Slice cache {CACHE_FILE} not found. Run "
            "python -m dataset.build_trainset first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if MODEL_KIND not in _KIND_SUFFIX:
        raise SystemExit(f"MODEL_KIND must be one of {sorted(_KIND_SUFFIX)}")

    _seed_everything(RUN_SEED)
    cache = SliceCache(CACHE_FILE, input_features=INPUT_FEATURES)
    train_ds, val_ds, test_ds = split_by_cone(
        cache, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION,
        seed=SPLIT_SEED,
    )
    normalization = None
    if cache.input_features.use_params:
        normalization = cache.fit_parameter_normalization(train_ds.indices)

    train_cones = _cone_ids(cache, train_ds)
    val_cones = _cone_ids(cache, val_ds)
    test_cones = _cone_ids(cache, test_ds)
    if set(train_cones) & set(val_cones) or set(train_cones) & set(test_cones):
        raise RuntimeError("cone leakage detected between splits")

    print(f"Cache: {CACHE_FILE} ({len(cache)} slices, "
          f"{len(np.unique(cache.cone_id))} cones)")
    print(f"Split: {len(train_ds)} / {len(val_ds)} / {len(test_ds)} slices")
    print(f"Input: {cache.in_channels} channels {cache.channel_names}")

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": DEVICE == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    refit_loader = None
    if CONTRAST_REFIT and CONTRAST_REFIT_CACHE:
        refit_cache = SliceCache(
            paths.compressed(CONTRAST_REFIT_CACHE),
            input_features=INPUT_FEATURES,
            parameter_normalization=cache.parameter_normalization,
        )
        # Hard guard rather than a comment: a refit pool containing held-out
        # cones would leak them into the model through the fitted schedule.
        pool_cones = set(int(c) for c in np.unique(refit_cache.cone_id))
        held_out = pool_cones - set(int(c) for c in train_cones)
        if held_out:
            raise SystemExit(
                f"CONTRAST_REFIT_CACHE {CONTRAST_REFIT_CACHE} contains "
                f"{len(held_out)} cones outside the training split "
                f"(e.g. {sorted(held_out)[:5]}); refitting on it would leak")
        refit_loader = DataLoader(refit_cache, shuffle=True, **loader_kwargs)
        print(f"Contrast refit pool: {CONTRAST_REFIT_CACHE} "
              f"({len(refit_cache)} slices, {len(pool_cones)} train cones)")
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    test_loaders = {"val": val_loader, "test": test_loader}

    inner, model_config, description = build_2d_model(
        MODEL_KIND, cache.in_channels
    )
    if CONTRAST_MODE != "off":
        schedule = None
        if CONTRAST_SCHEDULE.startswith("theta="):
            v = float(CONTRAST_SCHEDULE.split("=", 1)[1])
            schedule = {"theta_lo": v, "theta_hi": v, "c": -1.5, "s": 0.4}
        elif CONTRAST_SCHEDULE:
            schedule = {k: float(v) for k, v in
                        json.loads(Path(CONTRAST_SCHEDULE).read_text()).items()
                        if k in ("theta_lo", "theta_hi", "c", "s")}
        inner = ContrastComposed(inner, CONTRAST_MODE, schedule=schedule,
                                 freeze=CONTRAST_FREEZE,
                                 schedule_kind=CONTRAST_SCHEDULE_KIND,
                                 n_bins=CONTRAST_BINS)
        model_config = {**model_config, "contrast_mode": CONTRAST_MODE,
                        "contrast_schedule": schedule,
                        "contrast_freeze": CONTRAST_FREEZE,
            "contrast_schedule_kind": CONTRAST_SCHEDULE_KIND,
            "contrast_bins": CONTRAST_BINS,
                        "contrast_schedule_kind": CONTRAST_SCHEDULE_KIND,
                        "contrast_bins": CONTRAST_BINS}
        detail = inner.contrast.describe()
        description = (f"{description} + contrast[{CONTRAST_MODE}"
                       f"{'' if detail is None else ': ' + detail}]")
    model = TrainerModel(inner).to(DEVICE)
    print(f"Model: {description} -> {count_model_params(model.fno):,} parameters")

    if GRAD_CLIP_NORM > 0:
        def _clip_before_step(optim, args, kwargs):
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=GRAD_CLIP_NORM)
            return None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    if GRAD_CLIP_NORM > 0:
        optimizer.register_step_pre_hook(_clip_before_step)
        print(f"Gradient clipping: max_norm={GRAD_CLIP_NORM}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )
    training_loss, eval_losses = build_losses()
    # Stateless copy of the active terms, so refitting the contrast map does
    # not pollute the trainer's per-term running means, and so the map is
    # fitted under the loss the run actually trains on.
    _active = [(w, eval_losses[n]) for n, w in zip(
        ("l2", "h1", "bce", "swd", "highk", "wall", "h1semi", "expwall"),
        (LOSS_L2_WEIGHT, LOSS_H1_WEIGHT, LOSS_BCE_WEIGHT, LOSS_SWD_WEIGHT,
         LOSS_HIGHK_WEIGHT, LOSS_WALL_WEIGHT, LOSS_H1SEMI_WEIGHT,
         LOSS_EXPWALL_WEIGHT)) if w > 0]

    def refit_objective(out, y):
        return sum(w * term(out, y) for w, term in _active)


    metadata = {
        "task": "2d",
        "model_config": model_config,
        "input_features": {
            "name": INPUT_FEATURES,
            "in_channels": cache.in_channels,
            "channel_names": list(cache.channel_names),
        },
        "target": {
            "name": "x_HI",
            "field": "neutral_fraction",
            "spatial_dimensions": 2,
        },
        "dataset": {
            "cache_file": str(CACHE_FILE.resolve()),
            **cache.selection,
        },
        "parameter_normalization": (
            normalization.to_dict() if normalization is not None else None
        ),
        "split": {
            "seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "train_cone_ids": train_cones,
            "val_cone_ids": val_cones,
            "test_cone_ids": test_cones,
        },
        "training": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "eval_interval": EVAL_INTERVAL,
            # Every term, so a checkpoint records which loss produced it.
            # wall/h1semi/expwall were missing here when they were added, so
            # runs that used them recorded all-zero weights and are not
            # self-describing; util/backfill_2d_loss_weights.py repairs those.
            "loss_weights": {
                "l2": LOSS_L2_WEIGHT,
                "h1": LOSS_H1_WEIGHT,
                "bce": LOSS_BCE_WEIGHT,
                "swd": LOSS_SWD_WEIGHT,
                "highk": LOSS_HIGHK_WEIGHT,
                "wall": LOSS_WALL_WEIGHT,
                "h1semi": LOSS_H1SEMI_WEIGHT,
                "expwall": LOSS_EXPWALL_WEIGHT,
            },
            "loss_params": {
                "wall_cap": WALL_CAP,
                "expwall_scale": EXPWALL_SCALE,
                "h1semi_cap": H1SEMI_CAP,
            },
            "edge_terms": {
                "warmup_epochs": LOSS_EDGE_WARMUP_EPOCHS,
                "swd_directions": SWD_DIRECTIONS,
                "highk_k_min": HIGHK_MIN,
            },
            "loss_modes": {"l2": "absolute", "h1": "absolute"},
            "contrast_mode": CONTRAST_MODE,
            "contrast_schedule": CONTRAST_SCHEDULE,
            "contrast_freeze": CONTRAST_FREEZE,
            "contrast_schedule_kind": CONTRAST_SCHEDULE_KIND,
            "contrast_bins": CONTRAST_BINS,
            "run_seed": RUN_SEED,
            "resume_dir": RESUME_DIR,
        },
    }
    write_run_metadata(CHECKPOINT_DIR, metadata)

    trainer = SliceLoggingTrainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        data_processor=None,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        use_distributed=False,
        verbose=True,
        metrics_path=CHECKPOINT_DIR / "metrics.jsonl",
        append=RESUME_DIR is not None,
        contrast_refit=CONTRAST_REFIT,
        refit_loader=refit_loader,
        refit_samples=CONTRAST_REFIT_SAMPLES,
        refit_steps=CONTRAST_REFIT_STEPS,
        refit_objective=refit_objective,
    )

    if CONTRAST_REFIT:
        print(f"Contrast refit: ON -- epoch 0 unmapped, then theta(mean_pred) "
              f"refit each epoch on {CONTRAST_REFIT_SAMPLES} slices "
              f"({CONTRAST_REFIT_STEPS} steps), frozen during training")
    print(f"Device: {DEVICE}; batch={BATCH_SIZE}; epochs={N_EPOCHS}; "
          f"lr={LEARNING_RATE:g}")
    print(f"Loss: {LOSS_L2_WEIGHT:g}*absL2 + "
          f"{LOSS_H1_WEIGHT:g}*absH1 + {LOSS_BCE_WEIGHT:g}*BCE + "
          f"{LOSS_SWD_WEIGHT:g}*SWD + {LOSS_HIGHK_WEIGHT:g}*highK + "
          f"{LOSS_WALL_WEIGHT:g}*wall[cap={WALL_CAP}] + "
          f"{LOSS_H1SEMI_WEIGHT:g}*H1semi + "
          f"{LOSS_EXPWALL_WEIGHT:g}*expwall[scale={EXPWALL_SCALE:g},cap={WALL_CAP}]"
          + (f" (edge terms warm up over {LOSS_EDGE_WARMUP_EPOCHS} epochs)"
             if (LOSS_SWD_WEIGHT or LOSS_HIGHK_WEIGHT)
             and LOSS_EDGE_WARMUP_EPOCHS else ""))
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_every=max(1, N_EPOCHS // 4),
        save_dir=str(CHECKPOINT_DIR),
        resume_from_dir=RESUME_DIR,
    )

    # Periodic resumable state may lag the final epoch. Preserve the exact
    # model evaluated below before running the comparatively expensive report.
    model.save_checkpoint(CHECKPOINT_DIR, "final_model")
    torch.save(optimizer.state_dict(), CHECKPOINT_DIR / "optimizer.pt")
    torch.save(scheduler.state_dict(), CHECKPOINT_DIR / "scheduler.pt")

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    report = final_report(model, loaders, DEVICE)
    report.update(
        model_kind=MODEL_KIND,
        model_description=description,
        n_epochs=N_EPOCHS,
        n_slices=len(cache),
        n_cones=len(np.unique(cache.cone_id)),
    )
    report_path = CHECKPOINT_DIR / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    figure_path = CHECKPOINT_DIR / "xhi2d_test_prediction.png"
    save_test_figure(model, cache, test_ds, DEVICE, figure_path)
    print(f"Final report: {report_path}")
    print(f"Test figure: {figure_path}")
    print(f"Test RMSE: {report['test_rmse']:.6f}; "
          f"gradient RMSE: {report['test_gradient_rmse']:.6f}; "
          f"high-k correlation: {report['test_high_k_cross_correlation']:.6f}")


if __name__ == "__main__":
    main()
