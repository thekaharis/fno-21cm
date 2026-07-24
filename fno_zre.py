#!/usr/bin/env python3
"""Train a 2-D FNO mapping density lightcones to z_re(x, y) maps.

Mapping:  matter density cone (LOS slices as input channels)  ->  z_re map.

The target is the per-pixel least-squares front midpoint computed by
``dataset/zre_target.py`` (Gompertz definition by default), normalized to
[0, 1] over the cone's redshift range. Pixels whose transition lies outside
the cone are clamped to the z_min edge and flagged by a mask; the masked MSE
is reported separately so the clamp never hides model error.

Environment overrides (defaults in parentheses):
  LIGHTCONE_DIR     lightcone directory (data)
  ZRE_TARGET_CACHE  target sidecar HDF5 (zre_targets.h5), built on demand
  ZRE_INPUT_CACHE   input sidecar HDF5 (zre_inputs.h5): density slices +
                    params (build: python -m dataset.zre_input_cache);
                    absent -> raw lightcone reads at startup
  TARGET_KIND       gompertz | step (gompertz)
  INPUT_FEATURES    density | density_params (density_params)
  N_Z_IN            LOS slices used as input channels (64)
  MODEL_KIND        fno | ufno | localfno | localwno | sirenfno |
                    localsirenfno (fno)
  N_MODES_X/Y (32), HIDDEN_CHANNELS (64), N_LAYERS (4)   [fno]
  UFNO_WIDTH (32), UFNO_NORM batchnorm|groupnorm          [ufno]
  LOCALFNO_BASE_WIDTH (16), LOCALFNO_WINDOW_X/Y (16),
  LOCALFNO_MODES_X/Y (6), LOCALFNO_GLOBAL_MODES_X/Y (16),
  LOCALFNO_SPECTRAL_RANK (16)                              [localfno]
  LOCALWNO_LEVELS (2)                                      [localwno]
  SIREN_HIDDEN_DIM (64), SIREN_OMEGA (30.0), SIREN_N_HIDDEN (1),
  SIREN_FEATURE_DIM (16), SIREN_FF_SIGMA (128.0),
  SIREN_LEARNABLE_FF (1), SIREN_MLP_DROPOUT (0.0),
  SIREN_SIGMOID_TEMPERATURE (2.0)                          [sirenfno]
  BATCH_SIZE (2), LEARNING_RATE (5e-4 fno, 1e-4 ufno/localfno),
  WEIGHT_DECAY (1e-5), N_EPOCHS (200)
  LOSS_L2_WEIGHT (0.5), LOSS_H1_WEIGHT (0.5),
  LOSS_L1_WEIGHT (0.0), LOSS_H2_WEIGHT (0.0),
  LOSS_RELATIVE (0 = absolute norms, 1 = relative norms)
  GRAD_CLIP_NORM    max grad norm per step; 0 = off
                    (1.0 for sirenfno, 0.0 otherwise)
  CHECKPOINT_DIR (checkpoints/checkpoints_zre[_<kind>]), EVAL_INTERVAL (5)
  RUN_SEED (0), DEVICE (auto: cuda > mps > cpu)
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

from neuralop.models import FNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

import neuralop as _neuralop
print(f"[fno_zre] using neuralop from {_neuralop.__file__}")

from dataset.dataset_zre import ZreMapDataset, split_by_cone
from dataset.zre_target import TARGET_KINDS, build_target_cache
from losses import AbsoluteLoss, H2Loss2d, RelativeLoss, WeightedLoss
from modeling import TrainerModel


# ------------------------------------------------------------------ config
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"
TARGET_CACHE = Path(os.environ.get("ZRE_TARGET_CACHE", "zre_targets.h5"))
# Input sidecar (density slices + params; dataset/zre_input_cache.py).
# Missing file -> silent fallback to reading the raw lightcones.
INPUT_CACHE = Path(os.environ.get("ZRE_INPUT_CACHE", "zre_inputs.h5"))
TARGET_KIND = os.environ.get("TARGET_KIND", "gompertz").lower()
INPUT_FEATURES = os.environ.get("INPUT_FEATURES", "density_params").lower()

N_Z_IN = int(os.environ.get("N_Z_IN", "64"))
Z_MIN, Z_MAX = 5.0, 25.0

MODEL_KIND = os.environ.get("MODEL_KIND", "fno").lower()
N_MODES = (
    int(os.environ.get("N_MODES_X", "32")),
    int(os.environ.get("N_MODES_Y", "32")),
)
HIDDEN_CHANNELS = int(os.environ.get("HIDDEN_CHANNELS", "64"))
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))
UFNO_WIDTH = int(os.environ.get("UFNO_WIDTH", "32"))
UFNO_NORM = os.environ.get("UFNO_NORM", "batchnorm").lower()
LOCALFNO_BASE_WIDTH = int(os.environ.get("LOCALFNO_BASE_WIDTH", "16"))
LOCALFNO_WINDOW = (
    int(os.environ.get("LOCALFNO_WINDOW_X", "16")),
    int(os.environ.get("LOCALFNO_WINDOW_Y", "16")),
)
LOCALFNO_MODES = (
    int(os.environ.get("LOCALFNO_MODES_X", "6")),
    int(os.environ.get("LOCALFNO_MODES_Y", "6")),
)
LOCALFNO_SPECTRAL_RANK = int(os.environ.get("LOCALFNO_SPECTRAL_RANK", "16"))
LOCALFNO_PATCH_CHUNK_SIZE = int(
    os.environ.get("LOCALFNO_PATCH_CHUNK_SIZE", "32")
)
LOCALWNO_LEVELS = int(os.environ.get("LOCALWNO_LEVELS", "2"))
SIREN_HIDDEN_DIM = int(os.environ.get("SIREN_HIDDEN_DIM", "64"))
SIREN_OMEGA = float(os.environ.get("SIREN_OMEGA", "30.0"))
SIREN_N_HIDDEN = int(os.environ.get("SIREN_N_HIDDEN", "1"))
SIREN_FEATURE_DIM = int(os.environ.get("SIREN_FEATURE_DIM", "16"))
SIREN_FF_SIGMA = float(os.environ.get("SIREN_FF_SIGMA", "128.0"))
SIREN_LEARNABLE_FF = os.environ.get("SIREN_LEARNABLE_FF", "1").strip() == "1"
SIREN_MLP_DROPOUT = float(os.environ.get("SIREN_MLP_DROPOUT", "0.0"))
SIREN_SIGMOID_TEMPERATURE = float(
    os.environ.get("SIREN_SIGMOID_TEMPERATURE", "2.0")
)
# The z_re target's clamped-to-zero majority makes a sigmoid output a
# saturation trap for the SIREN variant (logits run to the rails within
# ~60 batches and gradients die); the task's best model (plain FNO) uses a
# linear output. Default 0 = linear; set 1 to restore the 3-D-style sigmoid.
SIREN_OUTPUT_SIGMOID = (
    os.environ.get("SIREN_OUTPUT_SIGMOID", "0").strip() == "1"
)
# The LocalFNO bottleneck runs at 1/4 map resolution (35x35 for 140x140
# cones), so its global modes are capped by 35//2 = 17 -- keep them separate
# from the full-resolution N_MODES used by the plain FNO.
LOCALFNO_GLOBAL_MODES = (
    int(os.environ.get("LOCALFNO_GLOBAL_MODES_X", "16")),
    int(os.environ.get("LOCALFNO_GLOBAL_MODES_Y", "16")),
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
# U-FNO's sigmoid output and LocalFNO's deep residual stack train more
# stably at a conservative LR, mirroring the 3-D pipeline's defaults.
_DEFAULT_LR = "5e-4" if MODEL_KIND == "fno" else "1e-4"
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", _DEFAULT_LR))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "200"))
LOSS_L2_WEIGHT = float(os.environ.get("LOSS_L2_WEIGHT", "0.5"))
LOSS_H1_WEIGHT = float(os.environ.get("LOSS_H1_WEIGHT", "0.5"))
LOSS_L1_WEIGHT = float(os.environ.get("LOSS_L1_WEIGHT", "0.0"))
LOSS_H2_WEIGHT = float(os.environ.get("LOSS_H2_WEIGHT", "0.0"))
# 0 keeps the established absolute norms; 1 switches every active term to
# its relative implementation (||out-y|| / ||y||), e.g. the L1+H2 ablation.
LOSS_RELATIVE = os.environ.get("LOSS_RELATIVE", "0").strip() == "1"
# Grad-norm clipping before every optimizer step; 0 disables. Defaults to
# 1.0 for sirenfno (which NaNs without it, like its 3-D twin) and off for
# the architectures that have trained stably unclipped.
GRAD_CLIP_NORM = float(os.environ.get(
    "GRAD_CLIP_NORM",
    "1.0" if MODEL_KIND in ("sirenfno", "localsirenfno") else "0.0",
))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "5"))

# Separate checkpoint directories per model kind so runs never overwrite
# each other (same convention as the 3-D pipeline).
_KIND_SUFFIX = {"fno": "", "ufno": "_ufno", "localfno": "_localfno",
                 "sirenfno": "_sirenfno",
                 "localsirenfno": "_localsirenfno",
                 "localwno": "_localwno"}
CHECKPOINT_DIR = Path(
    os.environ.get(
        "CHECKPOINT_DIR",
        f"checkpoints/checkpoints_zre{_KIND_SUFFIX.get(MODEL_KIND, '')}",
    )
)

# Resume a timed-out run from the periodic training state in a checkpoint
# dir (written every ``save_every`` epochs): restores model, optimizer,
# scheduler, and the epoch counter via the neuralop Trainer.
RESUME_DIR = os.environ.get("RESUME_DIR") or None

SPLIT_SEED = 42
RUN_SEED = int(os.environ.get("RUN_SEED", "0"))
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu",
)


class ZreLoggingTrainer(Trainer):
    """``neuralop.Trainer`` + per-epoch metrics appended to metrics.jsonl.

    Same schema as the 3-D pipeline's ``LoggingTrainer`` (one JSON object
    per epoch: ``epoch``, ``train_err``, ``avg_loss``, ``epoch_train_time``,
    plus ``val_*``/``test_*`` on eval epochs), which is exactly what the
    dashboard scans ``checkpoints/*/metrics.jsonl`` for.  Single-process
    only -- no DDP reductions.
    """

    def __init__(self, *args, metrics_path=None, append=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            # fresh runs start a fresh history; resumed runs append
            if not append:
                self.metrics_path.unlink(missing_ok=True)
        self._last_train: dict | None = None

    def train_one_epoch(self, epoch, train_loader, training_loss):
        out = super().train_one_epoch(epoch, train_loader, training_loss)
        train_err, avg_loss, _avg_lasso, t = out
        self._last_train = dict(
            epoch=int(epoch),
            train_err=float(train_err),
            avg_loss=float(avg_loss),
            epoch_train_time=float(t),
        )
        # eval epochs are flushed (with their metrics) by evaluate_all
        if self.eval_interval and (epoch % self.eval_interval != 0):
            self._flush_row({})
        return out

    def evaluate_all(self, *args, **kwargs):
        eval_metrics = super().evaluate_all(*args, **kwargs)
        self._flush_row({k: float(v) for k, v in eval_metrics.items()})
        return eval_metrics

    def resume_state_from_dir(self, save_dir):
        super().resume_state_from_dir(save_dir)
        # neuralop manifests store the epoch that just completed.
        self.start_epoch += 1
        if self.verbose:
            print(f"Continuing with epoch {self.start_epoch}")

    def _flush_row(self, eval_metrics: dict) -> None:
        if self.metrics_path is None or self._last_train is None:
            return
        row = {**self._last_train, **eval_metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def build_zre_model(kind: str, in_channels: int):
    """Construct the configured 2-D architecture (returns model, description)."""
    if kind == "ufno":
        from models_zre_2d import UFNO2d

        model = UFNO2d(
            modes1=N_MODES[0],
            modes2=N_MODES[1],
            width=UFNO_WIDTH,
            in_channels=in_channels,
            out_channels=1,
            sigmoid=True,
            norm=UFNO_NORM,
        )
        desc = (f"U-FNO2d modes={N_MODES} width={UFNO_WIDTH} "
                f"norm={UFNO_NORM} sigmoid-output")
        return model, desc
    if kind in ("localfno", "localsirenfno", "localwno"):
        from models_zre_2d import LocalFNO2d

        siren = kind == "localsirenfno"
        wavelet = kind == "localwno"
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
            siren=siren,
            siren_hidden_dim=SIREN_HIDDEN_DIM,
            siren_omega=SIREN_OMEGA,
            siren_n_hidden=SIREN_N_HIDDEN,
            siren_feature_dim=SIREN_FEATURE_DIM,
            siren_ff_sigma=SIREN_FF_SIGMA,
            siren_learnable_ff=SIREN_LEARNABLE_FF,
            local_operator="wavelet" if wavelet else "fourier",
            wavelet_levels=LOCALWNO_LEVELS,
        )
        name = (
            "LocalWNO2d" if wavelet
            else "LocalSirenFNO2d" if siren
            else "LocalFNO2d"
        )
        siren_desc = (
            f" siren={SIREN_HIDDEN_DIM}x{SIREN_N_HIDDEN} "
            f"ff={SIREN_FEATURE_DIM}@{SIREN_FF_SIGMA:g}"
            if siren
            else ""
        )
        wavelet_desc = (
            f" wavelet=haar levels={LOCALWNO_LEVELS}" if wavelet else ""
        )
        local_modes_desc = (
            "" if wavelet else f"local-modes={LOCALFNO_MODES} "
        )
        desc = (f"{name} window={LOCALFNO_WINDOW} "
                f"{local_modes_desc}"
                 f"global-modes={LOCALFNO_GLOBAL_MODES} "
                 f"widths={LOCALFNO_BASE_WIDTH}/{2 * LOCALFNO_BASE_WIDTH}/"
                 f"{4 * LOCALFNO_BASE_WIDTH} rank={LOCALFNO_SPECTRAL_RANK} "
                 f"chunk={LOCALFNO_PATCH_CHUNK_SIZE}"
                f"{siren_desc}{wavelet_desc} sigmoid-output")
        return model, desc
    if kind == "sirenfno":
        from models_zre_2d import SirenFNO2d

        model = SirenFNO2d(
            n_modes=N_MODES,
            hidden_channels=HIDDEN_CHANNELS,
            in_channels=in_channels,
            out_channels=1,
            n_layers=N_LAYERS,
            siren_hidden_dim=SIREN_HIDDEN_DIM,
            siren_omega=SIREN_OMEGA,
            siren_n_hidden=SIREN_N_HIDDEN,
            siren_feature_dim=SIREN_FEATURE_DIM,
            siren_ff_sigma=SIREN_FF_SIGMA,
            siren_learnable_ff=SIREN_LEARNABLE_FF,
            mlp_dropout=SIREN_MLP_DROPOUT,
            output_sigmoid=SIREN_OUTPUT_SIGMOID,
            sigmoid_temperature=SIREN_SIGMOID_TEMPERATURE,
        )
        out_kind = "sigmoid" if SIREN_OUTPUT_SIGMOID else "linear"
        desc = (f"SirenFNO2d modes={N_MODES} hidden={HIDDEN_CHANNELS} "
                f"layers={N_LAYERS} siren={SIREN_HIDDEN_DIM}x{SIREN_N_HIDDEN} "
                f"ff={SIREN_FEATURE_DIM}@{SIREN_FF_SIGMA} {out_kind}-output")
        return model, desc
    model = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=in_channels,
        out_channels=1,
        n_layers=N_LAYERS,
        projection_channel_ratio=2,
        positional_embedding="grid",
    )
    desc = (f"FNO2d modes={N_MODES} hidden={HIDDEN_CHANNELS} "
            f"layers={N_LAYERS} pos-emb=grid")
    return model, desc


class MaskedMSE:
    """Mean squared error restricted to pixels with a real transition.

    Complements the dense L2/H1 objective: the clamped (masked-out) pixels
    carry a physically meaningful value, but this metric shows how the model
    does where z_re is genuinely defined.
    """

    def __call__(self, out, y, mask=None, **_):
        if mask is None:
            return torch.mean((out - y) ** 2)
        weight = mask.sum().clamp(min=1.0)
        return (((out - y) ** 2) * mask).sum() / weight


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_losses():
    """Construct train/eval losses from the LOSS_* env configuration.

    Train terms are the positively weighted subset of L2/H1/L1/H2, each
    absolute by default (LOSS_RELATIVE=0). The default absolute norms avoid
    dividing by the near-zero target norm of late-reionizing cones (the
    normalized target is zero over clamped regions); LOSS_RELATIVE=1 opts
    into relative norms for ablation runs regardless. Eval losses always
    report absolute L1/L2/H1/H2 plus the masked MSE, so metrics stay
    comparable across loss ablations.
    """
    adapter = RelativeLoss if LOSS_RELATIVE else AbsoluteLoss
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    l1_loss = LpLoss(d=2, p=1)
    h2_loss = H2Loss2d()
    weighted_terms = (
        (LOSS_L2_WEIGHT, "L2", l2_loss),
        (LOSS_H1_WEIGHT, "H1", h1_loss),
        (LOSS_L1_WEIGHT, "L1", l1_loss),
        (LOSS_H2_WEIGHT, "H2", h2_loss),
    )
    if not any(weight > 0 for weight, _, _ in weighted_terms):
        raise SystemExit("At least one LOSS_*_WEIGHT must be positive")
    train_loss_fn = WeightedLoss(
        *[(weight, adapter(loss)) for weight, _, loss in weighted_terms],
        term_names=tuple(name.lower() for _, name, _ in weighted_terms),
    )
    eval_losses = {
        "l2": AbsoluteLoss(l2_loss),
        "h1": AbsoluteLoss(h1_loss),
        "l1": AbsoluteLoss(l1_loss),
        "h2": AbsoluteLoss(h2_loss),
        "masked_mse": MaskedMSE(),
    }
    norm_kind = "rel" if LOSS_RELATIVE else "abs"
    description = " + ".join(
        f"{weight}*{norm_kind}{name}"
        for weight, name, _ in weighted_terms
        if weight > 0
    )
    return train_loss_fn, eval_losses, description


@torch.no_grad()
def _final_report(model, loaders, dataset, device) -> dict:
    """Dense/masked MSE per split, in normalized and physical units."""
    model.eval()
    masked_mse = MaskedMSE()
    z_span = dataset.z_max - dataset.z_min
    report: dict[str, float] = {}
    for split, loader in loaders.items():
        dense_sum, masked_sum, n = 0.0, 0.0, 0
        for sample in loader:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            mask = sample["mask"].to(device)
            pred = model(x)
            b = x.shape[0]
            dense_sum += float(torch.mean((pred - y) ** 2)) * b
            masked_sum += float(masked_mse(pred, y, mask)) * b
            n += b
        report[f"{split}_mse_norm"] = dense_sum / n
        report[f"{split}_mse_masked_norm"] = masked_sum / n
        report[f"{split}_rmse_z"] = float(np.sqrt(dense_sum / n) * z_span)
        report[f"{split}_rmse_masked_z"] = float(
            np.sqrt(masked_sum / n) * z_span
        )
    return report


@torch.no_grad()
def _save_test_figure(model, test_ds, dataset, device, out_path: Path,
                      max_cones: int | None = None) -> None:
    """Truth vs. prediction maps for a sample of test cones.

    Capped at ``max_cones`` rows (env FIGURE_MAX_CONES, default 8): with a
    large design the test split holds hundreds of cones and one row each
    would produce a several-hundred-megapixel PNG no viewer can open.
    The sample is seeded, so repeated runs show the same cones.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_cones is None:
        max_cones = int(os.environ.get("FIGURE_MAX_CONES", "8"))
    indices = list(test_ds.indices)
    total = len(indices)
    if total > max_cones:
        picker = np.random.default_rng(SPLIT_SEED)
        indices = sorted(
            picker.choice(total, size=max_cones, replace=False).tolist()
        )
        indices = [test_ds.indices[i] for i in indices]

    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.9 * n), dpi=140,
                             squeeze=False)
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        pred = model(sample["x"][None].to(device))[0, 0].cpu().numpy()
        truth = sample["y"][0].numpy()
        mask = sample["mask"][0].numpy() > 0.5
        truth_z = dataset.denormalize_zre(truth)
        pred_z = dataset.denormalize_zre(pred)
        vmin, vmax = truth_z.min(), max(truth_z.max(), 5.5)
        err = pred_z - truth_z
        stem = dataset.file_paths[idx].stem

        panels = (
            (truth_z, "magma", dict(vmin=vmin, vmax=vmax), f"{stem}\ntruth"),
            (pred_z, "magma", dict(vmin=vmin, vmax=vmax), "prediction"),
            (err, "RdBu_r",
             dict(vmin=-np.abs(err).max(), vmax=np.abs(err).max()),
             f"error (masked frac {1 - mask.mean():.0%})"),
        )
        for col, (img, cmap, kw, title) in enumerate(panels):
            ax = axes[row][col]
            im = ax.imshow(img, origin="lower", cmap=cmap,
                           interpolation="nearest", **kw)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"z_re(x, y): {n} of {total} test cones", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    if TARGET_KIND not in TARGET_KINDS:
        raise SystemExit(f"TARGET_KIND must be one of {TARGET_KINDS}")
    if INPUT_FEATURES not in {"density", "density_params"}:
        raise SystemExit("INPUT_FEATURES must be 'density' or 'density_params'")
    if MODEL_KIND not in _KIND_SUFFIX:
        raise SystemExit(
            f"MODEL_KIND must be one of {sorted(_KIND_SUFFIX)}, "
            f"got {MODEL_KIND!r}"
        )

    _seed_everything(RUN_SEED)

    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if len(files) < 3:
        print(f"Need at least 3 lightcones in {DATA_DIR}, found {len(files)}.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} lightcones in {DATA_DIR}")

    # Target maps are cached; first run pays the fitting cost once.
    build_target_cache(files, TARGET_CACHE, kind=TARGET_KIND)

    print(f"Input cache: {INPUT_CACHE} "
          f"({'found' if INPUT_CACHE.is_file() else 'absent - raw reads'})")
    dataset = ZreMapDataset(
        files,
        target_cache=TARGET_CACHE,
        target_kind=TARGET_KIND,
        n_z_in=N_Z_IN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        use_params=(INPUT_FEATURES == "density_params"),
        density_cache=INPUT_CACHE,
    )
    train_ds, val_ds, test_ds = split_by_cone(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION,
        seed=SPLIT_SEED,
    )
    normalization = dataset.fit_parameter_normalization(train_ds.indices)
    dataset.set_parameter_normalization(normalization)
    print(f"Split: train {len(train_ds)} / val {len(val_ds)} / "
          f"test {len(test_ds)} cones")
    print(f"Input: {dataset.in_channels} channels "
          f"({N_Z_IN} density slices"
          + (f" + {dataset.n_params} params" if dataset.n_params else "")
          + f"), map {dataset.map_shape}")
    print(f"Target: {TARGET_KIND} z_re, normalized over "
          f"z = [{Z_MIN}, {Z_MAX}]")

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0,
                     pin_memory=(DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    test_loaders = {"val": val_loader, "test": test_loader}

    inner, description = build_zre_model(MODEL_KIND, dataset.in_channels)
    model = TrainerModel(inner).to(DEVICE)
    print(f"Model: {description} -> "
          f"{count_model_params(model.fno):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    if GRAD_CLIP_NORM > 0:
        # Same mechanism as fno_21cm_3d.py: the SIREN hypernetwork diverges
        # to NaN within the first epoch without clipping (its 3-D twin
        # defaults to clip=1.0 for the same reason).
        def _clip_before_step(optim, args, kwargs):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
            return None

        optimizer.register_step_pre_hook(_clip_before_step)
        print(f"Gradient clipping: max_norm={GRAD_CLIP_NORM}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    train_loss_fn, eval_losses, loss_description = build_losses()

    from util.run_metadata import write_run_metadata
    write_run_metadata(CHECKPOINT_DIR, {
        "task": "zre",
        # Record every architecture knob, not just the generic ones: a sweep
        # over base width / window / local modes / omega is otherwise
        # indistinguishable in the run metadata (and therefore on the
        # dashboard), leaving the job log as the only record of what ran.
        "model_config": {
            "kind": MODEL_KIND,
            "in_channels": dataset.in_channels,
            "out_channels": 1,
            "n_modes": list(N_MODES),
            "hidden_channels": HIDDEN_CHANNELS,
            "n_layers": N_LAYERS,
            **({"ufno_width": UFNO_WIDTH, "ufno_norm": UFNO_NORM}
               if MODEL_KIND == "ufno" else {}),
            **({"localfno_base_width": LOCALFNO_BASE_WIDTH,
                "localfno_window": list(LOCALFNO_WINDOW),
                "localfno_modes": list(LOCALFNO_MODES),
                "localfno_global_modes": list(LOCALFNO_GLOBAL_MODES),
                "localfno_spectral_rank": LOCALFNO_SPECTRAL_RANK,
                "localfno_patch_chunk_size": LOCALFNO_PATCH_CHUNK_SIZE}
               if MODEL_KIND in ("localfno", "localsirenfno", "localwno") else {}),
            **({"localwno_levels": LOCALWNO_LEVELS,
                "localwno_wavelet": "haar"}
               if MODEL_KIND == "localwno" else {}),
            **({"siren_omega": SIREN_OMEGA,
                "siren_hidden_dim": SIREN_HIDDEN_DIM,
                "siren_n_hidden": SIREN_N_HIDDEN,
                "siren_feature_dim": SIREN_FEATURE_DIM,
                "siren_ff_sigma": SIREN_FF_SIGMA,
                "siren_output_sigmoid": SIREN_OUTPUT_SIGMOID}
               if MODEL_KIND in ("sirenfno", "localsirenfno") else {}),
        },
        "training": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": {
                "l2": LOSS_L2_WEIGHT,
                "h1": LOSS_H1_WEIGHT,
                "l1": LOSS_L1_WEIGHT,
                "h2": LOSS_H2_WEIGHT,
            },
            "loss_relative": LOSS_RELATIVE,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "target_kind": TARGET_KIND,
            "input_features": INPUT_FEATURES,
            "n_z_in": N_Z_IN,
            "run_seed": RUN_SEED,
            "resume_dir": RESUME_DIR,
        },
    })

    trainer = ZreLoggingTrainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        data_processor=None,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        use_distributed=False,
        verbose=True,
        metrics_path=CHECKPOINT_DIR / "metrics.jsonl",
        append=(RESUME_DIR is not None),
    )

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, "
          f"epochs: {N_EPOCHS}")
    print(f"Loss: {loss_description}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss_fn,
        eval_losses=eval_losses,
        save_every=max(1, N_EPOCHS // 4),
        save_dir=str(CHECKPOINT_DIR),
        resume_from_dir=RESUME_DIR,
    )

    model.save_checkpoint(CHECKPOINT_DIR, "final_model")
    torch.save(optimizer.state_dict(), CHECKPOINT_DIR / "optimizer.pt")
    torch.save(scheduler.state_dict(), CHECKPOINT_DIR / "scheduler.pt")

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    report = _final_report(model, loaders, dataset, DEVICE)
    report.update(
        model_kind=MODEL_KIND,
        model_description=description,
        target_kind=TARGET_KIND,
        input_features=INPUT_FEATURES,
        n_z_in=N_Z_IN,
        n_modes=list(N_MODES),
        hidden_channels=HIDDEN_CHANNELS,
        n_layers=N_LAYERS,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        run_seed=RUN_SEED,
        n_cones=len(files),
        test_cones=[dataset.file_paths[i].stem for i in test_ds.indices],
    )
    report_path = CHECKPOINT_DIR / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print("\nFinal metrics (RMSE in redshift units):")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: rmse_z {report[f'{split}_rmse_z']:.3f}  "
              f"masked rmse_z {report[f'{split}_rmse_masked_z']:.3f}")
    print(f"Report: {report_path}")

    fig_path = CHECKPOINT_DIR / "zre_test_prediction.png"
    _save_test_figure(model, test_ds, dataset, DEVICE, fig_path)
    print(f"Test prediction figure: {fig_path}")


if __name__ == "__main__":
    main()
