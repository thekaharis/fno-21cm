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
  MODEL_KIND        fno | ufno | localfno | localwno | localwhno |
                    sirenfno | localsirenfno | localop (fno)
  LOCAL_OPERATOR / GLOBAL_OPERATOR   with MODEL_KIND=localop, the two
                    operator slots of the local-global U-Net: fourier |
                    siren_fourier | wavelet | hadamard | cnn (fourier)
  WHNO_ORDERING sequency|natural (sequency)                [hadamard]
  CNN_DEPTH (3), CNN_KERNEL_SIZE (3), CNN_DROPOUT (0.0),
  CNN_NORM groupnorm|batchnorm                             [cnn]
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
  CHECKPOINT_DIR (checkpoints/checkpoints_zre[_<kind>]), EVAL_INTERVAL (1)
  RUN_SEED (0), DEVICE (auto: cuda > mps > cpu)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from training import MetricsTrainer
from util import seed_everything
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
from dataset import paths
from dataset.zre_target import TARGET_KINDS, build_target_cache
from losses import AbsoluteLoss, H2Loss2d, RelativeLoss, WeightedLoss
from modeling import ModelConfig, TrainerModel, build_model
from learned_waveform_operator import waveform_parameter_groups
from waveform_training import WaveformTrainingConfig, WaveformTrainingController, warm_start


# ------------------------------------------------------------------ config
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", paths.LIGHTCONES))
FILE_GLOB = "21cmfast_11d_sample*.h5"
TARGET_CACHE = Path(os.environ.get("ZRE_TARGET_CACHE", paths.ZRE_TARGETS))
# Input sidecar (density slices + params; dataset/zre_input_cache.py).
# Missing file -> silent fallback to reading the raw lightcones.
INPUT_CACHE = Path(os.environ.get("ZRE_INPUT_CACHE", paths.ZRE_INPUTS))
TARGET_KIND = os.environ.get("TARGET_KIND", "gompertz").lower()
INPUT_FEATURES = os.environ.get("INPUT_FEATURES", "density_params").lower()

N_Z_IN = int(os.environ.get("N_Z_IN", "64"))
Z_MIN, Z_MAX = 5.0, 25.0

# Same switches, registry and metadata as the x_HI entry points; only the task
# and ndim differ.
MODEL_CONFIG = ModelConfig.from_env(ndim=2)
WAVEFORM_TRAINING = WaveformTrainingConfig.from_env()
MODEL_KIND = MODEL_CONFIG.kind

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
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "1"))

# Separate checkpoint directories per model kind so runs never overwrite
# each other (same convention as the 3-D pipeline).
CHECKPOINT_DIR = Path(
    os.environ.get(
        "CHECKPOINT_DIR",
        f"checkpoints/checkpoints_zre_{MODEL_CONFIG.checkpoint_tag}",
    )
)

# Resume a timed-out run from the periodic training state in a checkpoint
# dir (written every ``save_every`` epochs): restores model, optimizer,
# scheduler, and the epoch counter via the neuralop Trainer.
RESUME_DIR = os.environ.get("RESUME_DIR") or None
INIT_CHECKPOINT = os.environ.get("INIT_CHECKPOINT") or None

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

    seed_everything(RUN_SEED)

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

    inner = build_model(MODEL_CONFIG, dataset.in_channels)
    description = MODEL_CONFIG.describe()
    model = TrainerModel(inner).to(DEVICE)
    report = warm_start(model, INIT_CHECKPOINT, resume_dir=RESUME_DIR,
                        strict=WAVEFORM_TRAINING.mode != "joint")
    if report is not None:
        print(f"Warm-started from {INIT_CHECKPOINT}: {report.matched}/{report.total} tensors matched")
    print(f"Model: {description} -> "
          f"{count_model_params(model.fno):,} parameters")

    optimizer = torch.optim.Adam(
        waveform_parameter_groups(
            model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            waveform_lr_ratio=MODEL_CONFIG.waveform_lr_ratio,
        ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    waveform_training = WaveformTrainingController(model, optimizer, WAVEFORM_TRAINING)
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
        "input_features": {
            "name": INPUT_FEATURES,
            "in_channels": dataset.in_channels,
            "spatial_shape": list(dataset.map_shape),
        },
        # Record every architecture knob, not just the generic ones: a sweep
        # over base width / window / local modes / omega is otherwise
        # indistinguishable in the run metadata (and therefore on the
        # dashboard), leaving the job log as the only record of what ran.
        "model_config": {**MODEL_CONFIG.to_dict(),
                         "in_channels": dataset.in_channels,
                         "out_channels": 1},
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
            "init_checkpoint": INIT_CHECKPOINT,
            "waveform_training": WAVEFORM_TRAINING.to_dict(),
        },
    })

    trainer = MetricsTrainer(
        waveform_training=waveform_training,
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

    from neuralop.training.training_state import save_training_state
    save_training_state(save_dir=CHECKPOINT_DIR, save_name="final_model",
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=trainer.epoch)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    report = _final_report(model, loaders, dataset, DEVICE)
    report.update(
        model_kind=MODEL_KIND,
        model_description=description,
        target_kind=TARGET_KIND,
        input_features=INPUT_FEATURES,
        n_z_in=N_Z_IN,
        n_modes=list(MODEL_CONFIG.modes),
        hidden_channels=MODEL_CONFIG.hidden_channels,
        n_layers=MODEL_CONFIG.n_layers,
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
