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
  TARGET_KIND       gompertz | step (gompertz)
  INPUT_FEATURES    density | density_params (density_params)
  N_Z_IN            LOS slices used as input channels (64)
  MODEL_KIND        fno | ufno | localfno (fno)
  N_MODES_X/Y (32), HIDDEN_CHANNELS (64), N_LAYERS (4)   [fno]
  UFNO_WIDTH (32), UFNO_NORM batchnorm|groupnorm          [ufno]
  LOCALFNO_BASE_WIDTH (16), LOCALFNO_WINDOW_X/Y (16),
  LOCALFNO_MODES_X/Y (6), LOCALFNO_GLOBAL_MODES_X/Y (16),
  LOCALFNO_SPECTRAL_RANK (16)                              [localfno]
  BATCH_SIZE (2), LEARNING_RATE (5e-4 fno, 1e-4 ufno/localfno),
  WEIGHT_DECAY (1e-5), N_EPOCHS (200)
  LOSS_L2_WEIGHT (0.5), LOSS_H1_WEIGHT (0.5)
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
from losses import AbsoluteLoss, WeightedLoss
from modeling import TrainerModel


# ------------------------------------------------------------------ config
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"
TARGET_CACHE = Path(os.environ.get("ZRE_TARGET_CACHE", "zre_targets.h5"))
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
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "5"))

# Separate checkpoint directories per model kind so runs never overwrite
# each other (same convention as the 3-D pipeline).
_KIND_SUFFIX = {"fno": "", "ufno": "_ufno", "localfno": "_localfno"}
CHECKPOINT_DIR = Path(
    os.environ.get(
        "CHECKPOINT_DIR",
        f"checkpoints/checkpoints_zre{_KIND_SUFFIX.get(MODEL_KIND, '')}",
    )
)

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
    if kind == "localfno":
        from models_zre_2d import LocalFNO2d

        model = LocalFNO2d(
            in_channels=in_channels,
            out_channels=1,
            base_width=LOCALFNO_BASE_WIDTH,
            local_window=LOCALFNO_WINDOW,
            local_modes=LOCALFNO_MODES,
            global_modes=LOCALFNO_GLOBAL_MODES,
            spectral_rank=LOCALFNO_SPECTRAL_RANK,
            output_sigmoid=True,
        )
        desc = (f"LocalFNO2d window={LOCALFNO_WINDOW} "
                f"local-modes={LOCALFNO_MODES} "
                f"global-modes={LOCALFNO_GLOBAL_MODES} "
                f"widths={LOCALFNO_BASE_WIDTH}/{2 * LOCALFNO_BASE_WIDTH}/"
                f"{4 * LOCALFNO_BASE_WIDTH} rank={LOCALFNO_SPECTRAL_RANK} "
                f"sigmoid-output")
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
def _save_test_figure(model, test_ds, dataset, device, out_path: Path) -> None:
    """Truth vs. prediction maps for every test cone (quick visual check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    n = len(test_ds)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.9 * n), dpi=140,
                             squeeze=False)
    for row, idx in enumerate(test_ds.indices):
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
    fig.suptitle("z_re(x, y): test cones", fontsize=13)
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

    dataset = ZreMapDataset(
        files,
        target_cache=TARGET_CACHE,
        target_kind=TARGET_KIND,
        n_z_in=N_Z_IN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        use_params=(INPUT_FEATURES == "density_params"),
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    # Both terms absolute, as in the 2-D slice pipeline: the normalized target
    # is zero over clamped regions, so a relative norm would divide by a
    # near-zero target norm on late-reionizing cones.
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    train_loss_fn = WeightedLoss(
        (LOSS_L2_WEIGHT, AbsoluteLoss(l2_loss)),
        (LOSS_H1_WEIGHT, AbsoluteLoss(h1_loss)),
    )
    eval_losses = {
        "l2": AbsoluteLoss(l2_loss),
        "h1": AbsoluteLoss(h1_loss),
        "masked_mse": MaskedMSE(),
    }

    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        data_processor=None,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        use_distributed=False,
        verbose=True,
    )

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, "
          f"epochs: {N_EPOCHS}")
    print(f"Loss: {LOSS_L2_WEIGHT}*absL2 + {LOSS_H1_WEIGHT}*absH1")

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
    )

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
