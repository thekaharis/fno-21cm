#!/usr/bin/env python3
"""Train a configurable 3-D neural operator on full 21cm lightcone cubes.

Mapping:  matter density cube  ->  neutral fraction (x_HI) cube.

Each lightcone is interpolated along the LOS axis to a fixed n_z grid so the
whole cube fits in a single forward pass on an A30 (24 GB) at batch=1.  The
input tensor carries the density (normalized by a fixed constant) and an
explicit ``1/(1+z)`` channel. FNO and SirenFNO append normalized grid
coordinates as additional channels. Because the cube cache is sampled
uniformly in redshift, the third grid coordinate is normalized redshift,
not comoving distance.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from neuralop import Trainer
from neuralop import LpLoss
from neuralop.training.training_state import save_training_state
from neuralop.utils import count_model_params

import neuralop as _neuralop
print(f"[fno_21cm_3d] using neuralop from {_neuralop.__file__}")

from dataset import paths
from dataset.dataset_3d import (
    InputFeatures,
    LightconeCubeDataset,
    LightconeCubeCache,
    split_cubes,
)
from losses import (
    AbsoluteLoss,
    BinaryCrossEntropyTerm,
    IonizedWallRMSE,
    ExponentialWallDistance,
    GranulometrySpectrum,
    LightconeH1Loss,
    LOSVolumeWeightedLoss,
    RelativeLoss,
    ScheduledWeightedLoss,
    WeightedLoss,
    los_volume_weights,
)
from contrast import ContrastComposed
from modeling import ModelConfig, TrainerModel, build_model, load_checkpoint
from training import (
    ContrastRefit,
    MetricsTrainer,
    ProgressLoader,
    all_reduce_mean,
    setup_distributed,
)
from util import seed_everything
from util.run_metadata import write_run_metadata
from util.spectral_weights import HISTORY_FILENAME, SpectralWeightHistory


# ------------------------------------------------------------------ config
# Lightcone directory (raw .h5 files): env var LIGHTCONE_DIR overrides; falls
# back to ./data so the SLURM sbatch can set the cluster path once without
# editing this file.
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"

# Pre-computed cube cache (built by dataset/build_cubes.py). Env var CUBES_CACHE
# overrides; default is ./cubes_3d.h5 next to the script.  If the cache file
# exists at startup, the training script uses LightconeCubeCache (fast,
# pre-interpolated cubes); otherwise it falls back to LightconeCubeDataset
# (raw streaming, ~10x slower per epoch).
CUBES_CACHE = Path(os.environ.get("CUBES_CACHE", paths.CUBES))

N_Z = 256                           # LOS resolution after interpolation
Z_MIN, Z_MAX = 5.0, 25.0

MODEL_CONFIG = ModelConfig.from_env()
INPUT_FEATURES = InputFeatures(
    os.environ.get("INPUT_FEATURES", "density_z_params").lower()
)
MODEL_KIND = MODEL_CONFIG.kind
N_MODES = MODEL_CONFIG.modes
HIDDEN_CHANNELS = MODEL_CONFIG.hidden_channels
UFNO_WIDTH = MODEL_CONFIG.ufno_width
UFNO_NORM = MODEL_CONFIG.ufno_norm
UFNO_UNET_VARIANT = MODEL_CONFIG.ufno_unet_variant
UFNO_GLOBAL_RESIDUAL = MODEL_CONFIG.ufno_global_residual
N_LAYERS = MODEL_CONFIG.n_layers
BATCH_SIZE = 1                      # 3-D cubes are heavy; raise after profiling
LEARNING_RATE = 5e-4
# U-FNO's sigmoid output can saturate under the derivative-heavy H1 objective.
# Use a more conservative base LR than the plain FNO; still apply the DDP
# scaling rule below. Both values remain environment-overridable.
UFNO_LEARNING_RATE = float(os.environ.get("UFNO_LEARNING_RATE", "1e-4"))
SIRENFNO_LEARNING_RATE = float(
    os.environ.get("SIRENFNO_LEARNING_RATE", "1e-4")
)
LOCALFNO_LEARNING_RATE = float(
    os.environ.get("LOCALFNO_LEARNING_RATE", "1e-4")
)
WEIGHT_DECAY = 1e-5
# N_EPOCHS overridable from the sbatch (U-FNO defaults to a shorter first run).
N_EPOCHS = int(os.environ.get("N_EPOCHS", "100"))
# Epochs between evaluation passes. The sbatch exports this; the definition
# was dropped in the pipeline refactor, so every run using it died at
# trainer construction with NameError until this was restored.
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", "1"))

# Loss term weights.  Defaults (0.5, 0.5, 0.0) match the v1 run.  All three
# are env-var overridable so experiments don't need to edit this file:
#   * (0.5, 0.5, 0.0)  v1 U-FNO baseline
#   * (0.5, 0.5, 0.5)  BCE-on experiment (Act 4)
#   * (0.3, 0.7, 0.0)  v2 H1-weighted (C of the v2 A+B+C bundle)
LOSS_L2_WEIGHT = float(os.environ.get("LOSS_L2_WEIGHT", "0.5"))
LOSS_H1_WEIGHT = float(os.environ.get("LOSS_H1_WEIGHT", "0.5"))
LOSS_BCE_WEIGHT = float(os.environ.get("LOSS_BCE_WEIGHT", "0.0"))

# On non-uniform LOS grids (warped cube caches) voxel count is loss weight,
# so densely sampled epochs dominate the L2/H1 terms in proportion to their
# slice count. "1" applies Delta-chi quadrature weights along the LOS so the
# loss is volume-weighted regardless of grid. No-op-ish for uniform-chi
# grids; mildly reweights uniform-z ones. Applies to L2/H1 only.
LOSS_LOS_VOLUME_WEIGHTS = (
    os.environ.get("LOSS_LOS_VOLUME_WEIGHTS", "0").strip() == "1"
)

# Norm mode per term. "absolute" (historical default) uses raw H1/L2 norms,
# under which the H1 term is ~2 orders of magnitude larger than the L2 term
# and the nominal weights above do not reflect the real balance. "relative"
# divides by the target norm, making both terms dimensionless and the weights
# directly interpretable.
def _loss_mode(name: str) -> str:
    mode = os.environ.get(name, "absolute").strip().lower()
    if mode not in {"absolute", "relative"}:
        raise ValueError(f"{name} must be 'absolute' or 'relative', got {mode!r}")
    return mode


LOSS_L2_MODE = _loss_mode("LOSS_L2_MODE")
LOSS_H1_MODE = _loss_mode("LOSS_H1_MODE")
LOSS_IONIZED_WALL_WEIGHT = float(
    os.environ.get("LOSS_IONIZED_WALL_WEIGHT", "0.0")
)
IONIZED_WALL_KERNEL_SIZE = int(
    os.environ.get("IONIZED_WALL_KERNEL_SIZE", "7")
)
# Exponential-in-distance wall placement (losses.ExponentialWallDistance).
# The 2-D sweep picked scale=16: it matched truth sharpness (width 1.33 vs
# 1.47, blur 0.108 vs 0.111) and halved the wall-placement error, at ~20% more
# RMSE. scale=4 diverged. Distances are in voxels, so the penalty is mildly
# anisotropic here -- the LOS axis is not on the transverse physical scale.
# EXPWALL_AXES="transverse" runs the distance transform per XY slice instead.
# Measured on this cache: transverse cell 1.43 Mpc vs LOS cell 9.7 Mpc median
# (41.7 Mpc at z=5), and the 3.6 Mpc truth front is 2.5 transverse cells but
# only 0.37 of a LOS cell -- unrepresentable along the LOS. The 3-D transform
# also almost never reaches `cap` (phi in [-6,+7], weight spread 1.5x, against
# [-32,+32] and 7x per-slice), so the exponential weighting barely engages.
EXPWALL_AXES = os.environ.get("EXPWALL_AXES", "3d").strip().lower()
LOSS_EXPWALL_WEIGHT = float(os.environ.get("LOSS_EXPWALL_WEIGHT", "0.0"))
# Ramp expwall in, because its magnitude relative to L2 *inverts* over training.
# Measured on real cubes: L2/expwall is 3.0 for a constant-0.5 prediction but
# 85.8 once the prediction is merely blurred. A fixed weight therefore either
# swamps L2 early or vanishes late; the ramp lets L2 establish structure first.
EXPWALL_WARMUP_EPOCHS = int(os.environ.get("EXPWALL_WARMUP_EPOCHS", "0"))
# Bubble-size spectrum (losses.GranulometrySpectrum): a differentiable stand-in
# for the MFP bubble-size distribution the evaluator reports. Auxiliary only --
# a size spectrum is blind to where the bubbles are, so it needs an L2 or
# expwall anchor exactly as the edge terms do. Radii are in cells after
# BSD_DOWNSAMPLE. With separable openings a full 256-slice cube at 4 radii and
# 2x downsampling costs ~164 ms/step on CPU; 32 slices ~32 ms.
LOSS_BSD_WEIGHT = float(os.environ.get("LOSS_BSD_WEIGHT", "0.0"))
BSD_RADII = tuple(int(v) for v in
                  os.environ.get("BSD_RADII", "1,2,4,8").split(","))
BSD_DOWNSAMPLE = int(os.environ.get("BSD_DOWNSAMPLE", "2"))
BSD_MAX_SLICES = int(os.environ.get("BSD_MAX_SLICES", "64"))
BSD_WARMUP_EPOCHS = int(os.environ.get("BSD_WARMUP_EPOCHS", "0"))
# Output contrast map (contrast.py), same machinery as the 2-D trainer. In 3-D
# theta is indexed per line-of-sight slice, not per cube: a cube spans the whole
# reionisation history, so one mean per cube averages x_HI ~ 0 and x_HI ~ 1
# together and describes neither.
CONTRAST_MODE = os.environ.get("CONTRAST_MODE", "off").lower()
CONTRAST_REFIT = os.environ.get("CONTRAST_REFIT", "0") not in ("0", "", "false")
CONTRAST_SCHEDULE_KIND = os.environ.get("CONTRAST_SCHEDULE_KIND", "stepped").lower()
CONTRAST_BINS = int(os.environ.get("CONTRAST_BINS", "14"))
# mean|monotone. The per-slice mean prediction estimates x_HI with MAE ~0.055,
# against a responsive band (x_HI 0.005-0.05) only 0.045 wide -- the key is
# noisier than what it has to resolve, which is why the 2-D gain pooled to
# -0.16%. "monotone" isotonically smooths the per-slice means along the LOS
# axis, using the one structure a cone has and a 2-D slice does not: its x_HI
# curve is monotone in redshift. Set "mean" for the unsmoothed 2-D behaviour.
CONTRAST_KEY = os.environ.get("CONTRAST_KEY", "monotone").lower()
# Counted in LOS slices, not cubes, in both refit modes -- and *per rank*, so
# under DDP the collective fit pools world_size times this many.
CONTRAST_REFIT_SAMPLES = int(os.environ.get("CONTRAST_REFIT_SAMPLES", "2048"))
CONTRAST_REFIT_STEPS = int(os.environ.get("CONTRAST_REFIT_STEPS", "300"))
# In cube mode this is a number of *cubes*, and it has to stay small: the fit
# holds the objective's intermediates for a whole batch of 5-D tensors.
CONTRAST_REFIT_BATCH = int(os.environ.get("CONTRAST_REFIT_BATCH", "2"))
# Split cubes into transverse LOS slices before fitting. Off by default: theta
# is constant within a slice so splitting buys nothing, and it changes the
# objective, since expwall's signed distance inside one transverse plane is not
# the 3-D distance the training loss computes.
CONTRAST_REFIT_FLATTEN = os.environ.get(
    "CONTRAST_REFIT_FLATTEN", "0") not in ("0", "", "false")
CONTRAST_THETA_FLOOR = float(os.environ.get("CONTRAST_THETA_FLOOR", "0.25"))
EXPWALL_SCALE = float(os.environ.get("EXPWALL_SCALE", "16.0"))
WALL_CAP = int(os.environ.get("WALL_CAP", "32"))
IONIZED_WALL_THRESHOLD = float(
    os.environ.get("IONIZED_WALL_THRESHOLD", "0.5")
)

# Stability controls for U-FNO. H1 starts at zero and ramps linearly to its
# configured weight, allowing the L2 value term to establish a non-saturated
# output before derivative matching begins.
UFNO_H1_WARMUP_EPOCHS = int(os.environ.get("UFNO_H1_WARMUP_EPOCHS", "5"))
UFNO_GRAD_CLIP_NORM = float(os.environ.get("UFNO_GRAD_CLIP_NORM", "1.0"))
SIRENFNO_H1_WARMUP_EPOCHS = int(
    os.environ.get("SIRENFNO_H1_WARMUP_EPOCHS", "5")
)
SIRENFNO_GRAD_CLIP_NORM = float(
    os.environ.get("SIRENFNO_GRAD_CLIP_NORM", "1.0")
)
LOCALFNO_H1_WARMUP_EPOCHS = int(
    os.environ.get("LOCALFNO_H1_WARMUP_EPOCHS", "5")
)
LOCALFNO_GRAD_CLIP_NORM = float(
    os.environ.get("LOCALFNO_GRAD_CLIP_NORM", "1.0")
)

# DataLoader workers.  Streamed loading (one ~370 MB HDF5 read per sample) is
# the throughput bottleneck on cluster filesystems; parallelizing across the
# allocated CPUs gets the GPU fed.  Defaults to SLURM_CPUS_PER_TASK on the
# cluster and 0 locally; override with NUM_WORKERS when the prefetch queue
# (num_workers x prefetch_factor x batch_size samples of ~280 MB) must fit a
# tighter host-memory budget.
NUM_WORKERS = int(
    os.environ.get("NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "0"))
)

# Per-step progress logging cadence (set to 0 to disable).
LOG_EVERY = 25

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

# Keep the data split fixed across repeated experiments, while RUN_SEED
# controls model initialization and training data order.
SPLIT_SEED = 42
RUN_SEED = int(os.environ.get("RUN_SEED", "0"))
DETERMINISTIC_RUN = os.environ.get(
    "DETERMINISTIC_RUN", "false"
).strip().lower() in {"1", "true", "yes", "on"}
if DETERMINISTIC_RUN:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# Separate checkpoint directories per model so a U-FNO run never overwrites
# the FNO baseline (or vice versa).  Override CHECKPOINT_DIR explicitly in
# the env for a one-off custom run.
_DEFAULT_CKPT = str(MODEL_CONFIG.default_checkpoint_dir)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _DEFAULT_CKPT)
METRICS_PATH = f"{CHECKPOINT_DIR}/metrics.jsonl"
SPECTRAL_HISTORY_PATH = f"{CHECKPOINT_DIR}/{HISTORY_FILENAME}"
# Optional model-only warm start. This intentionally does not restore the
# optimizer or scheduler, which is useful after a short feasibility run whose
# cosine schedule used a tiny N_EPOCHS (for example T_max=1).
INIT_CHECKPOINT = os.environ.get("INIT_CHECKPOINT")

# Learning-rate scaling rule for multi-GPU DDP runs.  "sqrt" is conservative
# and rarely diverges; "linear" extracts more wall-clock speed but may need
# warmup at large effective batches.
LR_SCALE_RULE = "sqrt"   # "linear" or "sqrt"


# ------------------------------------------------------------------ reproducibility
def _seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker a reproducible Python/NumPy RNG state."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ------------------------------------------------------------------ distributed
def _build_h1_loss() -> LightconeH1Loss:
    """H1 objective with periodic X/Y and centered interior LOS differences."""
    return LightconeH1Loss(measure=(1.0, 1.0, 1.0))


def main():
    # -------------------------------------------- 0. distributed setup
    rank, local_rank, world_size = setup_distributed()
    is_rank_0 = (rank == 0)
    is_distributed = (world_size > 1)
    # All ranks use the same initialization seed so DDP starts from identical
    # parameters. DistributedSampler applies rank-specific partitioning.
    seed_everything(RUN_SEED, deterministic=DETERMINISTIC_RUN)

    # Per-rank device.  Under DDP each rank pins to its own GPU; in single-GPU
    # mode this is just the module-level DEVICE.
    device = (f"cuda:{local_rank}" if torch.cuda.is_available()
              else DEVICE)

    def rprint(*args, **kwargs):
        """Print only on rank 0 (silent on other ranks)."""
        if is_rank_0:
            print(*args, **kwargs)

    rprint(f"\n[ddp] world_size={world_size}  rank={rank}  "
           f"local_rank={local_rank}  device={device}")

    # -------------------------------------------- 1. dataset (cache or stream)
    # Prefer the pre-built cube cache when it exists: ~10x faster reads per
    # sample, no scipy interpolation in the hot path.
    if CUBES_CACHE.exists():
        rprint(f"Using pre-computed cube cache: {CUBES_CACHE}")
        dataset = LightconeCubeCache(
            CUBES_CACHE,
            input_features=INPUT_FEATURES,
        )
        rprint(f"Dataset: {len(dataset)} cubes  "
               f"({dataset.n_x} x {dataset.n_y} x {dataset.n_z}, "
               f"z in [{dataset.target_z[0]:.2f}, {dataset.target_z[-1]:.2f}])")
        cache_cone_ids = np.asarray(dataset.cone_ids)
        if cache_cone_ids.size and np.any(np.diff(cache_cone_ids) < 0):
            rprint("WARNING: cube cache rows are not sorted by cone_id "
                   "(cache built before the sorted merge). Row-position "
                   "splits will not match the raw-streaming pipeline; the "
                   "run metadata records cone ids so visualization stays "
                   "consistent.")
    else:
        rprint(f"No cube cache at {CUBES_CACHE}; streaming raw lightcones "
               f"from {DATA_DIR}. Run python -m dataset.build_cubes to precompute and speed "
               f"up training ~10x.")
        files = sorted(DATA_DIR.glob(FILE_GLOB))
        if not files:
            rprint(f"No lightcone files found under {DATA_DIR}/{FILE_GLOB}",
                   file=sys.stderr)
            sys.exit(1)
        rprint(f"Found {len(files)} lightcone files in {DATA_DIR}")
        dataset = LightconeCubeDataset(
            file_paths=files,
            n_z=N_Z, z_min=Z_MIN, z_max=Z_MAX,
            preload=False,
            input_features=INPUT_FEATURES,
        )
        rprint(f"Dataset: {len(dataset)} cubes  ({N_Z} LOS cells each, "
               f"z in [{Z_MIN}, {Z_MAX}])")

    # -------------------------------------------- 2. split by cone
    train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx) = split_cubes(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED,
    )
    train_cone_ids = [int(dataset.cone_ids[i]) for i in train_idx]
    val_cone_ids = [int(dataset.cone_ids[i]) for i in val_idx]
    test_cone_ids = [int(dataset.cone_ids[i]) for i in test_idx]
    overlap = (
        set(train_cone_ids) & set(val_cone_ids)
        | set(train_cone_ids) & set(test_cone_ids)
        | set(val_cone_ids) & set(test_cone_ids)
    )
    assert not overlap, f"Split leakage (cone ids): {overlap}"
    parameter_normalization = dataset.fit_parameter_normalization(train_idx)
    dataset.set_parameter_normalization(parameter_normalization)
    rprint(f"Train: {len(train_ds)} cones {train_cone_ids}")
    rprint(f"Val:   {len(val_ds)} cones {val_cone_ids}")
    rprint(f"Test:  {len(test_ds)} cones {test_cone_ids}")

    # -------------------------------------------- 3. dataloaders
    # Under DDP each rank consumes a disjoint shard of each split.  The
    # DistributedSampler pads to be divisible by world_size if needed.
    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=False, seed=RUN_SEED,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False, seed=RUN_SEED,
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False, seed=RUN_SEED,
        )
        train_shuffle = None  # mutually exclusive with sampler
    else:
        train_sampler = val_sampler = test_sampler = None
        train_shuffle = True

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=_seed_worker,
    )
    train_generator = torch.Generator().manual_seed(RUN_SEED + rank)
    val_generator = torch.Generator().manual_seed(RUN_SEED + 10_000 + rank)
    test_generator = torch.Generator().manual_seed(RUN_SEED + 20_000 + rank)
    train_loader = DataLoader(train_ds, shuffle=train_shuffle,
                              sampler=train_sampler,
                              generator=train_generator, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False,
                            sampler=val_sampler,
                            generator=val_generator, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False,
                             sampler=test_sampler,
                             generator=test_generator, **dl_kwargs)

    # Wrap the train loader in a per-step progress reporter so the SLURM log
    # shows life within an epoch (the neuralop Trainer logs per-epoch only).
    train_loader = ProgressLoader(train_loader, log_every=LOG_EVERY,
                                  tag="train", rank=rank)
    val_loader = ProgressLoader(val_loader, log_every=LOG_EVERY,
                                tag="val", rank=rank)
    test_loader = ProgressLoader(test_loader, log_every=LOG_EVERY,
                                 tag="test", rank=rank)
    test_loaders = {"val": val_loader, "test": test_loader}

    # -------------------------------------------- 4. model + DDP wrap
    # in_channels read from the dataset so it auto-adjusts when parameter
    # conditioning is enabled (2 -> 13 with the 11 LHS params broadcast in).
    in_channels = dataset.in_channels
    rprint(f"Input channels ({in_channels}): "
           + ", ".join(dataset.input_features.channel_names))

    fno = build_model(MODEL_CONFIG, in_channels)
    if CONTRAST_MODE != "off":
        fno = ContrastComposed(fno, CONTRAST_MODE,
                               schedule_kind=CONTRAST_SCHEDULE_KIND,
                               n_bins=CONTRAST_BINS, key_mode=CONTRAST_KEY)
        rprint(f"Contrast map: {CONTRAST_MODE}/{CONTRAST_SCHEDULE_KIND} "
               f"({CONTRAST_BINS} bins), theta per LOS slice, key={CONTRAST_KEY}"
               + (f"; refit each epoch on {CONTRAST_REFIT_SAMPLES} slices"
                  f" ({'flattened' if CONTRAST_REFIT_FLATTEN else 'whole cubes'},"
                  f" batch {CONTRAST_REFIT_BATCH})"
                  if CONTRAST_REFIT else " (fixed)"))
    # Count params BEFORE the DDP wrap (DDP nests model under .module which
    # would confuse count_model_params).
    n_params = count_model_params(fno)
    model = TrainerModel(fno).to(device)
    if INIT_CHECKPOINT:
        report = load_checkpoint(model, INIT_CHECKPOINT)
        rprint(
            f"Warm-started from {INIT_CHECKPOINT}: "
            f"{report.matched}/{report.total} parameters matched "
            f"({report.transform})"
        )
    if is_distributed:
        # SyncBatchNorm: convert every BatchNormNd in the model to its
        # synchronised counterpart BEFORE the DDP wrap.  Without this, each
        # rank maintains its own BN running_mean / running_var (DDP syncs
        # gradients but not buffers); the saved checkpoint contains only
        # rank-0's stats; the 4 ranks drift during training and produce
        # spiky / inconsistent eval losses.  No-op for FNO (no BN layers);
        # critical for U-FNO whose mini U-Net path is BN-heavy.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # find_unused_parameters=False is the fast path; flip to True only if
        # DDP barks about unused params (FNO with positional_embedding="grid"
        # uses every parameter every step, so this should be fine).
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False)
    rprint(f"Model: {n_params:,} parameters")

    # -------------------------------------------- 5. optimizer / scheduler
    # LR scaling rule for the effective global batch (BATCH_SIZE * world_size).
    base_lr = {
        "fno": LEARNING_RATE,
        "ufno": UFNO_LEARNING_RATE,
        "sirenfno": SIRENFNO_LEARNING_RATE,
    }.get(MODEL_KIND, LOCALFNO_LEARNING_RATE)
    if is_distributed:
        global_bs = BATCH_SIZE * world_size
        if LR_SCALE_RULE == "linear":
            scaled_lr = base_lr * world_size
        else:  # sqrt
            scaled_lr = base_lr * (world_size ** 0.5)
    else:
        global_bs = BATCH_SIZE
        scaled_lr = base_lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=scaled_lr, weight_decay=WEIGHT_DECAY)
    grad_clip_norm = {
        "fno": 0.0,
        "ufno": UFNO_GRAD_CLIP_NORM,
        "sirenfno": SIRENFNO_GRAD_CLIP_NORM,
    }.get(MODEL_KIND, LOCALFNO_GRAD_CLIP_NORM)
    if grad_clip_norm > 0:
        def _clip_before_step(optim, args, kwargs):
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )
            # Avoid synchronizing CUDA on every batch. The epoch logger
            # converts only the final recorded norm to a Python float.
            optim._last_grad_norm = norm.detach()
            return None

        optimizer.register_step_pre_hook(_clip_before_step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=N_EPOCHS)

    # -------------------------------------------- 6. losses (3-D)
    # L2 + H1 (both absolute, d=3) are the v2/v3 baseline.  Historical note:
    # relative norms blew up on the v2 sub-cube pipeline, where an all-ionized
    # late-z chunk has ||y|| = 0; full lightcones always retain a neutral
    # high-z region, so LOSS_*_MODE=relative is safe here and makes the term
    # weights dimensionless and directly comparable.  BCE is a confidence
    # regulariser that rewards bimodal {0, 1} predictions -- see BCETerm
    # docstring.
    l2_loss = LpLoss(d=3, p=2)
    h1_loss = _build_h1_loss()
    bce_loss = BinaryCrossEntropyTerm()
    ionized_wall_loss = IonizedWallRMSE(
        band_kernel_size=IONIZED_WALL_KERNEL_SIZE,
        threshold=IONIZED_WALL_THRESHOLD,
    )
    norm_wrapper = {"absolute": AbsoluteLoss, "relative": RelativeLoss}
    l2_term = norm_wrapper[LOSS_L2_MODE](l2_loss)
    h1_term = norm_wrapper[LOSS_H1_MODE](h1_loss)
    if LOSS_LOS_VOLUME_WEIGHTS:
        los_w = los_volume_weights(dataset.target_z)
        l2_term = LOSVolumeWeightedLoss(l2_term, los_w)
        h1_term = LOSVolumeWeightedLoss(h1_term, los_w)
        print(f"[loss] LOS volume weights ON: w in "
              f"[{float(los_w.min()):.2f}, {float(los_w.max()):.2f}] "
              f"(mean 1.0, {len(los_w)} slices)")
    expwall_loss = ExponentialWallDistance(scale=EXPWALL_SCALE, cap=WALL_CAP,
                                           axes=EXPWALL_AXES)
    bsd_loss = GranulometrySpectrum(
        radii=BSD_RADII, downsample=BSD_DOWNSAMPLE, max_slices=BSD_MAX_SLICES)
    loss_terms = (
        (LOSS_L2_WEIGHT, l2_term),
        (LOSS_H1_WEIGHT, h1_term),
        (LOSS_BCE_WEIGHT, bce_loss),
        (LOSS_IONIZED_WALL_WEIGHT, ionized_wall_loss),
        (LOSS_EXPWALL_WEIGHT, expwall_loss),
        (LOSS_BSD_WEIGHT, bsd_loss),
    )
    loss_term_names = ("l2", "h1", "bce", "ionized_wall", "expwall", "bsd")
    if LOSS_BSD_WEIGHT > 0 and LOSS_L2_WEIGHT == 0 and LOSS_EXPWALL_WEIGHT == 0:
        # Measured: a field with every bubble displaced 37 px scores 21x better
        # than one with the wrong sizes. Without an anchor this run would go
        # the way of edgeonly (0.9098, never beat its epoch-0 value).
        raise SystemExit(
            "LOSS_BSD_WEIGHT needs an anchor: set LOSS_L2_WEIGHT or "
            "LOSS_EXPWALL_WEIGHT as well. A bubble-size spectrum is invariant "
            "to translation and constrains neither level nor position."
        )
    h1_warmup_epochs = {
        "fno": 0,
        "ufno": UFNO_H1_WARMUP_EPOCHS,
        "sirenfno": SIRENFNO_H1_WARMUP_EPOCHS,
    }.get(MODEL_KIND, LOCALFNO_H1_WARMUP_EPOCHS)
    # index into loss_terms: 0 l2, 1 h1, 2 bce, 3 ionized_wall, 4 expwall, 5 bsd
    warmup_terms, warmup_epochs = (), 0
    if h1_warmup_epochs > 0:
        warmup_terms, warmup_epochs = (1,), h1_warmup_epochs
    if LOSS_EXPWALL_WEIGHT > 0 and EXPWALL_WARMUP_EPOCHS > 0:
        warmup_terms = tuple(sorted(set(warmup_terms) | {4}))
        warmup_epochs = max(warmup_epochs, EXPWALL_WARMUP_EPOCHS)
    if LOSS_BSD_WEIGHT > 0 and BSD_WARMUP_EPOCHS > 0:
        # Ramped for the same reason as expwall: the anchor should establish
        # position before a morphology term starts pulling on sizes.
        warmup_terms = tuple(sorted(set(warmup_terms) | {5}))
        warmup_epochs = max(warmup_epochs, BSD_WARMUP_EPOCHS)
    if warmup_terms:
        train_loss_fn = ScheduledWeightedLoss(
            *loss_terms,
            warmup_terms=warmup_terms,
            warmup_epochs=warmup_epochs,
            term_names=loss_term_names,
        )
        print(f"[loss] warmup over {warmup_epochs} epochs for terms "
              f"{[loss_term_names[i] for i in warmup_terms]}")
    else:
        train_loss_fn = WeightedLoss(*loss_terms, term_names=loss_term_names)
    # Eval losses are tracked separately in metrics.jsonl so we can see how
    # each component evolves.  Keys here become column names in JSONL.
    # val_l2 / val_h1 stay absolute for continuity with every archived run;
    # the *_rel columns track the dimensionless variants regardless of which
    # mode the training loss uses.
    eval_losses = {
        "l2": AbsoluteLoss(l2_loss),
        "h1": AbsoluteLoss(h1_loss),
        "l2_rel": RelativeLoss(l2_loss),
        "h1_rel": RelativeLoss(h1_loss),
        "bce": bce_loss,
        "ionized_wall": ionized_wall_loss,
        "expwall": expwall_loss,
        "bsd": bsd_loss,
    }

    # -------------------------------------------- 7. trainer
    contrast = None
    if CONTRAST_REFIT and CONTRAST_MODE != "off":
        # Fit the map under the loss the run trains on, not a hardcoded MSE.
        active = [(w, term) for w, term in loss_terms if w > 0]
        contrast = ContrastRefit(
            samples=CONTRAST_REFIT_SAMPLES,
            steps=CONTRAST_REFIT_STEPS,
            batch=CONTRAST_REFIT_BATCH,
            theta_floor=CONTRAST_THETA_FLOOR,
            flatten=CONTRAST_REFIT_FLATTEN,
            objective=lambda out, y: sum(w * term(out, y) for w, term in active),
        )

    spectral_history = None
    if SPECTRAL_HISTORY_PATH is not None and is_rank_0:
        try:
            spectral_history = SpectralWeightHistory(
                SPECTRAL_HISTORY_PATH, model, reset=True)
            # The constructor stores the model without inspecting it, so an
            # architecture with no Fourier layer only raises on the first
            # extraction. Both calls must sit inside the try.
            spectral_history.record(-1)
        except ValueError as error:
            # Wavelet or Walsh-Hadamard in both slots has no mode-weight
            # profile to track. Not a reason to refuse to train.
            rprint(f"[spectral-weights] disabled: {error}")
            spectral_history = None

    trainer = MetricsTrainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=device,
        data_processor=None,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        # The model is already wrapped exactly once above. Setting this True
        # would make neuralop.Trainer add a second nested DDP wrapper.
        use_distributed=False,
        verbose=is_rank_0,                 # silence non-rank-0 Trainer prints
        metrics_path=METRICS_PATH,
        spectral_history=spectral_history,
        contrast=contrast,
        saturation_ndim=3,
        rank=rank,
        world_size=world_size,
    )

    rprint(f"\nDevice: {device}")
    rprint(f"Batch size (per rank): {BATCH_SIZE}  global: {global_bs}")
    rprint(f"LR: {scaled_lr:g}  (scaled from {base_lr:g} by {LR_SCALE_RULE} "
           f"rule for {world_size} ranks)")
    rprint(f"Epochs: {N_EPOCHS}")
    rprint(f"Model: {MODEL_CONFIG.describe()}")
    rprint(f"Run seed: {RUN_SEED}  deterministic={DETERMINISTIC_RUN}")
    rprint(f"Initial checkpoint: {INIT_CHECKPOINT or '(fresh initialization)'}")
    rprint(f"Input ablation: {INPUT_FEATURES.name}")
    rprint(f"Out: x_HI")
    l2_tag = "relL2" if LOSS_L2_MODE == "relative" else "absL2"
    h1_tag = "relH1" if LOSS_H1_MODE == "relative" else "absH1"
    rprint(f"Loss: {LOSS_L2_WEIGHT}*{l2_tag} + {LOSS_H1_WEIGHT}*{h1_tag} "
           f"+ {LOSS_BCE_WEIGHT}*BCE "
           f"+ {LOSS_IONIZED_WALL_WEIGHT}*ionized-wall-RMSE  "
           f"(H1: periodic X/Y, centered interior-only Z)")
    rprint(
        "Ionized-wall mask: "
        f"threshold={IONIZED_WALL_THRESHOLD:g}, "
        f"transverse kernel={IONIZED_WALL_KERNEL_SIZE}"
    )
    if MODEL_KIND == "ufno":
        rprint(f"UFNO stability: H1 warmup={UFNO_H1_WARMUP_EPOCHS} epochs, "
               f"gradient clip={UFNO_GRAD_CLIP_NORM:g}")
    elif MODEL_KIND == "sirenfno":
        rprint(
            "SirenFNO stability: "
            f"H1 warmup={SIRENFNO_H1_WARMUP_EPOCHS} epochs, "
            f"gradient clip={SIRENFNO_GRAD_CLIP_NORM:g}, "
            f"output sigmoid={MODEL_CONFIG.siren_output_sigmoid}, "
            f"temperature={MODEL_CONFIG.siren_sigmoid_temperature:g}"
        )
    elif MODEL_CONFIG.is_local_global:
        (local_slot, local_kwargs), (global_slot, global_kwargs) = (
            MODEL_CONFIG.operator_slots()
        )
        rprint(
            f"{MODEL_KIND} stability: "
            f"H1 warmup={LOCALFNO_H1_WARMUP_EPOCHS} epochs, "
            f"gradient clip={LOCALFNO_GRAD_CLIP_NORM:g}, "
            f"window={MODEL_CONFIG.localfno_window}, "
            f"chunk={MODEL_CONFIG.localfno_patch_chunk_size}"
        )
        rprint(
            f"Operator slots: local={local_slot}{local_kwargs or ''} "
            f"global={global_slot}{global_kwargs or ''} "
            f"windowed-local={MODEL_CONFIG.local_slot_is_windowed}"
        )
    rprint(f"DataLoader workers: {NUM_WORKERS} "
           f"(per-step log every {LOG_EVERY} batches)")
    rprint(f"Eval interval: every {EVAL_INTERVAL} epoch(s)")
    rprint(f"Metrics JSONL: {METRICS_PATH}")
    rprint(f"Spectral weight history: {SPECTRAL_HISTORY_PATH}")

    # -------------------------------------------- 8. train
    metadata = {
        "model_config": MODEL_CONFIG.to_dict(),
        "input_features": {
            "name": INPUT_FEATURES.name,
            "channel_names": list(INPUT_FEATURES.channel_names),
            "in_channels": in_channels,
        },
        "parameter_normalization": (
            parameter_normalization.to_dict()
            if parameter_normalization is not None
            else None
        ),
        "split": {
            "seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            # Cone ids are invariant to cache row ordering; downstream
            # tooling (dataset.dataset_3d.resolve_split) prefers these over the
            # row indices above.
            "train_cone_ids": [int(dataset.cone_ids[i]) for i in train_idx],
            "val_cone_ids": [int(dataset.cone_ids[i]) for i in val_idx],
            "test_cone_ids": [int(dataset.cone_ids[i]) for i in test_idx],
        },
        "training": {
            "epochs": N_EPOCHS,
            "run_seed": RUN_SEED,
            "deterministic": DETERMINISTIC_RUN,
            "init_checkpoint": INIT_CHECKPOINT,
            "base_learning_rate": base_lr,
            "scaled_learning_rate": scaled_lr,
            "lr_scale_rule": LR_SCALE_RULE,
            "loss_weights": {
                "l2": LOSS_L2_WEIGHT,
                "h1": LOSS_H1_WEIGHT,
                "bce": LOSS_BCE_WEIGHT,
                "ionized_wall": LOSS_IONIZED_WALL_WEIGHT,
                "expwall": LOSS_EXPWALL_WEIGHT,
                "bsd": LOSS_BSD_WEIGHT,
                "expwall_warmup_epochs": EXPWALL_WARMUP_EPOCHS,
            },
            "loss_modes": {
                "l2": LOSS_L2_MODE,
                "h1": LOSS_H1_MODE,
            },
            "los_volume_weights": LOSS_LOS_VOLUME_WEIGHTS,
            "ionized_wall_kernel_size": IONIZED_WALL_KERNEL_SIZE,
            "ionized_wall_threshold": IONIZED_WALL_THRESHOLD,
            "ufno_h1_warmup_epochs": (
                UFNO_H1_WARMUP_EPOCHS if MODEL_KIND == "ufno" else 0
            ),
            "ufno_grad_clip_norm": (
                UFNO_GRAD_CLIP_NORM if MODEL_KIND == "ufno" else None
            ),
            "sirenfno_h1_warmup_epochs": (
                SIRENFNO_H1_WARMUP_EPOCHS
                if MODEL_KIND == "sirenfno"
                else 0
            ),
            "sirenfno_grad_clip_norm": (
                SIRENFNO_GRAD_CLIP_NORM
                if MODEL_KIND == "sirenfno"
                else None
            ),
            "localfno_h1_warmup_epochs": (
                LOCALFNO_H1_WARMUP_EPOCHS
                if MODEL_CONFIG.is_local_global
                else 0
            ),
            "localfno_grad_clip_norm": (
                LOCALFNO_GRAD_CLIP_NORM
                if MODEL_CONFIG.is_local_global
                else None
            ),
            "best_metric_name": "val_l2",
            "best_metric": None,
            "best_epoch": None,
            "final_epoch": None,
        },
        "checkpoints": {
            "best": "best_model_state_dict.pt",
            "final": "final_model_state_dict.pt",
            "spectral_weight_history": HISTORY_FILENAME,
        },
    }
    if is_rank_0:
        write_run_metadata(CHECKPOINT_DIR, metadata)

    try:
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=train_loss_fn,
            eval_losses=eval_losses,
            save_best="val_l2",
            save_dir=CHECKPOINT_DIR,
        )
        if is_rank_0:
            save_training_state(
                save_dir=CHECKPOINT_DIR,
                save_name="final_model",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                regularizer=None,
                epoch=N_EPOCHS - 1,
            )
            metadata["training"].update(
                best_metric=trainer.best_metric,
                best_epoch=trainer.best_epoch,
                final_epoch=N_EPOCHS - 1,
            )
            write_run_metadata(CHECKPOINT_DIR, metadata)
            rprint(
                f"Saved best checkpoint from epoch {trainer.best_epoch} "
                f"(val_l2={trainer.best_metric}) and final epoch {N_EPOCHS - 1}"
            )
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
