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

from dataset.dataset_3d import (
    InputFeatures,
    LightconeCubeDataset,
    LightconeCubeCache,
    split_cubes,
)
from losses import (
    AbsoluteLoss,
    BinaryCrossEntropyTerm,
    LightconeH1Loss,
    ScheduledWeightedLoss,
    WeightedLoss,
)
from modeling import ModelConfig, TrainerModel, build_3d_model
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
CUBES_CACHE = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))

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
WEIGHT_DECAY = 1e-5
# N_EPOCHS overridable from the sbatch (U-FNO defaults to a shorter first run).
N_EPOCHS = int(os.environ.get("N_EPOCHS", "100"))

# Loss term weights.  Defaults (0.5, 0.5, 0.0) match the v1 run.  All three
# are env-var overridable so experiments don't need to edit this file:
#   * (0.5, 0.5, 0.0)  v1 U-FNO baseline
#   * (0.5, 0.5, 0.5)  BCE-on experiment (Act 4)
#   * (0.3, 0.7, 0.0)  v2 H1-weighted (C of the v2 A+B+C bundle)
LOSS_L2_WEIGHT = float(os.environ.get("LOSS_L2_WEIGHT", "0.5"))
LOSS_H1_WEIGHT = float(os.environ.get("LOSS_H1_WEIGHT", "0.5"))
LOSS_BCE_WEIGHT = float(os.environ.get("LOSS_BCE_WEIGHT", "0.0"))

# Stability controls for U-FNO. H1 starts at zero and ramps linearly to its
# configured weight, allowing the L2 value term to establish a non-saturated
# output before derivative matching begins.
UFNO_H1_WARMUP_EPOCHS = int(os.environ.get("UFNO_H1_WARMUP_EPOCHS", "5"))
UFNO_GRAD_CLIP_NORM = float(os.environ.get("UFNO_GRAD_CLIP_NORM", "1.0"))

# DataLoader workers.  Streamed loading (one ~370 MB HDF5 read per sample) is
# the throughput bottleneck on cluster filesystems; parallelizing across the
# allocated CPUs gets the GPU fed.  Defaults to SLURM_CPUS_PER_TASK on the
# cluster and 0 locally.
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))

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

# Learning-rate scaling rule for multi-GPU DDP runs.  "sqrt" is conservative
# and rarely diverges; "linear" extracts more wall-clock speed but may need
# warmup at large effective batches.
LR_SCALE_RULE = "sqrt"   # "linear" or "sqrt"


# ------------------------------------------------------------------ reproducibility
def _seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed model initialization and training-time random number generators."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker a reproducible Python/NumPy RNG state."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ------------------------------------------------------------------ distributed
def _setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched under multi-task SLURM.

    Returns ``(rank, local_rank, world_size)``.  Single-process runs return
    ``(0, 0, 1)`` and do not call ``init_process_group``.

    Conventions:
      * Multi-GPU is detected via ``SLURM_NTASKS`` > 1.  Launch with
        ``srun --ntasks=4 python fno_21cm_3d.py`` from inside the sbatch script.
      * The master address is taken from the first hostname in
        ``SLURM_NODELIST``; single-node multi-GPU is the only configuration
        tested.  Multi-node would need a more careful parse of the nodelist
        (Slurm range notation like ``gpu[01-04]``).
      * NCCL backend is used unconditionally -- it's the only backend that
        actually works for multi-GPU on NVIDIA hardware.
    """
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if world_size <= 1:
        return 0, 0, 1

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    nodelist = os.environ.get("SLURM_NODELIST", "localhost")
    # Single-node case: nodelist is just one hostname.  If we ever go
    # multi-node, this will need slurm range-expansion handling.
    master_addr = nodelist.split(",")[0]
    if "[" in master_addr:
        # Range notation like gpu[01-04] -- bail rather than guess
        master_addr = "localhost"
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def _all_reduce_mean(value: float, world_size: int) -> float:
    """Average a scalar across all DDP ranks (returns the input if not DDP)."""
    if world_size <= 1 or not dist.is_initialized():
        return float(value)
    t = torch.tensor([float(value)], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


def _all_reduce_weighted_metrics(
    metrics: dict[str, float | torch.Tensor],
    local_sample_count: int,
    world_size: int,
    device: str | torch.device,
) -> dict[str, float]:
    """Combine per-rank metric means using their local sample counts.

    ``neuralop.Trainer.evaluate`` returns a mean over the samples handled by
    the current rank. Reconstruct each local sum, reduce sums and counts across
    ranks, then divide once so uneven rank-local shard sizes remain correct.
    """
    if local_sample_count < 0:
        raise ValueError(
            f"local_sample_count must be non-negative, got {local_sample_count}"
        )

    keys = list(metrics)
    if world_size <= 1 or not dist.is_initialized():
        return {key: float(metrics[key]) for key in keys}

    local_count = float(local_sample_count)
    packed = [float(metrics[key]) * local_count for key in keys]
    packed.append(local_count)
    reduced = torch.tensor(packed, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    global_count = float(reduced[-1].item())
    if global_count <= 0:
        raise RuntimeError("distributed evaluation processed zero samples")

    return {
        key: float(reduced[i].item() / global_count)
        for i, key in enumerate(keys)
    }


# How often the Trainer runs val/test evaluation AND prints the per-epoch
# metrics line.  Set to 1 to see train+val+test losses every epoch (adds the
# eval pass to each epoch's wall clock).  Bump to 5 once training has settled
# if you want to reduce the eval-loop overhead.
EVAL_INTERVAL = 1


def _build_h1_loss() -> LightconeH1Loss:
    """H1 objective with periodic X/Y and centered interior LOS differences."""
    return LightconeH1Loss(measure=(1.0, 1.0, 1.0))


class LoggingTrainer(Trainer):
    """``neuralop.Trainer`` + per-epoch metrics written to a JSONL file.

    The base ``Trainer`` prints metrics to stdout and returns the final
    ``epoch_metrics`` dict, but never writes intermediate metrics to disk.
    On a long run that's risky -- SLURM rotates logs, and there's nothing
    to plot from afterward.  This subclass intercepts ``train_one_epoch``
    and ``evaluate_all`` to append one JSON object per epoch to
    ``metrics_path``.  Each line has at minimum ``epoch``, ``train_err``,
    ``avg_loss``, ``epoch_train_time``, plus any eval-loop metrics from
    that epoch (e.g. ``val_l2``, ``test_h1``).

    Read it back with ``pandas.read_json(path, lines=True)``.
    """

    def __init__(
        self,
        *args,
        metrics_path: str | Path | None = None,
        spectral_history_path: str | Path | None = None,
        rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ):
        if kwargs.get("use_distributed", False):
            raise ValueError(
                "LoggingTrainer must not wrap DDP itself; fno_21cm_3d.py "
                "constructs the single DDP wrapper before the optimizer."
            )
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        # File and directory side effects happen on rank 0 only -- otherwise
        # all four ranks race to create / append to the same file.
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._is_rank_0 = (self._rank == 0)
        if self.metrics_path is not None and self._is_rank_0:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            # This entry point does not resume runs. Avoid mixing a fresh
            # trajectory with stale rows from an older checkpoint directory.
            self.metrics_path.unlink(missing_ok=True)
        self._last_train: dict | None = None
        self.best_epoch: int | None = None
        self.best_metric: float | None = None
        self.spectral_history = None
        if spectral_history_path is not None and self._is_rank_0:
            self.spectral_history = SpectralWeightHistory(
                spectral_history_path,
                self.model,
                reset=True,
            )
            self.spectral_history.record(-1)

    def train_one_epoch(self, epoch, train_loader, training_loss):
        # DistributedSampler must be told the epoch so it reshuffles
        # consistently across ranks each epoch.
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(int(epoch))
        if hasattr(training_loss, "set_epoch"):
            training_loss.set_epoch(int(epoch))

        out = super().train_one_epoch(epoch, train_loader, training_loss)
        train_err, avg_loss, avg_lasso, t = out

        # Under DDP each rank sees only its shard of the train set, so the
        # per-rank train_err / avg_loss are partial.  Average across ranks
        # to get a globally meaningful number for the JSONL log.
        train_err_global = _all_reduce_mean(train_err, self._world_size)
        avg_loss_global = _all_reduce_mean(avg_loss, self._world_size)
        avg_lasso_global = (_all_reduce_mean(avg_lasso, self._world_size)
                            if avg_lasso is not None else 0.0)

        self._last_train = dict(
            epoch=int(epoch),
            train_err=float(train_err_global),
            avg_loss=float(avg_loss_global),
            avg_lasso_loss=float(avg_lasso_global),
            epoch_train_time=float(t),
        )
        if hasattr(training_loss, "active_weights"):
            active = training_loss.active_weights
            self._last_train.update(
                active_l2_weight=float(active[0]),
                active_h1_weight=float(active[1]),
                active_bce_weight=float(active[2]),
            )
        grad_norm = getattr(self.optimizer, "_last_grad_norm", None)
        if grad_norm is not None:
            self._last_train["last_grad_norm"] = float(grad_norm)
        if self.spectral_history is not None:
            self.spectral_history.record(int(epoch))
        if self.eval_interval and (epoch % self.eval_interval != 0):
            self._flush_row({})
        return out

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        losses, output = super().eval_one_batch(
            sample, eval_losses, return_output=True
        )
        # A regular 8-cell stride samples ~0.2% of the cube, which is ample
        # for detecting all-zero/all-one collapse without adding several
        # full-volume float64 reductions to every evaluation batch.
        sampled = output.detach()[..., ::8, ::8, ::8]
        self._pred_sum += sampled.sum()
        self._pred_sq_sum += sampled.square().sum()
        self._pred_low_count += (sampled <= 1e-4).sum()
        self._pred_high_count += (sampled >= 1.0 - 1e-4).sum()
        self._pred_count += sampled.numel()
        return losses, output if return_output else None

    def evaluate(self, *args, **kwargs):
        log_prefix = str(kwargs.get("log_prefix", "")).strip()
        metric_prefix = f"{log_prefix}_" if log_prefix else ""
        self._pred_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._pred_sq_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._pred_low_count = torch.zeros(
            (), dtype=torch.float64, device=self.device
        )
        self._pred_high_count = torch.zeros(
            (), dtype=torch.float64, device=self.device
        )
        self._pred_count = 0
        local_metrics = super().evaluate(*args, **kwargs)
        count = max(int(self._pred_count), 1)
        pred_mean = self._pred_sum / count
        pred_variance = (self._pred_sq_sum / count - pred_mean.square()).clamp_min(0)
        local_metrics.update(
            **{
                f"{metric_prefix}pred_mean": float(pred_mean.item()),
                f"{metric_prefix}pred_std": float(pred_variance.sqrt().item()),
                f"{metric_prefix}pred_sat_low": float(
                    (self._pred_low_count / count).item()
                ),
                f"{metric_prefix}pred_sat_high": float(
                    (self._pred_high_count / count).item()
                ),
            }
        )
        return _all_reduce_weighted_metrics(
            local_metrics,
            local_sample_count=self.n_samples,
            world_size=self._world_size,
            device=self.device,
        )

    def evaluate_all(self, *args, **kwargs):
        eval_metrics = super().evaluate_all(*args, **kwargs)
        # evaluate() has already reduced every loader's metrics across ranks.
        clean = {k: float(v) for k, v in eval_metrics.items()}
        monitored = clean.get("val_l2")
        if monitored is not None and (
            self.best_metric is None or monitored < self.best_metric
        ):
            self.best_metric = monitored
            self.best_epoch = int(kwargs.get("epoch", -1))
        self._flush_row(clean)
        return eval_metrics

    def _flush_row(self, eval_metrics: dict) -> None:
        if not self._is_rank_0:
            return
        if self.metrics_path is None or self._last_train is None:
            return
        import json
        row = {**self._last_train, **eval_metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")


class ProgressLoader:
    """Wrap a DataLoader to print throughput every ``log_every`` steps.

    The neuralop Trainer only logs per-epoch summaries, so on long epochs
    (5k+ samples) you get no signal at all until the first epoch completes.
    This wrapper preserves the DataLoader interface (length + iter) and
    prints ``[step k/N] r samples/s, ETA M:SS`` lines so the SLURM log
    shows life.
    """

    def __init__(self, loader, log_every: int = 25, tag: str = "train",
                 rank: int = 0):
        self.loader = loader
        self.log_every = int(log_every)
        self.tag = tag
        # Only rank 0 prints; other ranks iterate silently.
        self._is_rank_0 = (int(rank) == 0)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        # Delegate anything not on the wrapper (e.g. .dataset, .sampler,
        # .batch_size) to the underlying DataLoader.  The neuralop Trainer
        # reads train_loader.dataset for its startup banner.
        return getattr(self.loader, name)

    def __iter__(self):
        import time
        n = len(self.loader)
        # batch_size is set on the underlying DataLoader; default to 1 if the
        # wrapped loader is something exotic that doesn't expose it.
        bs = int(getattr(self.loader, "batch_size", 1) or 1)
        t0 = time.time()
        for i, batch in enumerate(self.loader, start=1):
            yield batch
            if (self._is_rank_0 and self.log_every
                    and (i % self.log_every == 0 or i == n)):
                elapsed = time.time() - t0
                batches_per_s = i / max(elapsed, 1e-6)
                samples_per_s = batches_per_s * bs
                eta = (n - i) / max(batches_per_s, 1e-6)
                print(f"    [{self.tag} {i}/{n}] "
                      f"{samples_per_s:.2f} samples/s "
                      f"({batches_per_s:.2f} batches/s, bs={bs})  "
                      f"elapsed {elapsed:6.1f}s  ETA {eta/60:5.1f} min",
                      flush=True)


# ------------------------------------------------------------------ main
def main():
    # -------------------------------------------- 0. distributed setup
    rank, local_rank, world_size = _setup_distributed()
    is_rank_0 = (rank == 0)
    is_distributed = (world_size > 1)
    # All ranks use the same initialization seed so DDP starts from identical
    # parameters. DistributedSampler applies rank-specific partitioning.
    _seed_everything(RUN_SEED, deterministic=DETERMINISTIC_RUN)

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

    fno = build_3d_model(MODEL_CONFIG, in_channels)
    # Count params BEFORE the DDP wrap (DDP nests model under .module which
    # would confuse count_model_params).
    n_params = count_model_params(fno)
    model = TrainerModel(fno).to(device)
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
    base_lr = (
        UFNO_LEARNING_RATE if MODEL_KIND == "ufno" else LEARNING_RATE
    )
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
    if MODEL_KIND == "ufno" and UFNO_GRAD_CLIP_NORM > 0:
        def _clip_before_step(optim, args, kwargs):
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=UFNO_GRAD_CLIP_NORM
            )
            # Avoid synchronizing CUDA on every batch. The epoch logger
            # converts only the final recorded norm to a Python float.
            optim._last_grad_norm = norm.detach()
            return None

        optimizer.register_step_pre_hook(_clip_before_step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=N_EPOCHS)

    # -------------------------------------------- 6. losses (3-D)
    # L2 + H1 (both absolute, d=3) are the v2/v3 baseline.  Relative norms
    # blow up over the all-ionized late-z portion of the cube where x_HI = 0,
    # so absolute is mandatory here.  BCE is a confidence regulariser that
    # rewards bimodal {0, 1} predictions -- see BCETerm docstring.
    l2_loss = LpLoss(d=3, p=2)
    h1_loss = _build_h1_loss()
    bce_loss = BinaryCrossEntropyTerm()
    loss_terms = (
        (LOSS_L2_WEIGHT, AbsoluteLoss(l2_loss)),
        (LOSS_H1_WEIGHT, AbsoluteLoss(h1_loss)),
        (LOSS_BCE_WEIGHT, bce_loss),
    )
    if MODEL_KIND == "ufno":
        train_loss_fn = ScheduledWeightedLoss(
            *loss_terms,
            warmup_terms=(1,),
            warmup_epochs=UFNO_H1_WARMUP_EPOCHS,
        )
    else:
        train_loss_fn = WeightedLoss(*loss_terms)
    # Eval losses are tracked separately in metrics.jsonl so we can see how
    # each component evolves.  Keys here become column names in JSONL.
    eval_losses = {
        "l2": AbsoluteLoss(l2_loss),
        "h1": AbsoluteLoss(h1_loss),
        "bce": bce_loss,
    }

    # -------------------------------------------- 7. trainer
    trainer = LoggingTrainer(
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
        spectral_history_path=SPECTRAL_HISTORY_PATH,
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
    rprint(f"Input ablation: {INPUT_FEATURES.name}")
    rprint(f"Out: x_HI")
    rprint(f"Loss: {LOSS_L2_WEIGHT}*absL2 + {LOSS_H1_WEIGHT}*absH1 "
           f"+ {LOSS_BCE_WEIGHT}*BCE  "
           f"(H1: periodic X/Y, centered interior-only Z)")
    if MODEL_KIND == "ufno":
        rprint(f"UFNO stability: H1 warmup={UFNO_H1_WARMUP_EPOCHS} epochs, "
               f"gradient clip={UFNO_GRAD_CLIP_NORM:g}")
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
            "base_learning_rate": base_lr,
            "scaled_learning_rate": scaled_lr,
            "lr_scale_rule": LR_SCALE_RULE,
            "ufno_h1_warmup_epochs": (
                UFNO_H1_WARMUP_EPOCHS if MODEL_KIND == "ufno" else 0
            ),
            "ufno_grad_clip_norm": (
                UFNO_GRAD_CLIP_NORM if MODEL_KIND == "ufno" else None
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
