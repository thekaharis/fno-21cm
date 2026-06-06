#!/usr/bin/env python3
"""Train a 3-D Fourier Neural Operator on full 21cm lightcone cubes.

Mapping:  matter density cube  ->  neutral fraction (x_HI) cube.

Each lightcone is interpolated along the LOS axis to a fixed n_z grid so the
whole cube fits in a single forward pass on an A30 (24 GB) at batch=1.  The
input tensor carries the density (normalized by a fixed constant) and an
explicit ``1/(1+z)`` channel.  The FNO's ``positional_embedding="grid"`` option
then appends normalized (x, y, z) coordinates as additional channels; on this
grid the z-coordinate IS the normalized comoving distance because the native
LOS cells are uniform in comoving distance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- Prefer a vendored neuralop checkout if one is available ---------------
# Search order:
#   ./neuraloperator/         (checkout vendored inside the repo)
#   ../neuraloperator/        (checkout sibling to the repo: project/{data,
#                              neuraloperator, fno-21cm} layout)
#   ./                        (neuralop dropped straight into the repo)
# If none has a valid __init__.py, fall back to an installed `neuralop`.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "neuraloperator",
              _HERE.parent / "neuraloperator",
              _HERE):
    if (_cand / "neuralop" / "__init__.py").is_file():
        sys.path.insert(0, str(_cand))
        break
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from neuralop.models import FNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

import neuralop as _neuralop
print(f"[fno_21cm_3d] using neuralop from {_neuralop.__file__}")

from dataset_3d import LightconeCubeDataset, LightconeCubeCache, split_cubes


# ------------------------------------------------------------------ config
# Lightcone directory (raw .h5 files): env var LIGHTCONE_DIR overrides; falls
# back to ./data so the SLURM sbatch can set the cluster path once without
# editing this file.
DATA_DIR = Path(os.environ.get("LIGHTCONE_DIR", "data"))
FILE_GLOB = "21cmfast_11d_sample*.h5"

# Pre-computed cube cache (built by build_cubes.py).  Env var CUBES_CACHE
# overrides; default is ./cubes_3d.h5 next to the script.  If the cache file
# exists at startup, the training script uses LightconeCubeCache (fast,
# pre-interpolated cubes); otherwise it falls back to LightconeCubeDataset
# (raw streaming, ~10x slower per epoch).
CUBES_CACHE = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))

N_Z = 256                           # LOS resolution after interpolation
Z_MIN, Z_MAX = 5.0, 25.0

# Which model to train.  "fno" = neuralop FNO (the v3 baseline).
# "ufno" = the Wen et al. U-FNO via models_ufno.UFNOWrapped (3 FNO blocks
# + 3 U-Fourier blocks with a mini 3-D U-Net path inside each).  Override
# at submit time with MODEL_KIND=ufno in the sbatch.
MODEL_KIND = os.environ.get("MODEL_KIND", "fno").lower()
if MODEL_KIND not in ("fno", "ufno"):
    raise ValueError(f"MODEL_KIND must be 'fno' or 'ufno', got {MODEL_KIND!r}")

N_MODES = (16, 16, 16)
HIDDEN_CHANNELS = 32                # neuralop FNO -- ignored for U-FNO
UFNO_WIDTH = 32                     # U-FNO body width (analog of hidden_channels)
N_LAYERS = 4                        # neuralop FNO only; U-FNO is fixed at 3+3 blocks
BATCH_SIZE = 1                      # 3-D cubes are heavy; raise after profiling
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
# N_EPOCHS overridable from the sbatch (U-FNO defaults to a shorter first run).
N_EPOCHS = int(os.environ.get("N_EPOCHS", "100"))

# Loss term weights.  L2 + H1 at (0.5, 0.5) is the v2/v3 baseline.  BCE adds
# bimodal pressure but the BCE-at-0.5 experiment plateaued at the same floor
# as L2+H1 only and slightly regressed at the hardest z, so it defaults off.
# Set LOSS_BCE_WEIGHT > 0 to re-enable.
LOSS_L2_WEIGHT = 0.5
LOSS_H1_WEIGHT = 0.5
LOSS_BCE_WEIGHT = 0.0

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

SPLIT_SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# Separate checkpoint directories per model so a U-FNO run never overwrites
# the FNO baseline (or vice versa).  Override CHECKPOINT_DIR explicitly in
# the env for a one-off custom run.
_DEFAULT_CKPT = "./checkpoints_3d" if MODEL_KIND == "fno" else "./checkpoints_3d_ufno"
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _DEFAULT_CKPT)
METRICS_PATH = f"{CHECKPOINT_DIR}/metrics.jsonl"

# Learning-rate scaling rule for multi-GPU DDP runs.  "sqrt" is conservative
# and rarely diverges; "linear" extracts more wall-clock speed but may need
# warmup at large effective batches.
LR_SCALE_RULE = "sqrt"   # "linear" or "sqrt"


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

# How often the Trainer runs val/test evaluation AND prints the per-epoch
# metrics line.  Set to 1 to see train+val+test losses every epoch (adds the
# eval pass to each epoch's wall clock).  Bump to 5 once training has settled
# if you want to reduce the eval-loop overhead.
EVAL_INTERVAL = 1


# ------------------------------------------------------------------ wrappers
class AbsLoss:
    """Wrap a neuralop loss to call ``.abs()`` and swallow Trainer kwargs."""

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **kwargs):
        return self.loss.abs(out, y)


class WeightedSumLoss:
    """Weighted sum of ``(weight, loss)`` terms; passes kwargs through."""

    def __init__(self, *terms):
        self.terms = terms

    def __call__(self, out, y, **kwargs):
        return sum(w * loss(out, y, **kwargs) for w, loss in self.terms)


class BCETerm:
    """Voxel-mean binary cross-entropy between predicted x_HI and {0, 1} truth.

    Why this exists
    ---------------
    x_HI is essentially binary at the voxel level -- either neutral (1) or
    ionized (0), with narrow transition regions at bubble walls.  L2 + H1
    alone reward the optimizer for hedging on uncertain voxels (predicting
    ~0.3 when the truth could be 0 or 1 minimises L2), which leaves bubble
    interiors too bright and walls too soft.  BCE penalises hedging
    explicitly: ``-log(p)`` blows up as ``p -> 0`` when truth is 1, so the
    model is pushed toward confident predictions.

    Predictions are clamped to ``(eps, 1 - eps)`` to avoid ``log(0)``.

    Mirrors the ``AbsLoss`` interface: callable with ``(out, y, **kwargs)``
    so the neuralop Trainer can use it directly in the training and eval
    loops.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, out, y, **kwargs):
        p = out.clamp(self.eps, 1.0 - self.eps)
        return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).mean()


class SilentFNO(nn.Module):
    """Discard extra kwargs the Trainer injects (``y``, etc.).

    Identical to the 2-D wrapper -- ``nn.Module`` attribute lookup falls
    through to the underlying FNO via ``__getattr__``.
    """

    def __init__(self, fno: FNO):
        super().__init__()
        self.fno = fno

    def forward(self, x, **kwargs):
        return self.fno(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        return getattr(self._modules["fno"], name)


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

    def __init__(self, *args, metrics_path: str | Path | None = None,
                 rank: int = 0, world_size: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        # File and directory side effects happen on rank 0 only -- otherwise
        # all four ranks race to create / append to the same file.
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._is_rank_0 = (self._rank == 0)
        if self.metrics_path is not None and self._is_rank_0:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_train: dict | None = None

    def train_one_epoch(self, epoch, train_loader, training_loss):
        # DistributedSampler must be told the epoch so it reshuffles
        # consistently across ranks each epoch.
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(int(epoch))

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
        if self.eval_interval and (epoch % self.eval_interval != 0):
            self._flush_row({})
        return out

    def evaluate_all(self, *args, **kwargs):
        eval_metrics = super().evaluate_all(*args, **kwargs)
        # Force floats for clean JSON; eval values are tensors or numpy scalars.
        # The neuralop Trainer with use_distributed=True is expected to
        # all-reduce eval metrics internally; if it doesn't, the values logged
        # below will be per-rank-0 partial means -- still informative as a
        # trend, just biased.
        clean = {k: float(v) for k, v in eval_metrics.items()}
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
        dataset = LightconeCubeCache(CUBES_CACHE)
        rprint(f"Dataset: {len(dataset)} cubes  "
               f"({dataset.n_x} x {dataset.n_y} x {dataset.n_z}, "
               f"z in [{dataset.target_z[0]:.2f}, {dataset.target_z[-1]:.2f}])")
    else:
        rprint(f"No cube cache at {CUBES_CACHE}; streaming raw lightcones "
               f"from {DATA_DIR}. Run build_cubes.py to precompute and speed "
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
        )
        rprint(f"Dataset: {len(dataset)} cubes  ({N_Z} LOS cells each, "
               f"z in [{Z_MIN}, {Z_MAX}])")

    # -------------------------------------------- 2. split by cone
    train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx) = split_cubes(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED,
    )
    overlap = set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) \
              | set(val_idx) & set(test_idx)
    assert not overlap, f"Split leakage: {overlap}"
    rprint(f"Train: {len(train_ds)} cones {train_idx}")
    rprint(f"Val:   {len(val_ds)} cones {val_idx}")
    rprint(f"Test:  {len(test_ds)} cones {test_idx}")

    # -------------------------------------------- 3. dataloaders
    # Under DDP each rank consumes a disjoint shard of each split.  The
    # DistributedSampler pads to be divisible by world_size if needed.
    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
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
    )
    train_loader = DataLoader(train_ds, shuffle=train_shuffle,
                              sampler=train_sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False,
                            sampler=val_sampler, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False,
                             sampler=test_sampler, **dl_kwargs)

    # Wrap the train loader in a per-step progress reporter so the SLURM log
    # shows life within an epoch (the neuralop Trainer logs per-epoch only).
    train_loader = ProgressLoader(train_loader, log_every=LOG_EVERY,
                                  tag="train", rank=rank)
    test_loaders = {"val": val_loader, "test": test_loader}

    # -------------------------------------------- 4. model + DDP wrap
    # in_channels read from the dataset so it auto-adjusts when parameter
    # conditioning is enabled (2 -> 13 with the 11 LHS params broadcast in).
    in_channels = getattr(dataset, "in_channels", 2)
    rprint(f"Input channels: {in_channels} "
           f"(density + 1/(1+z)"
           + (f" + {in_channels - 2} astrophysical params" if in_channels > 2 else "")
           + ")")

    if MODEL_KIND == "ufno":
        # Wen et al. U-FNO: 3 FNO blocks + 3 U-Fourier blocks (U-Net inside).
        # The U-FNO body has no positional embedding -- its mini U-Net path
        # provides the local spatial inductive bias instead.
        from models_ufno import UFNOWrapped
        fno = UFNOWrapped(
            modes1=N_MODES[0], modes2=N_MODES[1], modes3=N_MODES[2],
            width=UFNO_WIDTH,
            in_channels=in_channels,
            out_channels=1,
            sigmoid=True,               # bound predictions to [0, 1] for x_HI
        )
    else:
        fno = FNO(
            n_modes=N_MODES,
            hidden_channels=HIDDEN_CHANNELS,
            in_channels=in_channels,
            out_channels=1,
            n_layers=N_LAYERS,
            projection_channel_ratio=2,
            positional_embedding="grid",
        )
    # Count params BEFORE the DDP wrap (DDP nests model under .module which
    # would confuse count_model_params).
    n_params = count_model_params(fno)
    model = SilentFNO(fno).to(device)
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
    if is_distributed:
        global_bs = BATCH_SIZE * world_size
        if LR_SCALE_RULE == "linear":
            scaled_lr = LEARNING_RATE * world_size
        else:  # sqrt
            scaled_lr = LEARNING_RATE * (world_size ** 0.5)
    else:
        global_bs = BATCH_SIZE
        scaled_lr = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=scaled_lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=N_EPOCHS)

    # -------------------------------------------- 6. losses (3-D)
    # L2 + H1 (both absolute, d=3) are the v2/v3 baseline.  Relative norms
    # blow up over the all-ionized late-z portion of the cube where x_HI = 0,
    # so absolute is mandatory here.  BCE is a confidence regulariser that
    # rewards bimodal {0, 1} predictions -- see BCETerm docstring.
    l2_loss = LpLoss(d=3, p=2)
    h1_loss = H1Loss(d=3)
    bce_loss = BCETerm()
    train_loss_fn = WeightedSumLoss(
        (LOSS_L2_WEIGHT, AbsLoss(l2_loss)),
        (LOSS_H1_WEIGHT, AbsLoss(h1_loss)),
        (LOSS_BCE_WEIGHT, bce_loss),
    )
    # Eval losses are tracked separately in metrics.jsonl so we can see how
    # each component evolves.  Keys here become column names in JSONL.
    eval_losses = {
        "l2": AbsLoss(l2_loss),
        "h1": AbsLoss(h1_loss),
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
        use_distributed=is_distributed,
        verbose=is_rank_0,                 # silence non-rank-0 Trainer prints
        metrics_path=METRICS_PATH,
        rank=rank,
        world_size=world_size,
    )

    rprint(f"\nDevice: {device}")
    rprint(f"Batch size (per rank): {BATCH_SIZE}  global: {global_bs}")
    rprint(f"LR: {scaled_lr:g}  (scaled from {LEARNING_RATE:g} by {LR_SCALE_RULE} "
           f"rule for {world_size} ranks)")
    rprint(f"Epochs: {N_EPOCHS}")
    if MODEL_KIND == "ufno":
        rprint(f"Model: U-FNO (3 FNO + 3 U-Fourier blocks)  "
               f"modes={N_MODES}  width={UFNO_WIDTH}  sigmoid-output")
    else:
        rprint(f"Model: FNO  modes={N_MODES}  hidden={HIDDEN_CHANNELS}  "
               f"layers={N_LAYERS}  pos-emb=grid")
    rprint(f"Out: x_HI")
    rprint(f"Loss: {LOSS_L2_WEIGHT}*absL2 + {LOSS_H1_WEIGHT}*absH1 "
           f"+ {LOSS_BCE_WEIGHT}*BCE  (d=3)")
    rprint(f"DataLoader workers: {NUM_WORKERS} "
           f"(per-step log every {LOG_EVERY} batches)")
    rprint(f"Eval interval: every {EVAL_INTERVAL} epoch(s)")
    rprint(f"Metrics JSONL: {METRICS_PATH}")

    # -------------------------------------------- 8. train
    try:
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=train_loss_fn,
            eval_losses=eval_losses,
            save_every=10,
            save_dir=CHECKPOINT_DIR,
        )
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
