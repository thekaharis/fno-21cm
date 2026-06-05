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
from torch.utils.data import DataLoader

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

N_MODES = (16, 16, 16)
HIDDEN_CHANNELS = 32
N_LAYERS = 4
BATCH_SIZE = 1                      # 3-D cubes are heavy; raise after profiling
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100

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

CHECKPOINT_DIR = "./checkpoints_3d"
METRICS_PATH = f"{CHECKPOINT_DIR}/metrics.jsonl"

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

    def __init__(self, *args, metrics_path: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_train: dict | None = None

    def train_one_epoch(self, epoch, train_loader, training_loss):
        out = super().train_one_epoch(epoch, train_loader, training_loss)
        train_err, avg_loss, avg_lasso, t = out
        self._last_train = dict(
            epoch=int(epoch),
            train_err=float(train_err),
            avg_loss=float(avg_loss),
            # avg_lasso is None when no regularizer is configured.
            avg_lasso_loss=(float(avg_lasso) if avg_lasso is not None else 0.0),
            epoch_train_time=float(t),
        )
        # If this epoch will NOT call evaluate_all, write the row now so we
        # still capture train metrics on non-eval epochs.
        if self.eval_interval and (epoch % self.eval_interval != 0):
            self._flush_row({})
        return out

    def evaluate_all(self, *args, **kwargs):
        eval_metrics = super().evaluate_all(*args, **kwargs)
        # Force floats for clean JSON; eval values are tensors or numpy scalars.
        clean = {k: float(v) for k, v in eval_metrics.items()}
        self._flush_row(clean)
        return eval_metrics

    def _flush_row(self, eval_metrics: dict) -> None:
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

    def __init__(self, loader, log_every: int = 25, tag: str = "train"):
        self.loader = loader
        self.log_every = int(log_every)
        self.tag = tag

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
        t0 = time.time()
        for i, batch in enumerate(self.loader, start=1):
            yield batch
            if self.log_every and (i % self.log_every == 0 or i == n):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (n - i) / max(rate, 1e-6)
                print(f"    [{self.tag} {i}/{n}] {rate:.2f} samples/s  "
                      f"elapsed {elapsed:6.1f}s  ETA {eta/60:5.1f} min",
                      flush=True)


# ------------------------------------------------------------------ main
def main():
    # -------------------------------------------- 1. dataset (cache or stream)
    # Prefer the pre-built cube cache when it exists: ~10x faster reads per
    # sample, no scipy interpolation in the hot path.
    if CUBES_CACHE.exists():
        print(f"Using pre-computed cube cache: {CUBES_CACHE}")
        dataset = LightconeCubeCache(CUBES_CACHE)
        print(f"Dataset: {len(dataset)} cubes  "
              f"({dataset.n_x} x {dataset.n_y} x {dataset.n_z}, "
              f"z in [{dataset.target_z[0]:.2f}, {dataset.target_z[-1]:.2f}])")
    else:
        print(f"No cube cache at {CUBES_CACHE}; streaming raw lightcones "
              f"from {DATA_DIR}. Run build_cubes.py to precompute and speed "
              f"up training ~10x.")
        files = sorted(DATA_DIR.glob(FILE_GLOB))
        if not files:
            print(f"No lightcone files found under {DATA_DIR}/{FILE_GLOB}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(files)} lightcone files in {DATA_DIR}")
        dataset = LightconeCubeDataset(
            file_paths=files,
            n_z=N_Z, z_min=Z_MIN, z_max=Z_MAX,
            preload=False,
        )
        print(f"Dataset: {len(dataset)} cubes  ({N_Z} LOS cells each, "
              f"z in [{Z_MIN}, {Z_MAX}])")

    # -------------------------------------------- 3. split by cone
    train_ds, val_ds, test_ds, (train_idx, val_idx, test_idx) = split_cubes(
        dataset, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED,
    )
    overlap = set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) \
              | set(val_idx) & set(test_idx)
    assert not overlap, f"Split leakage: {overlap}"
    print(f"Train: {len(train_ds)} cones {train_idx}")
    print(f"Val:   {len(val_ds)} cones {val_idx}")
    print(f"Test:  {len(test_ds)} cones {test_idx}")

    # -------------------------------------------- 4. dataloaders
    # persistent_workers keeps workers alive across epochs (avoids paying the
    # fork + scipy import startup cost on every epoch).  Only meaningful when
    # num_workers > 0.
    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE != "cpu"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # Wrap the train loader in a per-step progress reporter so the SLURM log
    # shows life within an epoch (the neuralop Trainer logs per-epoch only).
    train_loader = ProgressLoader(train_loader, log_every=LOG_EVERY,
                                  tag="train")
    test_loaders = {"val": val_loader, "test": test_loader}

    # -------------------------------------------- 5. model
    fno = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=2,                  # density + 1/(1+z)
        out_channels=1,                 # x_HI
        n_layers=N_LAYERS,
        projection_channel_ratio=2,
        positional_embedding="grid",    # injects normalized (x, y, z) coords
    )
    model = SilentFNO(fno).to(DEVICE)
    print(f"Model: {count_model_params(model.fno):,} parameters")
    print(model)

    # -------------------------------------------- 6. optimizer / scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=N_EPOCHS)

    # -------------------------------------------- 7. losses (3-D)
    # Same recipe as v2: absolute L2 + absolute H1, both d=3.  Relative norms
    # blow up over the all-ionized late-z portion of the cube where x_HI = 0.
    l2_loss = LpLoss(d=3, p=2)
    h1_loss = H1Loss(d=3)
    train_loss_fn = WeightedSumLoss(
        (0.5, AbsLoss(l2_loss)),
        (0.5, AbsLoss(h1_loss)),
    )
    eval_losses = {"l2": AbsLoss(l2_loss), "h1": AbsLoss(h1_loss)}

    # -------------------------------------------- 8. trainer
    trainer = LoggingTrainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        data_processor=None,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        use_distributed=False,
        verbose=True,
        metrics_path=METRICS_PATH,
    )

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Modes: {N_MODES}, hidden: {HIDDEN_CHANNELS}, pos-emb: grid")
    print(f"In channels: density/10 + 1/(1+z), Out: x_HI")
    print(f"Loss: 0.5*absL2 + 0.5*absH1 (d=3)")
    print(f"DataLoader workers: {NUM_WORKERS} "
          f"(per-step log every {LOG_EVERY} batches)")
    print(f"Eval interval: every {EVAL_INTERVAL} epoch(s)")
    print(f"Metrics JSONL: {METRICS_PATH}")

    # -------------------------------------------- 9. train
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


if __name__ == "__main__":
    main()
