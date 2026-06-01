#!/usr/bin/env python3
"""Train a 2-D Fourier Neural Operator on 21cm lightcone slices.

Mapping:  matter density → neutral fraction (x_HI) at each redshift.

Normalization: density is divided by a fixed constant (10) in the dataset.
Neutral fraction is left in its natural [0, 1] range.  No per-file
statistics are used — the same scale applies to all cosmologies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- Use the local neuraloperator checkout (./neuraloperator/neuralop) ----
# Prepend the in-repo copy so it shadows any pip-installed `neuralop`.
_LOCAL_NEURALOP = Path(__file__).resolve().parent / "neuraloperator"
if _LOCAL_NEURALOP.is_dir():
    sys.path.insert(0, str(_LOCAL_NEURALOP))
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

# If the local checkout is present, confirm we actually picked it up (not an
# installed copy).  If it is absent (e.g. a fresh clone without the vendored
# library), fall back silently to whatever ``neuralop`` is installed.
import neuralop as _neuralop
if _LOCAL_NEURALOP.is_dir():
    assert Path(_neuralop.__file__).resolve().is_relative_to(_LOCAL_NEURALOP), (
        f"Expected local neuralop from {_LOCAL_NEURALOP}, got {_neuralop.__file__}"
    )

from dataset import LightconeSliceDataset, split_by_file


# ------------------------------------------------------------------ config
DATA_DIR = Path("data")
N_Z = 256
Z_MIN, Z_MAX = 5.0, 25.0
INPUT_FIELD = "density"
TARGET_FIELD = "neutral_fraction"

N_MODES = (32, 32)
HIDDEN_CHANNELS = 64
N_LAYERS = 4
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Train / val / test split by file index (10 files total)
TRAIN_FILES = list(range(0, 7))
VAL_FILES = list(range(7, 9))
TEST_FILES = [9]


# ------------------------------------------------------------------ wrapper
class AbsLoss:
    """Wrap a neuralop loss to call its ``.abs()`` (absolute) instead of
    ``.rel()`` (relative), and swallow extra kwargs from the Trainer.
    """

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **kwargs):
        return self.loss.abs(out, y)


class RelLoss:
    """Wrap a neuralop loss to call its ``.rel()`` (relative) and swallow
    extra kwargs from the Trainer.  Relative losses normalize per sample, so
    near-constant target maps no longer dominate the objective.
    """

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **kwargs):
        return self.loss.rel(out, y)


class WeightedSumLoss:
    """Weighted sum of ``(weight, loss)`` terms; passes kwargs through."""

    def __init__(self, *terms):
        self.terms = terms  # iterable of (float weight, callable loss)

    def __call__(self, out, y, **kwargs):
        return sum(w * loss(out, y, **kwargs) for w, loss in self.terms)


class SilentFNO(nn.Module):
    """Thin wrapper that discards extra kwargs injected by the Trainer.

    The Trainer calls ``model(**sample)`` where *sample* contains both
    ``x`` and ``y``.  ``FNO.forward`` only accepts ``x``, so this wrapper
    swallows the rest and avoids a warning on every batch.

    Delegates other attribute access (``save_checkpoint``, etc.) to the
    underlying FNO model.  Submodule/parameter/buffer lookup via
    ``nn.Module``'s own ``_modules``/``_parameters``/``_buffers`` dicts
    is preserved (and falls through to the outer wrapper, which has no
    such entries since everything lives on ``self.fno``).
    """

    def __init__(self, fno: FNO):
        super().__init__()
        self.fno = fno

    def forward(self, x, **kwargs):
        return self.fno(x)

    def __getattr__(self, name):
        # Replicate nn.Module's standard lookup (params, buffers, modules).
        # Without this, ``self.fno`` itself cannot be resolved because
        # PyTorch stores it in ``_modules``, not ``__dict__``.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Delegate anything not on the wrapper to the underlying FNO.
        return getattr(self._modules["fno"], name)


# ------------------------------------------------------------------ main
def main():
    # ------------------------------------------------- 1. discover files
    h5_files = sorted(DATA_DIR.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {DATA_DIR.resolve()}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(h5_files)} lightcone files")

    # ------------------------------------------------- 2. build dataset
    print("Loading dataset (interpolating to common redshift grid) ...")
    ds = LightconeSliceDataset(
        file_paths=h5_files,
        n_z=N_Z,
        z_min=Z_MIN,
        z_max=Z_MAX,
        input_field=INPUT_FIELD,
        target_field=TARGET_FIELD,
        preload=True,
    )
    print(f"Total slices: {len(ds)}  ({len(h5_files)} files x {N_Z} z-bins)")

    # ------------------------------------------------- 3. split
    train_ds, val_ds, test_ds = split_by_file(
        ds, TRAIN_FILES, VAL_FILES, TEST_FILES,
    )
    print(f"Train: {len(train_ds)} slices ({len(TRAIN_FILES)} files)")
    print(f"Val:   {len(val_ds)} slices ({len(VAL_FILES)} files)")
    print(f"Test:  {len(test_ds)} slices ({len(TEST_FILES)} files)")

    # ------------------------------------------------- 4. dataloaders
    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE != "cpu"))
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    test_loaders = {"val": val_loader, "test": test_loader}

    # ------------------------------------------------- 5. model
    fno = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=1,
        out_channels=1,
        n_layers=N_LAYERS,
        projection_channel_ratio=2,
        positional_embedding="grid",
    )
    model = SilentFNO(fno).to(DEVICE)
    print(f"Model: {count_model_params(model.fno):,} parameters")
    print(model)

    # ------------------------------------------------- 6. optimizer / scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # ------------------------------------------------- 7. losses
    # Train on a relative-L2 + absolute-H1 blend: the relative L2 normalizes
    # per sample (so near-constant x_HI maps no longer dominate the loss), and
    # the H1 term penalizes gradient mismatch -> sharper ionization-bubble edges.
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    train_loss_fn = WeightedSumLoss(
        (0.5, RelLoss(l2_loss)),
        (0.5, AbsLoss(h1_loss)),
    )
    eval_losses = {"l2": RelLoss(l2_loss), "h1": AbsLoss(h1_loss)}

    # ------------------------------------------------- 8. trainer
    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        data_processor=None,
        wandb_log=False,
        eval_interval=5,
        use_distributed=False,
        verbose=True,
    )

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Modes: {N_MODES}, hidden: {HIDDEN_CHANNELS}, pos-emb: grid")
    print(f"Loss: 0.5*relL2 + 0.5*absH1")
    print(f"Normalization: density / 10 (physics-based, fixed)")

    # ------------------------------------------------- 9. train
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss_fn,
        eval_losses=eval_losses,
        save_every=10,
        save_dir="./checkpoints",
    )


if __name__ == "__main__":
    main()
