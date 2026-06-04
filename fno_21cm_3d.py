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

import sys
from pathlib import Path

# ---- Prefer a vendored neuralop next to this script ------------------------
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "neuraloperator", _HERE):
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

from dataset_3d import LightconeCubeDataset, split_cubes


# ------------------------------------------------------------------ config
DATA_DIR = Path("data")
FILE_GLOB = "21cmfast_11d_sample*.h5"

N_Z = 256                           # LOS resolution after interpolation
Z_MIN, Z_MAX = 5.0, 25.0

N_MODES = (16, 16, 16)
HIDDEN_CHANNELS = 32
N_LAYERS = 4
BATCH_SIZE = 1                      # 3-D cubes are heavy; raise after profiling
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

SPLIT_SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

CHECKPOINT_DIR = "./checkpoints_3d"


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


# ------------------------------------------------------------------ main
def main():
    # -------------------------------------------- 1. discover lightcones
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        print(f"No lightcone files found under {DATA_DIR}/{FILE_GLOB}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} lightcone files in {DATA_DIR}")

    # -------------------------------------------- 2. dataset (streamed)
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
    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0,
                     pin_memory=(DEVICE != "cpu"))
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
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
    print(f"In channels: density/10 + 1/(1+z), Out: x_HI")
    print(f"Loss: 0.5*absL2 + 0.5*absH1 (d=3)")

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
