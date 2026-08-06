"""GPU probe for the SirenFNO2d z_re NaN: find the first non-finite tensor.

Replays the real training path (input cache, real cones, batch 8, Adam +
clip 1.0) for a limited number of batches with per-batch finite checks on
output, loss, gradients, and weights. On the first failure, re-runs that
batch under autograd anomaly detection and with TF32 disabled to separate
"bad op" from "TF32 numerics".

Run: python tests/probe_sirenfno_gpu.py  (needs a GPU)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset_zre import ZreMapDataset, split_by_cone
from models_zre_2d import SirenFNO2d
from neuralop import LpLoss, H1Loss
from losses import AbsoluteLoss, WeightedLoss

N_BATCHES = 700
ROOT = Path(__file__).resolve().parent.parent

def finite(t):
    return bool(torch.isfinite(t).all())

def main():
    device = "cuda"
    print("tf32 matmul:", torch.backends.cuda.matmul.allow_tf32,
          "| cudnn tf32:", torch.backends.cudnn.allow_tf32)
    files = sorted(Path("/pfs/10/work/hd_id260-fno_training/data/data")
                   .glob("21cmfast_11d_sample*.h5"))
    ds = ZreMapDataset(files, target_cache=ROOT / "zre_targets.h5",
                       n_z_in=64, use_params=True, preload=False,
                       density_cache=ROOT / "zre_inputs.h5")
    train_ds, _, _ = split_by_cone(ds, val_frac=0.1, test_frac=0.1, seed=42)
    ds.set_parameter_normalization(
        ds.fit_parameter_normalization(train_ds.indices))
    gen = torch.Generator().manual_seed(0)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                        generator=gen, num_workers=0)

    torch.manual_seed(0)
    model = SirenFNO2d(n_modes=(32, 32), hidden_channels=64,
                       in_channels=75).to(device)
    # pure L2, matching the failing zre-sirenfno-l2 run
    loss_fn = WeightedLoss((1.0, AbsoluteLoss(LpLoss(d=2, p=2))))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for i, sample in enumerate(loader):
        if i >= N_BATCHES:
            print(f"no NaN in {N_BATCHES} batches -- probe inconclusive")
            return
        x = sample["x"].to(device)
        y = sample["y"].to(device)
        if not finite(x) or not finite(y):
            print(f"batch {i}: NON-FINITE INPUT  x_ok={finite(x)} y_ok={finite(y)}")
            return
        opt.zero_grad()
        out = model(x)
        if not finite(out):
            print(f"batch {i}: NON-FINITE FORWARD OUTPUT")
            rerun_with_anomaly(model, loss_fn, x, y)
            return
        loss = loss_fn(out, y=y)
        if not finite(loss):
            print(f"batch {i}: NON-FINITE LOSS (output was finite)")
            print("  out range:", out.min().item(), out.max().item())
            rerun_with_anomaly(model, loss_fn, x, y)
            return
        loss.backward()
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and not finite(p.grad)]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bad_weights = [n for n, p in model.named_parameters()
                       if not finite(p)]
        if i % 20 == 0 or bad_grads or bad_weights:
            omega_max = max(p.abs().max().item()
                            for n, p in model.named_parameters()
                            if n.endswith(".omega"))
            proj_max = max(p.abs().max().item()
                           for n, p in model.named_parameters()
                           if n.endswith("mapping.projection"))
            sat = ((out < 1e-4) | (out > 1 - 1e-4)).float().mean().item()
            print(f"batch {i:3d}: loss={loss.item():.4f} "
                  f"grad_norm={float(gn):.3e} omega_max={omega_max:.1f} "
                  f"proj_max={proj_max:.1f} sat={sat:.3f} "
                  f"bad_grads={bad_grads[:3]} bad_weights={bad_weights[:3]}",
                  flush=True)
        if bad_grads or bad_weights:
            print(f"FIRST NON-FINITE at batch {i}")
            torch.save({"x": x.cpu(), "y": y.cpu(),
                        "state": {k: v.cpu() for k, v in model.state_dict().items()}},
                       ROOT / "tests" / "sirenfno_nan_dump.pt")
            print("dump saved to tests/sirenfno_nan_dump.pt")
            rerun_with_anomaly(model, loss_fn, x, y)
            return


def rerun_with_anomaly(model, loss_fn, x, y):
    print("--- rerun with anomaly detection ---", flush=True)
    try:
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            loss_fn(model(x), y=y).backward()
        print("anomaly rerun: no error raised")
    except RuntimeError as e:
        print("ANOMALY:", str(e)[:2000])
    print("--- rerun with TF32 disabled ---", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.zero_grad()
    out = model(x)
    loss = loss_fn(out, y=y)
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"tf32-off: out_finite={finite(out)} loss={loss.item():.4f} "
          f"bad_grads={bad[:5]}")


if __name__ == "__main__":
    main()
