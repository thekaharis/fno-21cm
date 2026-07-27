"""Refit the contrast theta-schedule from the model's own recent predictions.

Alternating (EM-style) scheme:

    epoch 0      train normally, map disabled -- the model has no predictions yet
    epoch n >= 1 fit theta(mean_pred) to the epoch-(n-1) model, freeze it, and
                 train the next epoch with the mapped output

E-step is the schedule fit, M-step the epoch of training.  What makes this
different from fitting once on a converged model: at epoch 1 the prediction is
genuinely blur-dominated, so theta starts small and the network trains under
sharpening from the beginning, which can land it in a different basin.

The honest caveat is that this optimises the same objective as a *learnable*
map, only in a different order, and that objective is known to prefer the
identity (a learned global theta went to 4.43 ~= identity; the post-hoc
schedule fit collapsed to identity above x_HI = 0.005 because mean_pred locates
the responsive regime too poorly -- MAE 0.055 against a regime 0.045 wide).

So the theta trajectory is the measurement.  If it drifts to THETA_MAX the
scheme has reproduced that result from a different direction; if it settles
somewhere useful, co-adaptation beat the static fit.  Either way it is logged
per epoch.
"""

from __future__ import annotations

import torch

from contrast import THETA_MAX, ThetaSchedule, apply_contrast


@torch.no_grad()
def collect_base_outputs(model, loader, device, max_samples: int = 2048):
    """Pre-map predictions and targets from the current model.

    Deliberately the *base* output: the schedule maps that to the target, so
    fitting against an already-mapped prediction would compound two maps.
    """
    base = model.fno.base if hasattr(model.fno, "base") else model.fno
    was_training = model.training
    model.eval()
    preds, truths = [], []
    seen = 0
    for sample in loader:
        x = sample["x"].to(device, non_blocking=True)
        y = sample["y"].to(device, non_blocking=True)
        preds.append(base(x)[:, 0].float())
        truths.append(y[:, 0].float())
        seen += len(x)
        if seen >= max_samples:
            break
    if was_training:
        model.train()
    return torch.cat(preds), torch.cat(truths)


def fit_schedule(pred, truth, steps: int = 400, lr: float = 0.05,
                 init_lo: float = 0.5, init_hi: float = 4.0) -> dict:
    """Fit theta(mean_pred) by direct post-map MSE minimisation.

    Initialised mid-range on purpose: at theta = THETA_MAX the sigmoid squash
    sits at sigmoid(13.8), where d(theta)/d(raw) ~ 1e-6 and the fit cannot move
    at all.
    """
    device = pred.device
    sched = ThetaSchedule(theta_lo=init_lo, theta_hi=init_hi).to(device)
    opt = torch.optim.Adam(sched.parameters(), lr=lr)
    key = pred.flatten(1).mean(1).detach()
    for _ in range(steps):
        opt.zero_grad()
        th = sched(key).view((-1,) + (1,) * (pred.dim() - 1))
        loss = ((apply_contrast(pred, th, 0.5) - truth) ** 2).mean()
        if not torch.isfinite(loss):
            break                       # keep the last finite parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sched.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        base = float(((pred - truth) ** 2).mean().sqrt())
        th = sched(key).view((-1,) + (1,) * (pred.dim() - 1))
        got = float(((apply_contrast(pred, th, 0.5) - truth) ** 2).mean().sqrt())
    out = sched.state_dict_floats()
    out["train_gain_pct"] = 100.0 * (got / base - 1.0) if base > 0 else 0.0
    out["theta_median"] = float(sched(key).median())
    return out


def refit_and_install(model, loader, device, max_samples: int = 2048,
                      steps: int = 400) -> dict:
    """One E-step: refit from current predictions and install the result."""
    contrast = getattr(getattr(model, "fno", model), "contrast", None)
    if contrast is None or contrast.mode != "xhi":
        raise RuntimeError("refit requires a model wrapped with CONTRAST_MODE=xhi")
    pred, truth = collect_base_outputs(model, loader, device, max_samples)
    stats = fit_schedule(pred, truth, steps=steps)
    # Never install a degenerate fit: a non-finite schedule produces NaN
    # predictions, which propagate into the weights and end the run.
    try:
        contrast.schedule.load_floats(stats)
    except ValueError as exc:
        print(f"[contrast] refit rejected ({exc}); keeping previous schedule",
              flush=True)
        stats["rejected"] = 1.0
        return stats
    stats["rejected"] = 0.0
    for p in contrast.schedule.parameters():      # keep it fixed during the M-step
        p.requires_grad_(False)
    contrast.enabled = True
    stats["n_slices"] = int(len(pred))
    return stats


def disable(model) -> None:
    """Identity map, for the first epoch before any prediction exists."""
    contrast = getattr(getattr(model, "fno", model), "contrast", None)
    if contrast is not None:
        contrast.enabled = False
        contrast.schedule.load_floats(
            {"theta_lo": THETA_MAX, "theta_hi": THETA_MAX, "c": -1.5, "s": 0.4})
