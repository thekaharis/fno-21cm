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

from contrast import SteppedThetaSchedule, ThetaSchedule, apply_contrast


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
        # Keep the channel axis: the refit objective is the run's own training
        # loss, and it must see the shapes it sees during training.
        preds.append(base(x).float())
        truths.append(y.float())
        seen += len(x)
        if seen >= max_samples:
            break
    if was_training:
        model.train()
    return torch.cat(preds), torch.cat(truths)


def _mse(out, y):
    return ((out - y) ** 2).mean()


def fit_schedule(pred, truth, steps: int = 400, lr: float = 0.05,
                 init_lo: float = 0.5, init_hi: float = 4.0,
                 objective=None, batch: int = 128,
                 template=None) -> dict:
    """Fit theta(mean_pred) against ``objective`` (default MSE).

    Pass the run's own training loss for ``objective``.  Fitting the map under
    MSE while the network trains under something else optimises the map for a
    criterion nobody is using -- and in a deliberately L2-free run it quietly
    reintroduces L2 through the back door.

    Initialised mid-range on purpose: at theta = THETA_MAX the sigmoid squash
    sits at sigmoid(13.8), where d(theta)/d(raw) ~ 1e-6 and the fit cannot move
    at all.
    """
    objective = objective or _mse
    device = pred.device
    if isinstance(template, SteppedThetaSchedule):
        # Warm-start from the installed table: a refit batch does not populate
        # every bin, and an unvisited bin must keep its value rather than
        # snapping back to the initialisation.
        sched = SteppedThetaSchedule(n_bins=template.n_bins,
                                     edges=[0.0] + [float(e) for e in template.edges])
        warm = template.state_dict_floats()
        # Unstick saturated bins. theta = THETA_MAX maps to sigmoid(13.8), where
        # d(theta)/d(raw) ~ 1e-6 and the bin cannot move at all -- so an
        # identity-initialised table would stay identity forever. Clamp the
        # starting point into the responsive part of the squash; learned values
        # below init_hi are carried over untouched.
        warm["thetas"] = [min(v, init_hi) for v in warm["thetas"]]
        sched.load_floats(warm)
        sched = sched.to(device)
    else:
        sched = ThetaSchedule(theta_lo=init_lo, theta_hi=init_hi).to(device)
    opt = torch.optim.Adam(sched.parameters(), lr=lr)
    key = pred.flatten(1).mean(1).detach()
    n = len(pred)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        # A minibatch, not the whole set: some loss terms build very large
        # intermediates (SWD's cumsum asked for 7.18 GiB over 2048 slices).
        sel = (torch.randperm(n, generator=gen)[:batch].to(pred.device)
               if n > batch else slice(None))
        opt.zero_grad()
        th = sched(key[sel]).view((-1,) + (1,) * (pred.dim() - 1))
        loss = objective(apply_contrast(pred[sel], th, 0.5), truth[sel])
        if not torch.isfinite(loss):
            break                       # keep the last finite parameters
        loss.backward()
        # A single non-finite gradient would otherwise make clip_grad_norm_'s
        # total norm NaN and poison every parameter in one step.
        if not all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in sched.parameters()):
            opt.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(sched.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        b_sum = g_sum = 0.0
        chunks = 0
        for s in range(0, n, batch):
            pc, tc = pred[s:s + batch], truth[s:s + batch]
            th = sched(key[s:s + batch]).view((-1,) + (1,) * (pred.dim() - 1))
            b_sum += float(objective(pc, tc))
            g_sum += float(objective(apply_contrast(pc, th, 0.5), tc))
            chunks += 1
        base, got = b_sum / max(chunks, 1), g_sum / max(chunks, 1)
    out = sched.state_dict_floats()
    out["train_gain_pct"] = 100.0 * (got / base - 1.0) if base > 0 else 0.0
    out["theta_median"] = float(sched(key).median())
    if isinstance(sched, SteppedThetaSchedule):
        counts = torch.bincount(sched.bin_of(key), minlength=sched.n_bins)
        out["bin_counts"] = [int(c) for c in counts]
        out["n_bins_seen"] = int((counts > 0).sum())
    return out


def refit_and_install(model, loader, device, max_samples: int = 2048,
                      steps: int = 400, objective=None,
                      theta_floor: float = 0.25) -> dict:
    """One E-step: refit from current predictions and install the result."""
    contrast = getattr(getattr(model, "fno", model), "contrast", None)
    if contrast is None or contrast.mode != "xhi":
        raise RuntimeError("refit requires a model wrapped with CONTRAST_MODE=xhi")
    pred, truth = collect_base_outputs(model, loader, device, max_samples)
    stats = fit_schedule(pred, truth, steps=steps, objective=objective,
                         template=contrast.schedule)
    stats["n_slices"] = int(len(pred))      # set before any early return
    # The E-step optimises theta for a frozen prediction and is blind to the
    # M-step's dynamics: the map amplifies gradients by ~1/(2*theta), and a
    # fitted 0.031 amplified 16x and produced NaN weights within one epoch.
    if "thetas" in stats:
        pre = list(stats["thetas"])
        stats["thetas"] = [max(v, theta_floor) for v in pre]
        stats["n_bins_floored"] = sum(1 for a, b in zip(pre, stats["thetas"]) if a < b)
        # theta_median came from the *unfloored* fit, so the log would otherwise
        # report a median below values it also prints as floored. Recompute it
        # over the installed table, weighted by how many slices each bin holds.
        counts = stats.get("bin_counts") or []
        if counts:
            paired = sorted((v, c) for v, c in zip(stats["thetas"], counts) if c)
            total, acc = sum(c for _, c in paired), 0
            for v, c in paired:
                acc += c
                if acc * 2 >= total:
                    stats["theta_median"] = v
                    break
    else:
        for k in ("theta_lo", "theta_hi"):
            if stats[k] < theta_floor:
                stats[k + "_prefloor"] = stats[k]
                stats[k] = theta_floor
        stats["theta_median"] = max(stats["theta_median"], theta_floor)
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
    return stats


def summary_line(stats: dict) -> str:
    """One line describing a refit, for either schedule shape.

    Lives here rather than in the trainer so the formatting is exercised by the
    same tests as the fit. Three separate consumers of this dict have now been
    broken by assuming the sigmoid's keys; there should only be one.
    """
    if stats.get("rejected"):
        return "refit rejected; keeping previous schedule"
    tail = (f"median={stats['theta_median']:.3f} "
            f"(train gain {stats['train_gain_pct']:+.2f}% on "
            f"{stats['n_slices']} slices)")
    if "thetas" in stats:                       # stepped: a per-bin table
        th, counts = stats["thetas"], stats.get("bin_counts") or []
        edges = stats["edges"]
        seen = [i for i in range(len(th)) if not counts or counts[i]]
        body = " ".join(f"{edges[i]:.3g}:{th[i]:.2f}" for i in seen)
        floored = stats.get("n_bins_floored") or 0
        return (f"bins[{len(seen)}/{len(th)} seen"
                f"{f', {floored} floored' if floored else ''}] {body}  {tail}")
    return (f"theta {stats['theta_lo']:.3f}->{stats['theta_hi']:.3f} "
            f"@log10(m)={stats['c']:.2f} w={stats['s']:.2f} {tail}")


def disable(model) -> None:
    """Identity map, for the first epoch before any prediction exists.

    The ``enabled`` gate is an exact passthrough, so there is nothing else to
    do.  This used to also overwrite the schedule with sigmoid-shaped floats,
    which raised KeyError('thetas') against a stepped schedule -- and was
    redundant even for the sigmoid, since both kinds initialise at the identity
    and a resumed run wants its learned values kept.
    """
    contrast = getattr(getattr(model, "fno", model), "contrast", None)
    if contrast is not None:
        contrast.enabled = False
