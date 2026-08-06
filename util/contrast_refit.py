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

In 3-D the key itself is attacked as well: a cone's per-slice means are noisy
samples of one monotone x_HI(z) curve, so an isotonic fit along the LOS axis
sharpens the estimate without using the truth (``contrast.los_key``).  That
helps only against the *jitter* part of the 0.055 and does nothing against
bias, so ``key_jitter`` is logged beside the schedule to say which case holds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from contrast import (
    SteppedThetaSchedule,
    ThetaSchedule,
    apply_contrast,
    isotonic,
    los_key,
    los_mean,
)

# Where sharpening measurably paid in 2-D (NOTES-contrast-map.md 6.1). Slices
# outside it are the identity, so the share of a cone that lands inside caps
# what the whole scheme can be worth.
BAND_LO, BAND_HI = 0.005, 0.05


@torch.no_grad()
def collect_base_outputs(model, loader, device, max_samples: int = 2048,
                         flatten: bool = False):
    """Pre-map predictions and targets from the current model.

    Deliberately the *base* output: the schedule maps that to the target, so
    fitting against an already-mapped prediction would compound two maps.

    ``max_samples`` counts LOS slices in both modes, so the knob means the same
    thing either way.

    ``flatten`` splits 3-D cubes into their transverse LOS slices, which makes
    the rest of the path identical to the 2-D case.  It is **off** by default,
    because it silently changes the objective: theta is fitted under the run's
    own training loss, and ``expwall``'s signed distance measured within a
    single transverse plane is not the distance the 3-D loss computes -- a wall
    sitting just off-slice along the LOS is invisible to it.  Since theta is
    constant within a slice anyway, there is nothing to gain by splitting.
    """
    inner = model.fno.module if hasattr(model.fno, "module") else model.fno
    base = inner.base if hasattr(inner, "base") else inner
    was_training = model.training
    model.eval()
    preds, truths = [], []
    seen = 0
    for sample in loader:
        x = sample["x"].to(device, non_blocking=True)
        y = sample["y"].to(device, non_blocking=True)
        # Keep the channel axis: the refit objective is the run's own training
        # loss, and it must see the shapes it sees during training.
        p, q = base(x).float(), y.float()
        if p.dim() == 5 and flatten:
            p = p.permute(0, 4, 1, 2, 3).reshape(-1, *p.shape[1:4])
            q = q.permute(0, 4, 1, 2, 3).reshape(-1, *q.shape[1:4])
        preds.append(p)
        truths.append(q)
        seen += len(p) * (p.shape[-1] if p.dim() == 5 else 1)
        if seen >= max_samples:
            break
    if was_training:
        model.train()
    pred, truth = torch.cat(preds), torch.cat(truths)
    if pred.dim() == 5:                       # max_samples is in slices
        keep = max(1, -(-max_samples // pred.shape[-1]))
    else:
        keep = max_samples
    return pred[:keep], truth[:keep]


def _mse(out, y):
    return ((out - y) ** 2).mean()


def _world(sync: bool) -> int:
    """Number of participating ranks; 1 whenever there is nothing to sync."""
    if not sync or not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def _sum_(values, device) -> list[float]:
    """all_reduce SUM of a small list of scalars, as floats."""
    t = torch.tensor([float(v) for v in values], dtype=torch.float64,
                     device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(v) for v in t]


def _sync_step(params, loss_ok: bool, grad_ok: bool, world: int
               ) -> tuple[bool, bool]:
    """Average the fit's gradients across ranks and agree on whether to step.

    One collective per step, carrying the gradients and both health flags
    together -- the schedule is a handful of scalars, so this is far cheaper
    than the model's own gradient reduction that already runs every step.

    The flags have to travel with the gradients rather than being decided
    locally.  A rank that saw a non-finite loss skips its ``backward`` and so
    would skip the collective too, leaving every other rank waiting forever on
    an all_reduce that never completes.  Deciding collectively means all ranks
    break, skip, or step on the same iteration.
    """
    flat = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in params
    ] + [torch.tensor([float(loss_ok), float(grad_ok)], device=params[0].device,
                      dtype=params[0].dtype)])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    n_grad = flat.numel() - 2
    all_loss_ok = float(flat[n_grad]) >= world
    all_grad_ok = float(flat[n_grad + 1]) >= world
    if all_loss_ok and all_grad_ok:
        offset = 0
        for p in params:
            size = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat[offset:offset + size].view_as(p) / world)
            offset += size
    return all_loss_ok, all_grad_ok


def _broadcast_floats(stats: dict, device, world: int) -> dict:
    """Replace the fitted schedule with rank 0's copy, in place.

    Gradient averaging already drives every rank through identical updates, so
    this is belt and braces -- but it makes "all ranks hold the same map" a
    property of the code rather than an assumption about floating-point
    determinism across ranks, and it costs one broadcast of ~14 floats.
    """
    keys = (["thetas"] if "thetas" in stats
            else ["theta_lo", "theta_hi", "c", "s"])
    packed, sizes = [], []
    for k in keys:
        v = stats[k]
        seq = list(v) if isinstance(v, (list, tuple)) else [v]
        sizes.append((k, len(seq), isinstance(v, (list, tuple))))
        packed.extend(float(x) for x in seq)
    t = torch.tensor(packed, dtype=torch.float64, device=device)
    dist.broadcast(t, src=0)
    vals = [float(x) for x in t]
    offset = 0
    for k, size, is_seq in sizes:
        chunk = vals[offset:offset + size]
        stats[k] = chunk if is_seq else chunk[0]
        offset += size
    return stats


def _schedule_key(pred, key_mode: str):
    """Bucketising key, and the reshape that broadcasts theta back over ``pred``.

    Cubes are keyed per LOS slice -- ``(N, W)`` -- so one fitted table serves
    every slice of every cone, and a cone whose reionisation happens early gets
    the same theta at x_HI = 0.02 as one where it happens late.  Flat inputs
    (2-D slices, or cubes the caller already split) are keyed per sample.
    """
    if pred.dim() == 5:
        key = los_key(pred.detach(), key_mode)                  # (N, W)
        return key, lambda th: th[:, None, None, None, :]       # (N,1,1,1,W)
    key = pred.flatten(1).mean(1).detach()
    shape = (-1,) + (1,) * (pred.dim() - 1)
    return key, lambda th: th.view(shape)


def _key_diagnostics(pred, key, world: int = 1) -> dict:
    """The two numbers that decide whether any of this can pay, logged per epoch.

    ``key_jitter`` is the RMS distance from the raw per-slice means to their
    monotone fit -- the part of the key error that is slice-to-slice *noise*
    rather than bias.  This is the whole premise of ``key_mode="monotone"``:
    isotonic smoothing removes jitter and is exactly powerless against bias, so
    a jitter near zero says the smoothed key is no better than the raw one and
    the responsive band will keep being missed for a reason smoothing cannot
    reach.  Measured against the model's own means, so it needs no truth.

    ``frac_band`` is the share of LOS slices landing in the range where
    sharpening paid in 2-D.  It bounds the pooled gain: the rest of the cone
    gets the identity no matter how well the schedule is fitted.
    """
    if pred.dim() != 5:
        return {}
    raw = los_mean(pred.detach()).float().cpu().numpy().astype(np.float64)
    _, sse = isotonic(raw)
    inside = float(((key >= BAND_LO) & (key <= BAND_HI)).sum())
    total = float(raw.size)
    if world > 1:
        # Reduce the sums, then divide once. Averaging each rank's ratio would
        # be wrong the moment the shards differ in size, which DistributedSampler
        # padding already allows.
        sse, inside, total = _sum_((sse, inside, total), key.device)
    return {"key_jitter": float(np.sqrt(sse / max(total, 1.0))),
            "frac_band": inside / max(total, 1.0)}


def fit_schedule(pred, truth, steps: int = 400, lr: float = 0.05,
                 init_lo: float = 0.5, init_hi: float = 4.0,
                 objective=None, batch: int = 128,
                 template=None, key_mode: str = "monotone",
                 sync: bool = False) -> dict:
    """Fit theta(x_HI) against ``objective`` (default MSE).

    ``sync`` makes the fit collective across DDP ranks: gradients are averaged
    every step, so the table is fitted on the *union* of the shards instead of
    each rank fitting its own from 1/world_size of the data.  That matters more
    here than the usual data-parallel speedup, because the bins that carry the
    gain hold only a few percent of the slices -- split across ranks they can
    reach counts too small to fit at all.  Reported counts and gains become
    global too.

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
    params = list(sched.parameters())
    world = _world(sync)
    key, spread = _schedule_key(pred, key_mode)
    n = len(pred)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        # A minibatch, not the whole set: some loss terms build very large
        # intermediates (SWD's cumsum asked for 7.18 GiB over 2048 slices).
        sel = (torch.randperm(n, generator=gen)[:batch].to(pred.device)
               if n > batch else slice(None))
        opt.zero_grad()
        th = spread(sched(key[sel]))
        loss = objective(apply_contrast(pred[sel], th, 0.5), truth[sel])
        loss_ok = bool(torch.isfinite(loss))
        grad_ok = True
        if loss_ok:
            loss.backward()
            # A single non-finite gradient would otherwise make
            # clip_grad_norm_'s total norm NaN and poison every parameter in
            # one step.
            grad_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                          for p in params)
        if world > 1:
            loss_ok, grad_ok = _sync_step(params, loss_ok, grad_ok, world)
        if not loss_ok:
            break                       # keep the last finite parameters
        if not grad_ok:
            opt.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        b_sum = g_sum = 0.0
        chunks = 0
        for s in range(0, n, batch):
            pc, tc = pred[s:s + batch], truth[s:s + batch]
            th = spread(sched(key[s:s + batch]))
            b_sum += float(objective(pc, tc))
            g_sum += float(objective(apply_contrast(pc, th, 0.5), tc))
            chunks += 1
        if world > 1:
            b_sum, g_sum, chunks = _sum_((b_sum, g_sum, chunks), key.device)
        base, got = b_sum / max(chunks, 1), g_sum / max(chunks, 1)
    out = sched.state_dict_floats()
    out["train_gain_pct"] = 100.0 * (got / base - 1.0) if base > 0 else 0.0
    out["theta_median"] = float(sched(key).detach().median())
    out["n_slices"] = int(key.numel())
    out.update(_key_diagnostics(pred, key, world))
    if isinstance(sched, SteppedThetaSchedule):
        counts = torch.bincount(sched.bin_of(key).reshape(-1),
                                minlength=sched.n_bins)
        if world > 1:
            # Global occupancy: a bin that is thin on every rank individually is
            # what the log has to show, since the fit now pools all of them.
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        out["bin_counts"] = [int(c) for c in counts]
        out["n_bins_seen"] = int((counts > 0).sum())
        out["n_slices"] = int(sum(out["bin_counts"]))
    return out


def refit_and_install(model, loader, device, max_samples: int = 2048,
                      steps: int = 400, objective=None,
                      theta_floor: float = 0.25, batch: int = 128,
                      flatten: bool = False, sync: bool = False) -> dict:
    """One E-step: refit from current predictions and install the result.

    The fit is keyed exactly as inference will be -- same ``key_mode``, same
    per-LOS-slice granularity -- so a theta fitted for a bin is the theta that
    bin will actually receive.

    Under DDP, pass ``sync=True``.  Every rank then fits the same table on the
    pooled shards and installs a bitwise-identical copy of it; without it each
    rank trains the next epoch under a different map.
    """
    contrast = getattr(getattr(model, "fno", model), "contrast", None)
    if contrast is None or contrast.mode != "xhi":
        raise RuntimeError("refit requires a model wrapped with CONTRAST_MODE=xhi")
    pred, truth = collect_base_outputs(model, loader, device, max_samples,
                                       flatten=flatten)
    stats = fit_schedule(pred, truth, steps=steps, objective=objective,
                         template=contrast.schedule, batch=batch,
                         key_mode=contrast.key_mode, sync=sync)
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
    # After the floor, so what is broadcast is exactly what gets installed.
    # It also makes the accept/reject decision below unanimous: every rank
    # validates the same numbers, so none can install a map the others rejected.
    world = _world(sync)
    if world > 1:
        _broadcast_floats(stats, device, world)
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
    if "key_jitter" in stats:
        # Read these together: frac_band caps the prize, key_jitter says whether
        # the monotone key can help find it. See _key_diagnostics.
        tail += (f" [key jitter {stats['key_jitter']:.4f}, "
                 f"{100 * stats['frac_band']:.1f}% of slices in band]")
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
