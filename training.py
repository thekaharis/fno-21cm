"""Training scaffolding shared by the three entry points.

The x_HI cube, x_HI slice and z_re map trainers differ in their datasets and
losses. Everything around that -- distributed setup, the dashboard's metrics
JSONL, the contrast refit, throughput logging -- was three near-copies and is
now one :class:`MetricsTrainer`. Every optional feature is off by default, so
a single-process run takes exactly the path the old single-process trainers did.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from neuralop import Trainer
from learned_waveform_operator import waveform_diagnostics

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------------------------------------------- distributed

def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group from SLURM's environment.

    Returns ``(rank, local_rank, world_size)``. Single-process runs return
    ``(0, 0, 1)`` without touching ``torch.distributed``.
    """
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if world_size <= 1:
        return 0, 0, 1
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
        try:
            hosts = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist], text=True
            ).split()
            os.environ["MASTER_ADDR"] = hosts[0]
        except (OSError, subprocess.CalledProcessError):
            os.environ["MASTER_ADDR"] = nodelist.split(",")[0]
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    return rank, local_rank, world_size


def all_reduce_mean(value: float, world_size: int) -> float:
    """Average a scalar across ranks; returns the input when not distributed."""
    if world_size <= 1 or not dist.is_initialized():
        return float(value)
    t = torch.tensor([float(value)], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


def all_reduce_weighted_metrics(
    metrics: Mapping[str, float | torch.Tensor],
    local_sample_count: int,
    world_size: int,
    device: str | torch.device,
) -> dict[str, float]:
    """Combine per-rank metric means using their local sample counts.

    ``neuralop.Trainer.evaluate`` returns a mean over this rank's samples.
    Reconstruct each local sum, reduce sums and counts, then divide once, so
    uneven shard sizes stay correct.
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
        key: float(reduced[i].item()) / global_count for i, key in enumerate(keys)
    }


# ---------------------------------------------------------------------- loader

class ProgressLoader:
    """Wrap a DataLoader to print throughput every ``log_every`` steps.

    ``neuralop.Trainer`` only logs per-epoch summaries, so a long epoch gives
    no signal at all until it completes. Preserves the DataLoader interface.
    """

    def __init__(self, loader, log_every: int = 25, tag: str = "train",
                 rank: int = 0):
        self.loader = loader
        self.log_every = int(log_every)
        self.tag = tag
        self._is_rank_0 = int(rank) == 0

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        total = len(self.loader)
        batch = int(getattr(self.loader, "batch_size", 1) or 1)
        start = time.time()
        for step, sample in enumerate(self.loader, start=1):
            yield sample
            if not (self._is_rank_0 and self.log_every
                    and step % self.log_every == 0):
                continue
            elapsed = max(time.time() - start, 1e-9)
            rate = step * batch / elapsed
            eta = (total - step) * elapsed / step
            print(f"[{self.tag} step {step}/{total}] {rate:.1f} samples/s, "
                  f"ETA {int(eta) // 60}:{int(eta) % 60:02d}", flush=True)


# --------------------------------------------------------------------- refit

@dataclass
class ContrastRefit:
    """Settings for the alternating contrast-schedule refit.

    Grouped so the trainer takes one optional object rather than eight
    keywords that only matter when the contrast map is on.
    """

    samples: int = 2048
    steps: int = 300
    batch: int = 128
    theta_floor: float = 0.25
    flatten: bool = False
    objective: Callable | None = None
    loader: Any = None


# -------------------------------------------------------------------- trainer

class MetricsTrainer(Trainer):
    """``neuralop.Trainer`` plus the dashboard's per-epoch metrics JSONL.

    One JSON object per epoch (``epoch``, ``train_err``, ``avg_loss``,
    ``epoch_train_time``, plus ``val_*``/``test_*`` on eval epochs) -- what
    ``dashboard/serve.py`` scans ``checkpoints/*/metrics.jsonl`` for.

    Optional, each inert unless configured:

    * ``world_size > 1`` reduces train and eval metrics across ranks, and
      restricts every file write to rank 0.
    * ``contrast`` runs the schedule refit at the top of each epoch.
    * ``spectral_history`` records Fourier-weight profiles per epoch.
    * ``saturation_ndim`` adds prediction-collapse statistics to eval.
    """

    def __init__(
        self,
        *args,
        metrics_path: str | Path | None = None,
        append: bool = False,
        rank: int = 0,
        world_size: int = 1,
        contrast: ContrastRefit | None = None,
        spectral_history=None,
        saturation_ndim: int | None = None,
        waveform_training=None,
        **kwargs,
    ):
        if kwargs.get("use_distributed", False):
            raise ValueError(
                "MetricsTrainer must not wrap DDP itself; the entry point "
                "constructs the single DDP wrapper before the optimizer."
            )
        super().__init__(*args, **kwargs)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._is_rank_0 = self._rank == 0
        self.contrast = contrast
        self.spectral_history = spectral_history
        self.saturation_ndim = saturation_ndim
        self.waveform_training = waveform_training
        self._waveform_initial_saved = False
        if (waveform_training is not None and waveform_training.config.mode != "joint"
                and contrast is not None):
            raise ValueError("disable contrast refitting for waveform-only/alternating training; it changes frozen parameters")
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path is not None and self._is_rank_0:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                # A fresh run starts a fresh history rather than mixing its
                # trajectory with stale rows from an older checkpoint dir.
                self.metrics_path.unlink(missing_ok=True)
        self._last_train: dict | None = None
        self._schedule_stats: dict | None = None
        self.best_epoch: int | None = None
        self.best_metric: float | None = None

    # -- training ---------------------------------------------------------
    def train_one_epoch(self, epoch, train_loader, training_loss):
        if self.waveform_training is not None:
            if self.waveform_training.optimizer is not self.optimizer:
                raise ValueError("waveform controller and trainer must share the same optimizer")
            self.waveform_training.begin_epoch(int(epoch))
            if self.waveform_training.enabled:
                if self.verbose:
                    print(f"[waveform] epoch {epoch}: {self.waveform_training.phase}", flush=True)
                if self._is_rank_0 and self.metrics_path is not None and not self._waveform_initial_saved:
                    torch.save(self.waveform_training.initial_tables,
                               self.metrics_path.parent / "waveform_initial_tables.pt")
                    self._waveform_initial_saved = True
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            # Reshuffle consistently across ranks each epoch.
            sampler.set_epoch(int(epoch))
        if hasattr(training_loss, "set_epoch"):
            training_loss.set_epoch(int(epoch))
        if self.contrast is not None:
            self._refit_contrast(int(epoch), train_loader)

        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        out = super().train_one_epoch(epoch, train_loader, training_loss)
        train_err, avg_loss, avg_lasso, elapsed = out

        # Under DDP each rank sees only its shard, so per-rank figures are
        # partial; average them for a globally meaningful log.
        row = dict(
            epoch=int(epoch),
            train_err=all_reduce_mean(train_err, self._world_size),
            avg_loss=all_reduce_mean(avg_loss, self._world_size),
            epoch_train_time=float(elapsed),
            train_samples_per_second=float(
                len(train_loader)
                * int(getattr(train_loader, "batch_size", 1) or 1)
                * self._world_size
                / max(float(elapsed), 1e-12)
            ),
        )
        if avg_lasso is not None:
            row["avg_lasso_loss"] = all_reduce_mean(avg_lasso, self._world_size)
        if self._schedule_stats:
            row.update({f"contrast_{k}": float(v)
                        for k, v in self._schedule_stats.items()
                        if isinstance(v, (int, float))})
            for key in ("thetas", "bin_counts"):
                if key in self._schedule_stats:
                    row[f"contrast_{key}"] = list(self._schedule_stats[key])
        if hasattr(training_loss, "active_weights"):
            # Keyed by term name, not position. The three entry points use
            # different term orders -- the 2-D stack has swd at index 3 where
            # the 3-D one has ionized_wall -- so positional logging silently
            # mislabels the weights, and crashes outright on a shorter stack.
            names = getattr(training_loss, "term_names", ())
            for name, weight in zip(names, training_loss.active_weights):
                row[f"active_{name}_weight"] = float(weight)
        if hasattr(training_loss, "pop_term_means"):
            # Raw per-term losses; the weights above make each term's weighted
            # contribution reconstructable. Terms skipped by a zero weight
            # (H1 during warmup) simply have no key.
            for name, value in training_loss.pop_term_means().items():
                row[f"train_{name}_term"] = all_reduce_mean(
                    value, self._world_size
                )
        grad_norm = getattr(self.optimizer, "_last_grad_norm", None)
        row.update(waveform_diagnostics(self.model))
        if self.waveform_training is not None:
            row.update(self.waveform_training.metrics(int(epoch)))
        if grad_norm is not None:
            row["last_grad_norm"] = float(grad_norm)
        if device.type == "cuda":
            row["peak_cuda_memory_gb"] = float(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
        self._last_train = row

        if self.spectral_history is not None:
            self.spectral_history.record(int(epoch))
        if self.eval_interval and epoch % self.eval_interval != 0:
            self._flush_row({})
        return out

    def train_one_batch(self, idx, sample, training_loss):
        # The parent calls model.train() after on_epoch_start, so enforce the
        # frozen module behavior here, immediately before each forward pass.
        if self.waveform_training is not None:
            self.waveform_training.prepare_batch()
        return super().train_one_batch(idx, sample, training_loss)

    def _refit_contrast(self, epoch: int, train_loader) -> None:
        from util import contrast_refit as refit

        if epoch == 0:
            refit.disable(self.model)      # nothing to fit against yet
            self._schedule_stats = None
            if self._is_rank_0:
                print("[contrast] epoch 0: map disabled", flush=True)
            return
        settings = self.contrast
        self._schedule_stats = refit.refit_and_install(
            self.model, settings.loader or train_loader, self.device,
            settings.samples, settings.steps,
            objective=settings.objective,
            theta_floor=settings.theta_floor,
            batch=settings.batch,
            flatten=settings.flatten,
            sync=self._world_size > 1,
        )
        if self._is_rank_0:
            print(f"[contrast] epoch {epoch}: "
                  f"{refit.summary_line(self._schedule_stats)}", flush=True)

    # -- evaluation -------------------------------------------------------
    def eval_one_batch(self, sample, eval_losses, return_output=False):
        if self.saturation_ndim is None:
            return super().eval_one_batch(sample, eval_losses, return_output)
        losses, output = super().eval_one_batch(
            sample, eval_losses, return_output=True
        )
        # A regular 8-cell stride samples ~0.2% of a cube -- ample to detect
        # all-zero/all-one collapse without adding full-volume float64
        # reductions to every evaluation batch.
        stride = (Ellipsis,) + (slice(None, None, 8),) * self.saturation_ndim
        sampled = output.detach()[stride]
        self._pred_sum += sampled.sum()
        self._pred_sq_sum += sampled.square().sum()
        self._pred_low_count += (sampled <= 1e-4).sum()
        # Simulated x_HI never reaches 1 (residual ionized floor caps it at
        # ~0.99983), so a 0.999 cutoff counts only unphysical clipping.
        self._pred_high_count += (sampled >= 0.999).sum()
        self._pred_count += sampled.numel()
        return losses, output if return_output else None

    def evaluate(self, *args, **kwargs):
        if self.saturation_ndim is not None:
            zero = lambda dtype: torch.zeros((), dtype=dtype, device=self.device)
            self._pred_sum = zero(torch.float32)
            self._pred_sq_sum = zero(torch.float32)
            self._pred_low_count = zero(torch.float64)
            self._pred_high_count = zero(torch.float64)
            self._pred_count = 0
        metrics = super().evaluate(*args, **kwargs)
        if self.saturation_ndim is not None:
            log_prefix = str(kwargs.get("log_prefix", "")).strip()
            prefix = f"{log_prefix}_" if log_prefix else ""
            count = max(int(self._pred_count), 1)
            mean = self._pred_sum / count
            variance = (self._pred_sq_sum / count - mean.square()).clamp_min(0)
            metrics.update({
                f"{prefix}pred_mean": float(mean.item()),
                f"{prefix}pred_std": float(variance.sqrt().item()),
                f"{prefix}pred_sat_low": float((self._pred_low_count / count).item()),
                f"{prefix}pred_sat_high": float((self._pred_high_count / count).item()),
            })
        if self._world_size <= 1:
            return metrics
        return all_reduce_weighted_metrics(
            metrics,
            local_sample_count=self.n_samples,
            world_size=self._world_size,
            device=self.device,
        )

    def evaluate_all(self, *args, **kwargs):
        metrics = super().evaluate_all(*args, **kwargs)
        # evaluate() has already reduced every loader's metrics across ranks.
        clean = {key: float(value) for key, value in metrics.items()}
        monitored = clean.get("val_l2")
        if monitored is not None and (
            self.best_metric is None or monitored < self.best_metric
        ):
            self.best_metric = monitored
            self.best_epoch = int(kwargs.get("epoch", -1))
        self._flush_row(clean)
        return metrics

    def resume_state_from_dir(self, save_dir):
        save_dir = Path(save_dir)
        manifest_path = save_dir / "manifest.pt"
        if not manifest_path.exists():
            if self.waveform_training is not None and self.waveform_training.config.mode != "joint":
                raise ValueError("waveform resume requires a complete training manifest; use INIT_CHECKPOINT for a warm start")
            super().resume_state_from_dir(save_dir)
            self.start_epoch += 1
        else:
            # Read the model named by the manifest. Preferencing best_model
            # would pair old weights with final optimizer/scheduler state.
            manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
            root = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            root.load_state_dict(torch.load(save_dir / manifest["model"], map_location="cpu", weights_only=True))
            for key in ("optimizer", "scheduler", "regularizer"):
                target = getattr(self, key, None)
                path = save_dir / manifest.get(key, f"{key}.pt")
                if target is not None:
                    if not path.exists():
                        raise FileNotFoundError(f"incomplete training state: missing {path}; use INIT_CHECKPOINT for weights only")
                    target.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.start_epoch = int(manifest.get("epoch", -1)) + 1
        if self.waveform_training is not None:
            self.waveform_training.validate_resume(self.start_epoch - 1)
        if self.verbose:
            print(f"Continuing with epoch {self.start_epoch}")

    def checkpoint(self, save_dir):
        if self._is_rank_0:
            super().checkpoint(save_dir)

    def _flush_row(self, eval_metrics: dict) -> None:
        if not self._is_rank_0 or self.metrics_path is None:
            return
        if self._last_train is None:
            return
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps({**self._last_train, **eval_metrics}) + "\n")
