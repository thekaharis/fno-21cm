"""Epoch-wise waveform/kernel optimization without changing DDP's graph.

Inactive gradients are set to None before clipping and optimizer.step. This
freezes both parameters and Adam state, including momentum and weight decay.
Autograd still traverses frozen operations so waveform gradients stay intact.
"""

from dataclasses import asdict, dataclass
import os

import torch

from learned_waveform_operator import LearnedWaveformOperator, WaveformBank


@dataclass(frozen=True)
class WaveformTrainingConfig:
    mode: str = "joint"
    waveform_epochs: int = 1
    kernel_epochs: int = 5
    first_phase: str = "waveform"
    kernel_scope: str = "spectral"
    adapt_epochs: int = 25

    def __post_init__(self):
        if self.mode not in {"joint", "waveform_only", "kernel_only", "alternating", "joint_then_kernel"}:
            raise ValueError("WAVEFORM_TRAINING_MODE must be joint, waveform_only, kernel_only, alternating, or joint_then_kernel")
        if self.adapt_epochs < 0:
            raise ValueError("WAVEFORM_ADAPT_EPOCHS must be nonnegative")
        if self.waveform_epochs < 1 or self.kernel_epochs < 1:
            raise ValueError("waveform/kernel phase epoch counts must be positive")
        if self.first_phase not in {"waveform", "kernel"}:
            raise ValueError("WAVEFORM_FIRST_PHASE must be waveform or kernel")
        if self.kernel_scope not in {"spectral", "all"}:
            raise ValueError("WAVEFORM_KERNEL_SCOPE must be spectral or all")

    @classmethod
    def from_env(cls):
        return cls(
            mode=os.environ.get("WAVEFORM_TRAINING_MODE", "joint").strip().lower(),
            waveform_epochs=int(os.environ.get("WAVEFORM_PHASE_EPOCHS", "1")),
            kernel_epochs=int(os.environ.get("WAVEFORM_KERNEL_EPOCHS", "5")),
            first_phase=os.environ.get("WAVEFORM_FIRST_PHASE", "waveform").strip().lower(),
            kernel_scope=os.environ.get("WAVEFORM_KERNEL_SCOPE", "spectral").strip().lower(),
            adapt_epochs=int(os.environ.get("WAVEFORM_ADAPT_EPOCHS", "25")),
        )

    def to_dict(self):
        config = asdict(self)
        # Older optimizer checkpoints compare this dictionary exactly. An
        # unused new field must not invalidate existing phase schedules.
        if self.mode != "joint_then_kernel":
            config.pop("adapt_epochs")
        return config

    def phase_at(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if self.mode == "joint_then_kernel":
            return "joint" if epoch < self.adapt_epochs else "kernel"
        if self.mode != "alternating":
            return {"joint": "joint", "waveform_only": "waveform", "kernel_only": "kernel"}[self.mode]
        position = epoch % (self.waveform_epochs + self.kernel_epochs)
        if self.first_phase == "waveform":
            return "waveform" if position < self.waveform_epochs else "kernel"
        return "kernel" if position < self.kernel_epochs else "waveform"


class WaveformTrainingController:
    """Register immediately after constructing the optimizer, before clipping.

    Schedule and actual initial tables live in optimizer group metadata, so
    existing training-state checkpoints save them together with Adam moments.
    Frozen requires_grad flags supplied by the caller are always respected.
    """

    state_key = "waveform_training"

    def __init__(self, model, optimizer, config=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or WaveformTrainingConfig()
        root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        self.tables = {}
        spectral_ids = set()
        for name, module in root.named_modules():
            if isinstance(module, WaveformBank):
                self.tables.update({f"{name}.{key}".lstrip("."): value
                                    for key, value in module.named_parameters() if value.requires_grad})
            if isinstance(module, LearnedWaveformOperator):
                spectral_ids.update(id(p) for p in module.parameters(recurse=False) if p.requires_grad)
        self.table_ids = {id(p) for p in self.tables.values()}
        self.parameters = [p for group in optimizer.param_groups for p in group["params"]]
        optimizer_ids = {id(p) for p in self.parameters}
        if not (self.table_ids | spectral_ids) <= optimizer_ids:
            raise ValueError("optimizer must contain all trainable waveform bins and mixing weights")
        self.kernel_ids = (spectral_ids if self.config.kernel_scope == "spectral" else
                           {id(p) for p in self.parameters if p.requires_grad} - self.table_ids)
        if self.config.mode != "joint" and not self.tables:
            raise ValueError("waveform training requires at least one trainable waveform bank")
        if self.config.mode in {"alternating", "kernel_only", "joint_then_kernel"} and not self.kernel_ids:
            raise ValueError("kernel phase has no trainable mixing weights")
        self.enabled = bool(self.tables)
        self.phase = None
        self._active_ids = set()
        self._before = {}
        self._last_grad_norm = None
        self._hook = optimizer.register_step_pre_hook(self._before_step) if self.enabled else None

    def validate_resume(self, completed_epoch):
        if not self.enabled:
            return
        state = self.optimizer.param_groups[0].get(self.state_key)
        if state is None:
            if self.config.mode != "joint":
                raise ValueError("checkpoint lacks waveform training state; use INIT_CHECKPOINT for a fresh phase schedule")
            return  # legacy joint training
        if state["config"] != self.config.to_dict():
            raise ValueError("waveform schedule differs from resumed checkpoint; use INIT_CHECKPOINT to change training mode")
        if state["epoch"] != completed_epoch:
            raise ValueError("checkpoint model epoch and waveform optimizer epoch disagree")

    def begin_epoch(self, epoch):
        if not self.enabled:
            return
        group = self.optimizer.param_groups[0]
        if self.state_key not in group:
            group[self.state_key] = {
                "config": self.config.to_dict(), "epoch": epoch,
                "reference_epoch": epoch,
                "initial_tables": {name: p.detach().cpu().clone() for name, p in self.tables.items()},
            }
        state = group[self.state_key]
        if state["config"] != self.config.to_dict():
            raise ValueError("waveform training configuration differs from saved optimizer state")
        state["epoch"] = epoch
        self.phase = self.config.phase_at(epoch)
        self._active_ids = (self.table_ids if self.phase == "waveform" else self.kernel_ids
                            if self.phase == "kernel" else {id(p) for p in self.parameters if p.requires_grad})
        self._before = {name: p.detach().clone() for name, p in self.tables.items()}
        self._last_grad_norm = None
        # Clear gradients at transitions even when an external training loop
        # did not use zero_grad(set_to_none=True).
        for p in self.parameters:
            p.grad = None

    def prepare_batch(self):
        if self.enabled and self.config.mode != "joint":
            # Eval mode freezes BatchNorm statistics and disables dropout;
            # it does NOT disable autograd or the trainable normalization affine weights.
            self.model.eval()

    def _before_step(self, optimizer, args, kwargs):
        if self.phase is None:
            raise RuntimeError("call begin_epoch before optimizing waveforms")
        gradients = [p.grad.detach().abs().square().sum() for p in self.tables.values() if p.grad is not None]
        self._last_grad_norm = torch.stack(gradients).sum().sqrt() if gradients else None
        for p in self.parameters:
            if id(p) not in self._active_ids:
                p.grad = None

    @property
    def initial_tables(self):
        return self.optimizer.param_groups[0][self.state_key]["initial_tables"]

    def metrics(self, epoch):
        if not self.enabled:
            return {}
        change = sum((p.detach() - self._before[name]).square().sum() for name, p in self.tables.items())
        scale = sum(p.square().sum() for p in self._before.values()).clamp_min(1e-30)
        initial_change = sum((p.detach() - self.initial_tables[name].to(p)).square().sum()
                             for name, p in self.tables.items())
        initial_scale = sum(p.square().sum() for p in self.initial_tables.values()).clamp_min(1e-30)
        result = {
            "waveform_training_mode": self.config.mode,
            "waveform_training_phase": self.phase,
            "waveform_training_cycle": epoch // (self.config.waveform_epochs + self.config.kernel_epochs)
                                       if self.config.mode == "alternating" else 0,
            "waveform_active_parameters": sum(p.numel() for p in self.parameters if id(p) in self._active_ids),
            "waveform_relative_update": float((change / scale).sqrt()),
            "waveform_distance_from_initial": float((initial_change / initial_scale.to(initial_change)).sqrt()),
        }
        if self._last_grad_norm is not None:
            result["waveform_last_grad_norm"] = float(self._last_grad_norm)
        return result


def warm_start(model, checkpoint, *, resume_dir=None, strict=False):
    """Shared weight-only start; checkpoint tensors always override init profiles."""
    if checkpoint and resume_dir:
        raise ValueError("choose INIT_CHECKPOINT (new optimizer/schedule) or RESUME_DIR (exact continuation), not both")
    if not checkpoint:
        return None
    from modeling import load_checkpoint
    report = load_checkpoint(model, checkpoint)
    if strict and (report.missing or report.unexpected):
        raise ValueError(f"waveform continuation requires a matching architecture: missing={report.missing}, unexpected={report.unexpected}")
    return report
