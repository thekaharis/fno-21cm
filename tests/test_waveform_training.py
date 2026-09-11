"""Frozen weights/moments, phase transitions, warm starts and exact resume."""

import copy
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from util.neuralop_setup import prefer_local_neuralop
prefer_local_neuralop()
from neuralop.training.training_state import save_training_state
from learned_waveform_operator import LearnedWaveformOperator, waveform_parameter_groups
from modeling import TrainerModel
from training import MetricsTrainer, ContrastRefit
from waveform_training import WaveformTrainingConfig, WaveformTrainingController, warm_start


class SmallModel(nn.Module):
    def __init__(self, transform="tied"):
        super().__init__()
        self.spectral = LearnedWaveformOperator(2, 2, (3, 3), bins=7, init="sine", transform=transform)
        self.skip = nn.Conv2d(2, 2, 1)
        self.norm = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(.5)
        self.head = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        return self.head(self.dropout(self.norm(self.spectral(x) + self.skip(x))))


def mse(out, y, **_):
    return (out - y).square().mean()


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("mode,scope", [("waveform_only", "spectral"), ("kernel_only", "spectral"),
                                       ("alternating", "spectral"), ("alternating", "all"),
                                       ("joint_then_kernel", "spectral"), ("joint_then_kernel", "all")])
@pytest.mark.parametrize("transform", ["tied", "separate"])
def test_inactive_parameters_and_adam_moments_are_bitwise_frozen(mode, scope, transform):
    torch.manual_seed(31)
    model = SmallModel(transform).double()
    optimizer = torch.optim.AdamW(waveform_parameter_groups(model, lr=.01, weight_decay=.2))
    x, y = torch.randn(2, 2, 9, 9, dtype=torch.float64), torch.randn(2, 1, 9, 9, dtype=torch.float64)
    # Populate momentum for EVERY parameter before freezing. Starting with
    # empty state would miss momentum/weight-decay updates on inactive groups.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        mse(model(x), y).backward()
        optimizer.step()
    config = WaveformTrainingConfig(mode, waveform_epochs=1, kernel_epochs=1, kernel_scope=scope,
                                    adapt_epochs=2)
    controller = WaveformTrainingController(model, optimizer, config)
    seen_active_norms = []

    def clip(optim, args, kwargs):
        inactive = [p for p in model.parameters() if id(p) not in controller._active_ids]
        assert all(p.grad is None for p in inactive)
        seen_active_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), .1))
    optimizer.register_step_pre_hook(clip)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    for epoch in range(4):
        before = {n: p.clone() for n, p in model.named_parameters()}
        state_before = {n: copy.deepcopy(optimizer.state[p]) for n, p in model.named_parameters()}
        controller.begin_epoch(epoch)
        model.train()  # deliberately emulate the parent's train() call
        controller.prepare_batch()
        optimizer.zero_grad(set_to_none=True)
        mse(model(x), y).backward()
        optimizer.step()
        changed = []
        for name, p in model.named_parameters():
            is_table = ".tables." in name
            is_mixing = name in {"spectral.weight", "spectral.phase_weight"}
            active = (True if controller.phase == "joint" else is_table if controller.phase == "waveform"
                      else (not is_table if scope == "all" else is_mixing))
            if not active:
                torch.testing.assert_close(p, before[name], atol=0, rtol=0)
                assert_state_equal(optimizer.state[p], state_before[name])
            else:
                changed.append(not torch.equal(p, before[name]))
        assert any(changed)
        for name, b in model.named_buffers():
            torch.testing.assert_close(b, buffers[name], atol=0, rtol=0)
        metrics = controller.metrics(epoch)
        if controller.phase == "kernel":
            assert metrics["waveform_relative_update"] == 0
        else:
            assert metrics["waveform_relative_update"] > 0
    assert len(seen_active_norms) == 4


def test_joint_mode_matches_ordinary_adam_exactly():
    torch.manual_seed(5)
    model = SmallModel()
    other = copy.deepcopy(model)
    opt = torch.optim.Adam(waveform_parameter_groups(model, lr=.01, weight_decay=.1))
    ref = torch.optim.Adam(waveform_parameter_groups(other, lr=.01, weight_decay=.1))
    controller = WaveformTrainingController(model, opt)
    x, y = torch.randn(2, 2, 9, 9), torch.randn(2, 1, 9, 9)
    for epoch in range(2):
        controller.begin_epoch(epoch)
        for m, o in ((model, opt), (other, ref)):
            torch.manual_seed(epoch)
            o.zero_grad()
            mse(m(x), y).backward()
            o.step()
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_configuration_and_schedule(monkeypatch):
    monkeypatch.setenv("WAVEFORM_TRAINING_MODE", "alternating")
    monkeypatch.setenv("WAVEFORM_PHASE_EPOCHS", "2")
    monkeypatch.setenv("WAVEFORM_KERNEL_EPOCHS", "3")
    monkeypatch.setenv("WAVEFORM_FIRST_PHASE", "kernel")
    monkeypatch.setenv("WAVEFORM_KERNEL_SCOPE", "all")
    config = WaveformTrainingConfig.from_env()
    assert WaveformTrainingConfig(**config.to_dict()) == config
    assert [config.phase_at(e) for e in range(10)] == ["kernel"] * 3 + ["waveform"] * 2 + ["kernel"] * 3 + ["waveform"] * 2
    for kwargs in ({"mode": "bad"}, {"waveform_epochs": 0}, {"kernel_epochs": -1},
                   {"first_phase": "bad"}, {"kernel_scope": "bad"}):
        with pytest.raises(ValueError):
            WaveformTrainingConfig(**kwargs)


def test_adaptation_schedule_and_environment(monkeypatch):
    monkeypatch.setenv("WAVEFORM_TRAINING_MODE", "joint_then_kernel")
    monkeypatch.setenv("WAVEFORM_ADAPT_EPOCHS", "2")
    config = WaveformTrainingConfig.from_env()
    assert config.adapt_epochs == 2
    assert config.to_dict()["adapt_epochs"] == 2
    assert WaveformTrainingConfig(**config.to_dict()) == config
    assert [config.phase_at(e) for e in range(5)] == ["joint", "joint", "kernel", "kernel", "kernel"]
    assert WaveformTrainingConfig("joint_then_kernel", adapt_epochs=0).phase_at(0) == "kernel"
    assert config.phase_at(1000) == "kernel"
    with pytest.raises(ValueError, match="ADAPT_EPOCHS"):
        WaveformTrainingConfig("joint_then_kernel", adapt_epochs=-1)


@pytest.mark.parametrize("mode", ["joint", "waveform_only", "kernel_only", "alternating"])
def test_existing_checkpoint_config_remains_compatible(mode):
    config = WaveformTrainingConfig(mode)
    # Literal schema written before adapt_epochs existed.
    legacy = dict(mode=mode, waveform_epochs=1, kernel_epochs=5,
                  first_phase="waveform", kernel_scope="spectral")
    assert config.to_dict() == legacy
    model = SmallModel()
    optimizer = torch.optim.Adam(model.parameters())
    controller = WaveformTrainingController(model, optimizer, config)
    controller.begin_epoch(0)
    optimizer.param_groups[0][controller.state_key]["config"] = legacy
    controller.validate_resume(0)
    controller.begin_epoch(1)


def setup_run(config, epochs, metrics_path, transform="tied"):
    model = TrainerModel(SmallModel(transform)).double()
    opt = torch.optim.AdamW(waveform_parameter_groups(model, lr=.001, weight_decay=.1))
    controller = WaveformTrainingController(model, opt, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    trainer = MetricsTrainer(model=model, n_epochs=epochs, device="cpu", data_processor=None,
                             wandb_log=False, eval_interval=1, use_distributed=False, verbose=False,
                             metrics_path=metrics_path, waveform_training=controller)
    return model, opt, scheduler, trainer


def train(run, loader, *, resume=None):
    model, optimizer, scheduler, trainer = run
    trainer.train(train_loader=loader, test_loaders={"val": loader}, optimizer=optimizer,
                  scheduler=scheduler, regularizer=None, training_loss=mse, eval_losses={"mse": mse},
                  resume_from_dir=resume)


def test_trainer_resume_uses_manifest_model_and_restores_cycle_and_initial_tables(tmp_path):
    torch.manual_seed(44)
    loader = DataLoader([{"x": torch.randn(2, 9, 9, dtype=torch.float64),
                          "y": torch.randn(1, 9, 9, dtype=torch.float64)} for _ in range(2)], batch_size=2)
    config = WaveformTrainingConfig("alternating", waveform_epochs=2, kernel_epochs=1)
    torch.manual_seed(8)
    full = setup_run(config, 6, tmp_path / "full/metrics.jsonl")
    train(full, loader)
    torch.manual_seed(8)
    partial = setup_run(config, 4, tmp_path / "partial/metrics.jsonl")
    train(partial, loader)
    save_dir = tmp_path / "checkpoint"
    # An older best model must not override the final model named in the manifest.
    TrainerModel(SmallModel()).double().save_checkpoint(save_dir, "best_model")
    save_training_state(save_dir, "final_model", partial[0], partial[1], partial[2], epoch=3)
    resumed = setup_run(config, 6, tmp_path / "resumed/metrics.jsonl")
    train(resumed, loader, resume=save_dir)
    assert_state_equal(full[0].state_dict(), resumed[0].state_dict())
    assert_state_equal(full[1].state_dict(), resumed[1].state_dict())
    assert_state_equal(full[2].state_dict(), resumed[2].state_dict())
    rows = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [4, 5]
    assert [row["waveform_training_phase"] for row in rows] == ["waveform", "kernel"]
    assert rows[1]["waveform_relative_update"] == 0
    assert_state_equal(torch.load(tmp_path / "full/waveform_initial_tables.pt", weights_only=True),
                       torch.load(tmp_path / "resumed/waveform_initial_tables.pt", weights_only=True))
    incompatible = setup_run(WaveformTrainingConfig("waveform_only"), 6, None)
    with pytest.raises(ValueError, match="schedule differs"):
        train(incompatible, loader, resume=save_dir)


def test_checkpoint_warm_start_requires_compatible_model_and_starts_fresh(tmp_path):
    source = TrainerModel(SmallModel())
    path = tmp_path / "trained.pt"
    torch.save(source.state_dict(), path)
    destination = TrainerModel(SmallModel())
    report = warm_start(destination, path, strict=True)
    assert not report.missing and not report.unexpected
    assert_state_equal(source.state_dict(), destination.state_dict())
    with pytest.raises(ValueError, match="not both"):
        warm_start(destination, path, resume_dir=tmp_path)
    incomplete = dict(source.state_dict())
    incomplete.pop("fno.head.bias")
    torch.save(incomplete, path)
    with pytest.raises(ValueError, match="matching architecture"):
        warm_start(destination, path, strict=True)


@pytest.mark.parametrize("completed_epoch", [1, 2, 3])
@pytest.mark.parametrize("transform", ["tied", "separate"])
def test_adaptation_resume_before_at_and_after_freeze(tmp_path, completed_epoch, transform):
    torch.manual_seed(44)
    loader = DataLoader([{"x": torch.randn(2, 9, 9, dtype=torch.float64),
                          "y": torch.randn(1, 9, 9, dtype=torch.float64)} for _ in range(2)], batch_size=2)
    config = WaveformTrainingConfig("joint_then_kernel", adapt_epochs=3, kernel_scope="all")
    torch.manual_seed(8)
    full = setup_run(config, 6, tmp_path / "full/metrics.jsonl", transform)
    train(full, loader)
    torch.manual_seed(8)
    partial = setup_run(config, completed_epoch + 1, tmp_path / "partial/metrics.jsonl", transform)
    train(partial, loader)
    save_dir = tmp_path / "checkpoint"
    save_training_state(save_dir, "final_model", partial[0], partial[1], partial[2], epoch=completed_epoch)
    resumed = setup_run(config, 6, tmp_path / "resumed/metrics.jsonl", transform)
    train(resumed, loader, resume=save_dir)
    for a, b in zip(full[:3], resumed[:3]):
        assert_state_equal(a.state_dict(), b.state_dict())
    rows = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [r["waveform_training_phase"] for r in rows] == [config.phase_at(e) for e in range(completed_epoch+1, 6)]
    assert all(r["waveform_relative_update"] == 0 for r in rows if r["epoch"] >= 3)
    controller = resumed[3].waveform_training
    # One batch per epoch. Frozen tables retain three Adam updates; every
    # other parameter retains six. The scheduler finishes its original cycle.
    for p in resumed[0].parameters():
        assert resumed[1].state[p]["step"] == (3 if id(p) in controller.table_ids else 6)
    assert resumed[2].last_epoch == 6
    incompatible = setup_run(WaveformTrainingConfig("joint_then_kernel", adapt_epochs=4, kernel_scope="all"), 6, None, transform)
    with pytest.raises(ValueError, match="schedule differs"):
        train(incompatible, loader, resume=save_dir)


def test_no_waveforms_and_contrast_refit_are_rejected():
    ordinary = nn.Linear(2, 2)
    opt = torch.optim.Adam(ordinary.parameters())
    with pytest.raises(ValueError, match="waveform bank"):
        WaveformTrainingController(ordinary, opt, WaveformTrainingConfig("waveform_only"))
    model = TrainerModel(SmallModel())
    opt = torch.optim.Adam(model.parameters())
    controller = WaveformTrainingController(model, opt, WaveformTrainingConfig("waveform_only"))
    with pytest.raises(ValueError, match="contrast refitting"):
        MetricsTrainer(model=model, n_epochs=1, device="cpu", waveform_training=controller,
                       contrast=ContrastRefit())
