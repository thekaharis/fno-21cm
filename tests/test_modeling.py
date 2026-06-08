from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from losses import WeightedLoss
from modeling import ModelConfig, TrainerModel, load_checkpoint


class TrainerModelTests(unittest.TestCase):
    def test_forwards_x_and_ignores_sample_metadata(self):
        model = TrainerModel(nn.Identity())
        x = torch.randn(2, 3)
        self.assertTrue(torch.equal(model(x=x, y=torch.zeros_like(x)), x))

    def test_loads_supported_checkpoint_prefixes(self):
        reference = TrainerModel(nn.Linear(3, 2))
        wrapped_state = reference.state_dict()
        raw_state = {
            key.removeprefix("fno."): value
            for key, value in wrapped_state.items()
        }
        variants = (
            wrapped_state,
            raw_state,
            {f"module.{key}": value for key, value in wrapped_state.items()},
            {f"module.{key}": value for key, value in raw_state.items()},
        )

        for state_dict in variants:
            with self.subTest(keys=tuple(state_dict)):
                target = TrainerModel(nn.Linear(3, 2))
                with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
                    torch.save(state_dict, checkpoint.name)
                    report = load_checkpoint(target, checkpoint.name)
                self.assertEqual(report.matched, report.total)
                for key, value in wrapped_state.items():
                    self.assertTrue(torch.equal(target.state_dict()[key], value))


class ModelConfigTests(unittest.TestCase):
    def test_reads_experiment_environment(self):
        env = {
            "MODEL_KIND": "ufno",
            "N_MODES_Z": "32",
            "UFNO_NORM": "groupnorm",
            "UFNO_UNET_VARIANT": "los1d",
            "UFNO_GLOBAL_RESIDUAL": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ModelConfig.from_env()
        self.assertEqual(config.kind, "ufno")
        self.assertEqual(config.modes, (16, 16, 32))
        self.assertTrue(config.ufno_global_residual)


class WeightedLossTests(unittest.TestCase):
    def test_skips_disabled_terms(self):
        def disabled(*_args, **_kwargs):
            raise AssertionError("zero-weight term should not run")

        loss = WeightedLoss((1.0, lambda out, y, **_: (out - y).abs().mean()),
                            (0.0, disabled))
        value = loss(torch.tensor([2.0]), torch.tensor([1.0]))
        self.assertEqual(value.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
