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

    def test_ufno_construction_respects_external_seed(self):
        from modeling import build_3d_model

        torch.manual_seed(17)
        first = build_3d_model(ModelConfig(kind="ufno"), in_channels=2)
        first_weight = first.body.conv0.weights1.detach().clone()

        torch.manual_seed(17)
        second = build_3d_model(ModelConfig(kind="ufno"), in_channels=2)
        second_weight = second.body.conv0.weights1.detach().clone()

        self.assertTrue(torch.equal(first_weight, second_weight))

    def test_reads_sirenfno_environment(self):
        env = {
            "MODEL_KIND": "sirenfno",
            "N_MODES_Z": "24",
            "SIREN_HIDDEN_DIM": "48",
            "SIREN_FEATURE_DIM": "32",
            "SIREN_PADDING_Z": "12",
            "SIREN_LEARNABLE_FF": "false",
            "SIREN_OUTPUT_SIGMOID": "true",
            "SIREN_SIGMOID_TEMPERATURE": "1.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ModelConfig.from_env()
        self.assertEqual(config.kind, "sirenfno")
        self.assertEqual(config.modes, (16, 16, 24))
        self.assertEqual(config.siren_hidden_dim, 48)
        self.assertEqual(config.siren_feature_dim, 32)
        self.assertEqual(config.siren_padding, (0, 0, 12))
        self.assertFalse(config.siren_learnable_ff)
        self.assertTrue(config.siren_output_sigmoid)
        self.assertEqual(config.siren_sigmoid_temperature, 1.5)

    def test_legacy_siren_metadata_keeps_unconstrained_output(self):
        config = ModelConfig.from_dict(
            {
                "kind": "sirenfno",
                "modes": [16, 16, 16],
            }
        )
        self.assertFalse(config.siren_output_sigmoid)


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
