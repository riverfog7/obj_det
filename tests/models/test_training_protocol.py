from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from obj_det.models.training import (
    CheckpointState,
    EarlyStoppingState,
    build_adamw_param_groups,
    optimizer_steps_per_epoch,
    require_metric,
    require_single_process,
    warmup_cosine_factor,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_shared_protocol_fails_explicitly_for_multi_process_execution(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(NotImplementedError, "one-writer"):
                require_single_process(context="controlled test")

    def test_hpo_and_final_schedules_match_for_first_ten_epochs(self):
        steps_per_epoch = optimizer_steps_per_epoch(101, 4)
        warmup_steps = steps_per_epoch
        total_steps = steps_per_epoch * 50

        hpo = [
            warmup_cosine_factor(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=0.01,
            )
            for step in range(steps_per_epoch * 10 + 1)
        ]
        final = [
            warmup_cosine_factor(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=0.01,
            )
            for step in range(steps_per_epoch * 50 + 1)
        ]

        self.assertEqual(hpo, final[: len(hpo)])
        self.assertEqual(hpo[warmup_steps], 1.0)
        self.assertAlmostEqual(final[-1], 0.01)
        self.assertGreater(hpo[-1], 0.01)

    def test_adamw_groups_share_bias_and_normalization_exclusions(self):
        class RMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))

        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.LayerNorm(4),
            RMSNorm(),
            nn.Linear(4, 2, bias=False),
        )
        groups = build_adamw_param_groups(model, weight_decay=1.0e-4)

        self.assertEqual([group["weight_decay"] for group in groups], [1.0e-4, 0.0])
        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertIn(id(model[0].weight), decay_ids)
        self.assertIn(id(model[3].weight), decay_ids)
        self.assertIn(id(model[0].bias), no_decay_ids)
        self.assertIn(id(model[1].weight), no_decay_ids)
        self.assertIn(id(model[1].bias), no_decay_ids)
        self.assertIn(id(model[2].weight), no_decay_ids)

    def test_early_stopping_min_epochs_patience_and_delta(self):
        cfg = SimpleNamespace(enabled=True, min_epochs=10, patience=8, min_delta=0.001)
        state = EarlyStoppingState()

        self.assertFalse(state.update(1, 0.2, cfg))
        for epoch in range(2, 9):
            self.assertFalse(state.update(epoch, 0.2005, cfg))
        self.assertFalse(state.update(9, 0.2005, cfg))
        self.assertTrue(state.update(10, 0.2005, cfg))
        self.assertEqual(state.best_epoch, 1)

    def test_checkpoint_state_distinguishes_best_and_last(self):
        cfg = SimpleNamespace(enabled=True, min_epochs=1, patience=8, min_delta=0.001)
        with TemporaryDirectory() as tmp:
            state = CheckpointState(Path(tmp))
            first = Path(tmp) / "epoch_001.pt"
            second = Path(tmp) / "epoch_002.pt"
            state.record_epoch(epoch=1, checkpoint_path=first, metric=0.3, early_stopping_cfg=cfg)
            state.record_epoch(epoch=2, checkpoint_path=second, metric=0.3005, early_stopping_cfg=cfg)

            self.assertEqual(state.best_checkpoint, second)
            self.assertEqual(state.last_checkpoint, second)
            self.assertEqual(state.best_epoch, 2)
            self.assertEqual(state.best_metric, 0.3005)
            self.assertEqual(state.early_stopping.best_epoch, 1)
            self.assertTrue(state.manifest_path.exists())

    def test_required_metric_has_no_fallback(self):
        self.assertEqual(require_metric({"map_50_95": 0.4}, "map_50_95", context="objective"), 0.4)
        with self.assertRaisesRegex(ValueError, "Missing required objective metric"):
            require_metric({"map_50": 0.5}, "map_50_95", context="objective")
        with self.assertRaisesRegex(ValueError, "not finite"):
            require_metric({"map_50_95": math.nan}, "map_50_95", context="objective")


if __name__ == "__main__":
    unittest.main()
