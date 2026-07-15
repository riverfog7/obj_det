from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from obj_det.models.data.loader import seed_dataloader_worker
from obj_det.models.utils.repro import set_seed


class ReproducibilityTest(unittest.TestCase):
    def test_deterministic_seed_enables_warn_only_algorithms(self):
        algorithms_enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        try:
            set_seed(7, deterministic=True)

            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(algorithms_enabled, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark

    def test_worker_seed_controls_python_and_numpy(self):
        with patch("obj_det.models.data.loader.torch.initial_seed", return_value=123):
            seed_dataloader_worker(0)
            first = (random.random(), float(np.random.random()))
            seed_dataloader_worker(1)
            second = (random.random(), float(np.random.random()))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
