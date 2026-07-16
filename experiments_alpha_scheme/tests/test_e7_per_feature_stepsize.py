import tempfile
import unittest

import numpy as np

from stream_rl_grid.config import AppConfig
from stream_rl_grid.trainer import Trainer


class PerFeatureStepSizeContractTests(unittest.TestCase):
    def test_tidbd_has_one_finite_positive_step_size_per_parameter(self):
        config = AppConfig()
        config.environment.obstacle_count = 0
        config.environment.context_maps = [[]]
        config.agent.iht_size = 1024
        config.training.auto_checkpoint_steps = 1_000_000
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, base_dir=directory)
            trainer.run_steps(30)
            alphas = np.exp(trainer.agent.beta)

        self.assertEqual(alphas.shape, trainer.agent.weights.shape)
        self.assertTrue(np.all(np.isfinite(alphas)))
        self.assertTrue(np.all(alphas > 0.0))


if __name__ == "__main__":
    unittest.main()
