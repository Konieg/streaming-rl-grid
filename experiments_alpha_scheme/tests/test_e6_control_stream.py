import tempfile
import unittest

from stream_rl_grid.config import AppConfig
from stream_rl_grid.trainer import Trainer


class ControlStreamContractTests(unittest.TestCase):
    def test_differential_sarsa_updates_online_without_ending_task(self):
        config = AppConfig()
        config.environment.width = 5
        config.environment.height = 5
        config.environment.obstacle_count = 0
        config.environment.profile = "stationary"
        config.environment.context_maps = [[]]
        config.agent.iht_size = 1024
        config.agent.use_tidbd = False
        config.training.auto_checkpoint_steps = 1_000_000
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, base_dir=directory)
            trainer.run_steps(25)
            self.assertEqual(trainer.step_count, 25)
            self.assertEqual(trainer.agent.update_count, 25)


if __name__ == "__main__":
    unittest.main()
