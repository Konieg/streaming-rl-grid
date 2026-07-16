import tempfile
import unittest

from stream_rl_grid.config import AppConfig
from stream_rl_grid.trainer import Trainer


class FeatureAblationContractTests(unittest.TestCase):
    def test_readonly_feature_evaluation_does_not_allocate_tiles(self):
        config = AppConfig()
        config.environment.obstacle_count = 0
        config.environment.context_maps = [[]]
        config.agent.iht_size = 1024
        config.training.auto_checkpoint_steps = 1_000_000
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, base_dir=directory)
            trainer.run_steps(10)
            used_before = len(trainer.coder.iht.dictionary)
            trainer.agent.action_values(trainer.environment.observation(), readonly=True)
            used_after = len(trainer.coder.iht.dictionary)

        self.assertEqual(used_after, used_before)


if __name__ == "__main__":
    unittest.main()
