import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.config import AppConfig
from stream_rl_grid.dyna_config import DynaConfig
from stream_rl_grid.dyna_model import RecencySampleModel
from stream_rl_grid.dyna_trainer import DynaTrainer
from stream_rl_grid.trainer import Trainer


class RecencySampleModelTests(unittest.TestCase):
    def test_latest_sample_age_and_capacity_lru(self):
        model = RecencySampleModel(capacity=2, max_age=3)
        rng = np.random.default_rng(7)
        model.learn((0, 0, 2, 2, 5), 0, 1.0, (0, 1, 2, 2, 0), 1)
        model.learn((1, 0, 2, 2, 5), 1, 2.0, (2, 0, 2, 2, 1), 2)
        # Refresh the first key through planning, making the second key the LRU entry.
        for _ in range(20):
            sample = model.sample(2, rng)
            if sample is not None and sample[0] == (0, 0, 2, 2, 5):
                break
        model.learn((2, 0, 2, 2, 5), 2, 3.0, (2, 1, 2, 2, 2), 3)
        records = model.state_dict()["records_lru_order"]
        keys = {(tuple(record["observation"]), record["action"]) for record in records}
        self.assertIn(((0, 0, 2, 2, 5), 0), keys)
        self.assertNotIn(((1, 0, 2, 2, 5), 1), keys)
        self.assertEqual(model.statistics(7)["model_valid_size"], 0.0)

    def test_latest_real_transition_overwrites_old_sample(self):
        model = RecencySampleModel(capacity=10, max_age=20)
        observation = (0, 0, 2, 2, 5)
        model.learn(observation, 0, -1.0, (0, 1, 2, 2, 0), 1)
        model.learn(observation, 0, 4.0, (1, 0, 2, 2, 0), 5)
        sample = model.sample(5, np.random.default_rng(1))
        self.assertIsNotNone(sample)
        self.assertEqual(sample[2], 4.0)
        self.assertEqual(sample[3], (1, 0, 2, 2, 0))
        self.assertEqual(sample[4], 0)


class DynaTrainerTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.environment.width = 6
        config.environment.height = 5
        config.environment.obstacle_count = 2
        config.environment.obstacle_coordinates = None
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.training.metric_window = 50
        config.training.ui_update_steps = 5
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def test_planning_does_not_change_real_trace_or_reward_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = DynaTrainer(self.config(), DynaConfig(planning_steps=0), base_dir=folder)
            trainer.run_steps(5)
            trace = trainer.agent.trace.copy()
            reward_rate = trainer.agent.reward_rate
            sample = trainer.model.sample(trainer.step_count, trainer.planning_rng)
            self.assertIsNotNone(sample)
            observation, action, reward, next_observation, _ = sample
            next_action = trainer.agent.select_planning_action(next_observation, trainer.planning_rng)
            trainer.agent.planning_update(observation, action, reward, next_observation, next_action)
            np.testing.assert_array_equal(trainer.agent.trace, trace)
            self.assertEqual(trainer.agent.reward_rate, reward_rate)

    def test_zero_planning_matches_original_real_learning_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Trainer(self.config(), base_dir=folder, run_id="base")
            dyna = DynaTrainer(
                self.config(), DynaConfig(planning_steps=0), base_dir=folder, run_id="dyna"
            )
            for _ in range(80):
                base.step_once(with_snapshot=False)
                dyna.step_once(with_snapshot=False)
            self.assertEqual(base.environment.state_dict(), dyna.environment.state_dict())
            self.assertEqual(base.current_observation, dyna.current_observation)
            self.assertEqual(base.current_action, dyna.current_action)
            np.testing.assert_array_equal(base.agent.weights, dyna.agent.weights)
            np.testing.assert_array_equal(base.agent.beta, dyna.agent.beta)
            np.testing.assert_array_equal(base.agent.h, dyna.agent.h)
            np.testing.assert_array_equal(base.agent.trace, dyna.agent.trace)
            self.assertEqual(base.agent.reward_rate, dyna.agent.reward_rate)

    def test_checkpoint_restores_model_and_planning_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = DynaTrainer(
                self.config(),
                DynaConfig(planning_steps=3, model_capacity=100, model_max_age=20),
                base_dir=folder,
            )
            trainer.run_steps(47)
            path = trainer.save(Path(folder) / "dyna-exact.pkl")
            trainer.run_steps(29)
            expected_environment = trainer.environment.state_dict()
            expected_agent = trainer.agent.state_dict()
            expected_model = trainer.model.state_dict()
            expected_action = trainer.current_action

            restored = DynaTrainer.from_checkpoint(path, base_dir=folder)
            restored.run_steps(29)
            self.assertEqual(restored.environment.state_dict(), expected_environment)
            self.assertEqual(restored.current_action, expected_action)
            np.testing.assert_array_equal(restored.agent.weights, expected_agent["weights"])
            np.testing.assert_array_equal(restored.agent.beta, expected_agent["beta"])
            np.testing.assert_array_equal(restored.agent.h, expected_agent["h"])
            np.testing.assert_array_equal(restored.agent.trace, expected_agent["trace"])
            self.assertEqual(restored.agent.reward_rate, expected_agent["reward_rate"])
            self.assertEqual(restored.model.state_dict(), expected_model)

    def test_checkpoint_types_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Trainer(self.config(), base_dir=folder, run_id="base")
            base_path = base.save(Path(folder) / "base.pkl")
            with self.assertRaisesRegex(ValueError, "model-free"):
                DynaTrainer.from_checkpoint(base_path, base_dir=folder)

            dyna = DynaTrainer(self.config(), DynaConfig(), base_dir=folder, run_id="dyna")
            dyna_path = dyna.save(Path(folder) / "dyna.pkl")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                Trainer.from_checkpoint(dyna_path, base_dir=folder)


if __name__ == "__main__":
    unittest.main()
