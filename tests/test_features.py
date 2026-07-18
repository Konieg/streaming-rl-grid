import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.algo import DifferentialQLearning, create_agent
from stream_rl_grid.config import AgentConfig, AppConfig
from stream_rl_grid.features import GridFeatureEncoder
from stream_rl_grid.trainer import Trainer


class HandcraftedFeatureTests(unittest.TestCase):
    observation = (1, 2, 4, 0, 0)

    def test_d55_matches_collaborator_definition_exactly(self):
        encoder = GridFeatureEncoder(5, 5)
        features = encoder.encode(self.observation, 3)

        raw = np.asarray([
            1.0, -0.5, 0.0, 0.25, 0.0, 0.0,
            -0.75, 0.5, 0.5625, 0.25, -0.375,
        ])
        expected = np.zeros(55)
        expected[33:44] = raw / np.linalg.norm(raw)

        np.testing.assert_array_equal(features, expected)
        self.assertEqual(features.shape, (55,))
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0)
        self.assertLessEqual(int(np.count_nonzero(features)), 11)

    def test_previous_action_is_intentionally_ignored(self):
        encoder = GridFeatureEncoder(5, 5)
        first = encoder.encode((1, 2, 4, 0, 0), 3)
        second = encoder.encode((1, 2, 4, 0, 4), 3)
        np.testing.assert_array_equal(first, second)

    def test_actions_use_disjoint_eleven_weight_blocks(self):
        encoder = GridFeatureEncoder(5, 5)
        first = encoder.encode(self.observation, 1)
        second = encoder.encode(self.observation, 2)
        self.assertFalse(np.any((first != 0.0) & (second != 0.0)))
        np.testing.assert_array_equal(first[11:22], second[22:33])

    def test_unit_norm_features_use_effective_step_without_tiling_division(self):
        config = AgentConfig(
            algorithm="q_learning",
            feature_representation="handcrafted_lfa",
            effective_initial_step=0.1,
            reward_rate_step=0.01,
        )
        encoder = GridFeatureEncoder(5, 5)
        agent = DifferentialQLearning(encoder, config)
        features = encoder.encode(self.observation, 0)

        delta = agent.update(self.observation, 0, 2.0, self.observation, 0)

        self.assertEqual(delta, 2.0)
        self.assertEqual(agent.alpha, 0.1)
        np.testing.assert_allclose(agent.weights, 0.2 * features)
        self.assertEqual(agent.reward_rate, 0.02)

    def test_d55_training_and_checkpoint_continue_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            config = AppConfig()
            config.agent.algorithm = "tidbd"
            config.agent.feature_representation = "handcrafted_lfa"
            config.environment.obstacle_coordinates = None
            config.training.ui_update_steps = 5
            config.training.auto_checkpoint_steps = 1_000_000
            trainer = Trainer(config, base_dir=folder, run_id="d55")
            trainer.run_steps(45)
            self.assertEqual(trainer.agent.weights.shape, (55,))
            self.assertEqual(trainer.snapshot()["feature_dimension"], 55)
            path = trainer.save(Path(folder) / "d55.pkl")

            trainer.run_steps(20)
            expected_weights = trainer.agent.weights.copy()
            expected_beta = trainer.agent.beta.copy()
            restored = Trainer.from_checkpoint(path, base_dir=folder)
            restored.run_steps(20)

            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.beta, expected_beta)

    def test_every_registered_algorithm_accepts_d55_features(self):
        state = (0, 0, 4, 4, 0)
        next_state = (1, 0, 4, 4, 1)
        for algorithm in ("q_learning", "q_lambda", "sarsa", "dyna_q", "tidbd"):
            with self.subTest(algorithm=algorithm):
                config = AgentConfig(
                    algorithm=algorithm,
                    feature_representation="handcrafted_lfa",
                    planning_steps=2,
                )
                encoder = GridFeatureEncoder(5, 5)
                agent = create_agent(encoder, config)
                next_action = 1 if agent.samples_next_action_before_update else None
                delta = agent.update(state, 0, -1.0, next_state, next_action)
                self.assertTrue(np.isfinite(delta))
                self.assertTrue(np.all(np.isfinite(agent.weights)))
                self.assertEqual(agent.weights.shape, (55,))


if __name__ == "__main__":
    unittest.main()
