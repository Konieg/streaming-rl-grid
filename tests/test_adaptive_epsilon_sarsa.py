import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.algo import DifferentialAdaptiveEpsilonSarsa, create_agent
from stream_rl_grid.cli import build_parser
from stream_rl_grid.config import AppConfig
from stream_rl_grid.discrete_features import DiscreteStateActionFeatures
from stream_rl_grid.trainer import Trainer


class AdaptiveEpsilonSarsaTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.agent.algorithm = "adaptive_epsilon_sarsa"
        config.agent.use_tidbd = False
        config.agent.adaptive_epsilon_kappa = 0.5
        config.agent.adaptive_epsilon_min = 0.05
        config.agent.adaptive_epsilon_max = 0.40
        config.agent.adaptive_epsilon_scale = 0.20
        config.agent.adaptive_epsilon_u_ref = 1.0
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def agent(self):
        config = self.config()
        features = DiscreteStateActionFeatures(config.environment)
        return DifferentialAdaptiveEpsilonSarsa(features, config.agent, seed=2), features

    def test_factory_and_initial_exploration_state(self):
        config = self.config()
        features = DiscreteStateActionFeatures(config.environment)
        agent = create_agent(features, config.agent, seed=2)
        self.assertIsInstance(agent, DifferentialAdaptiveEpsilonSarsa)
        self.assertEqual(agent.td_error_magnitude, config.agent.adaptive_epsilon_u_ref)
        self.assertEqual(agent.epsilon, config.agent.adaptive_epsilon_min)

    def test_td_error_anomaly_raises_then_decays_epsilon(self):
        agent, _ = self.agent()
        observation = (0, 0, 4, 4, 5)
        action = 1

        reward_for_delta_five = agent.reward_rate + 5.0
        delta = agent.update(
            observation, action, reward_for_delta_five, observation, action
        )
        self.assertAlmostEqual(delta, 5.0)
        self.assertAlmostEqual(agent.td_error_magnitude, 3.0)
        self.assertAlmostEqual(agent.epsilon, 0.40)

        reward_for_zero_delta = agent.reward_rate
        agent.update(observation, action, reward_for_zero_delta, observation, action)
        self.assertAlmostEqual(agent.td_error_magnitude, 1.5)
        self.assertAlmostEqual(agent.epsilon, 0.15)

        reward_for_zero_delta = agent.reward_rate
        agent.update(observation, action, reward_for_zero_delta, observation, action)
        self.assertAlmostEqual(agent.td_error_magnitude, 0.75)
        self.assertAlmostEqual(agent.epsilon, 0.05)

    def test_policy_probabilities_use_current_adaptive_epsilon(self):
        agent, features = self.agent()
        observation = (1, 1, 4, 4, 5)
        agent.weights[features.active(observation, 3)] = 2.0
        agent.current_epsilon = 0.25

        probabilities = agent.action_probabilities(observation)

        np.testing.assert_allclose(probabilities, [0.05, 0.05, 0.05, 0.80, 0.05])

    def test_checkpoint_restores_adaptive_state_and_exact_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(53)
            path = trainer.save(Path(folder) / "adaptive-epsilon.pkl")
            trainer.run_steps(31)
            expected_weights = trainer.agent.weights.copy()
            expected_trace = trainer.agent.trace.copy()
            expected_u = trainer.agent.td_error_magnitude
            expected_epsilon = trainer.agent.epsilon

            restored = Trainer.from_checkpoint(path, base_dir=folder)
            restored.run_steps(31)
            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.trace, expected_trace)
            self.assertEqual(restored.agent.td_error_magnitude, expected_u)
            self.assertEqual(restored.agent.epsilon, expected_epsilon)
            self.assertEqual(restored.current_action, trainer.current_action)
            self.assertEqual(restored.environment.state_dict(), trainer.environment.state_dict())

    def test_snapshot_curves_record_epsilon_and_smoothed_td_error(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config()
            config.training.ui_update_steps = 5
            trainer = Trainer(config, base_dir=folder)
            snapshot = trainer.run_steps(15)
            curves = snapshot["curves"]
            self.assertEqual(len(curves["steps"]), 3)
            self.assertEqual(len(curves["epsilon"]), 3)
            self.assertEqual(len(curves["td_error_magnitude"]), 3)
            self.assertTrue(all(config.agent.adaptive_epsilon_min <= value <= config.agent.adaptive_epsilon_max
                                for value in curves["epsilon"]))
            self.assertTrue(all(np.isfinite(value) for value in curves["td_error_magnitude"]))

    def test_cli_accepts_adaptive_algorithm_and_hyperparameters(self):
        args = build_parser().parse_args(
            [
                "--algorithm", "adaptive_epsilon_sarsa",
                "--epsilon-kappa", "0.2",
                "--epsilon-min", "0.03",
                "--epsilon-max", "0.5",
                "--epsilon-scale", "0.4",
                "--epsilon-u-ref", "2.5",
            ]
        )
        self.assertEqual(args.algorithm, "adaptive_epsilon_sarsa")
        self.assertEqual(args.epsilon_kappa, 0.2)
        self.assertEqual(args.epsilon_min, 0.03)
        self.assertEqual(args.epsilon_max, 0.5)
        self.assertEqual(args.epsilon_scale, 0.4)
        self.assertEqual(args.epsilon_u_ref, 2.5)


if __name__ == "__main__":
    unittest.main()
