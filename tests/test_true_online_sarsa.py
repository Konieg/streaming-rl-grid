import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.algo import DifferentialSarsa, DifferentialTrueOnlineSarsa, create_agent
from stream_rl_grid.config import AppConfig
from stream_rl_grid.discrete_features import DiscreteStateActionFeatures
from stream_rl_grid.trainer import Trainer


class TrueOnlineSarsaTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.agent.algorithm = "true_online_sarsa"
        config.agent.use_tidbd = False
        config.agent.lambda_ = 0.6
        config.agent.effective_initial_step = 0.2
        config.agent.reward_rate_step = 0.05
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def test_factory_selects_registered_algorithm(self):
        config = self.config()
        features = DiscreteStateActionFeatures(config.environment)
        agent = create_agent(features, config.agent, seed=4)
        self.assertIsInstance(agent, DifferentialTrueOnlineSarsa)
        self.assertEqual(agent.algorithm_name, "true_online_sarsa")

    def test_dutch_trace_and_weight_correction_match_reference_update(self):
        config = self.config()
        features = DiscreteStateActionFeatures(config.environment)
        agent = DifferentialTrueOnlineSarsa(features, config.agent, seed=0)
        transitions = [
            ((0, 0, 4, 4, 5), 1, -1.0, (1, 0, 4, 4, 1), 1),
            ((1, 0, 4, 4, 1), 1, 2.0, (2, 0, 4, 4, 1), 2),
            ((2, 0, 4, 4, 1), 2, -0.5, (2, 1, 4, 4, 2), 3),
        ]
        for observation, action, _, next_observation, next_action in transitions:
            agent.weights[features.active(observation, action)] = 0.1 * (action + 1)
            agent.weights[features.active(next_observation, next_action)] = -0.05 * (next_action + 1)
        agent.reward_rate = 0.3

        expected_weights = agent.weights.copy()
        expected_trace = np.zeros(features.size, dtype=np.float64)
        expected_reward_rate = agent.reward_rate
        expected_q_old = 0.0
        alpha = config.agent.effective_initial_step
        lambda_ = config.agent.lambda_

        for observation, action, reward, next_observation, next_action in transitions:
            active = features.active(observation, action)
            next_active = features.active(next_observation, next_action)
            q = float(expected_weights[active].sum())
            q_next = float(expected_weights[next_active].sum())
            delta = reward - expected_reward_rate + q_next - q
            trace_dot = float(expected_trace[active].sum())
            expected_trace *= lambda_
            expected_trace[active] += 1.0 - alpha * lambda_ * trace_dot
            correction = q - expected_q_old
            expected_weights += alpha * (delta + correction) * expected_trace
            expected_weights[active] -= alpha * correction
            expected_q_old = q_next
            expected_reward_rate += config.agent.reward_rate_step * delta

            actual_delta = agent.update(
                observation, action, reward, next_observation, next_action
            )
            self.assertAlmostEqual(actual_delta, delta)
            np.testing.assert_allclose(agent.trace, expected_trace, rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(agent.weights, expected_weights, rtol=0.0, atol=1e-15)
            self.assertAlmostEqual(agent.q_old, expected_q_old)
            self.assertAlmostEqual(agent.reward_rate, expected_reward_rate)

    def test_lambda_zero_matches_one_step_differential_sarsa(self):
        true_config = self.config()
        true_config.agent.lambda_ = 0.0
        sarsa_config = self.config()
        sarsa_config.agent.algorithm = "sarsa"
        sarsa_config.agent.lambda_ = 0.0
        features = DiscreteStateActionFeatures(true_config.environment)
        true_online = DifferentialTrueOnlineSarsa(features, true_config.agent, seed=0)
        sarsa = DifferentialSarsa(features, sarsa_config.agent, seed=0)
        initial_weights = np.linspace(-0.2, 0.2, features.size)
        true_online.weights = initial_weights.copy()
        sarsa.weights = initial_weights.copy()
        true_online.reward_rate = sarsa.reward_rate = -0.15

        transitions = [
            ((0, 0, 4, 4, 5), 0, -1.0, (0, 0, 4, 4, 0), 1),
            ((0, 0, 4, 4, 0), 1, -1.0, (1, 0, 4, 4, 1), 1),
            ((1, 0, 4, 4, 1), 1, 10.0, (3, 2, 4, 4, 1), 4),
        ]
        for transition in transitions:
            self.assertAlmostEqual(true_online.update(*transition), sarsa.update(*transition))
            np.testing.assert_allclose(true_online.weights, sarsa.weights, rtol=0.0, atol=1e-15)
            self.assertAlmostEqual(true_online.reward_rate, sarsa.reward_rate)

    def test_checkpoint_restores_exact_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(47)
            path = trainer.save(Path(folder) / "true-online.pkl")
            trainer.run_steps(29)
            expected_weights = trainer.agent.weights.copy()
            expected_trace = trainer.agent.trace.copy()
            expected_q_old = trainer.agent.q_old

            restored = Trainer.from_checkpoint(path, base_dir=folder)
            restored.run_steps(29)
            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.trace, expected_trace)
            self.assertEqual(restored.agent.q_old, expected_q_old)
            self.assertEqual(restored.current_action, trainer.current_action)
            self.assertEqual(restored.environment.state_dict(), trainer.environment.state_dict())


if __name__ == "__main__":
    unittest.main()
