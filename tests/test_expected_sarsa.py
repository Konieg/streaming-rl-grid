import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.algo import (
    DifferentialExpectedSarsa,
    DifferentialExpectedSarsaTIDBD,
    create_agent,
)
from stream_rl_grid.cli import build_parser
from stream_rl_grid.config import ALGORITHMS, AppConfig
from stream_rl_grid.discrete_features import DiscreteStateActionFeatures
from stream_rl_grid.trainer import Trainer


class ExpectedSarsaTests(unittest.TestCase):
    def config(self, algorithm="expected_sarsa"):
        config = AppConfig()
        config.agent.algorithm = algorithm
        config.agent.use_tidbd = algorithm == "expected_sarsa_tidbd"
        config.agent.lambda_ = 0.6
        config.agent.epsilon = 0.2
        config.agent.theta = 0.01
        config.agent.effective_initial_step = 0.2
        config.agent.reward_rate_step = 0.05
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    @staticmethod
    def transition():
        return (
            (0, 0, 4, 4, 5),
            1,
            -1.0,
            (1, 0, 4, 4, 1),
            0,
        )

    def initialized_agent(self, algorithm):
        config = self.config(algorithm)
        features = DiscreteStateActionFeatures(config.environment)
        agent = create_agent(features, config.agent, seed=3)
        observation, action, _, next_observation, _ = self.transition()
        agent.weights[features.active(observation, action)] = 0.4
        for candidate, value in enumerate((-2.0, -1.0, 0.0, 1.0, 3.0)):
            agent.weights[features.active(next_observation, candidate)] = value
        agent.reward_rate = 0.3
        return config, features, agent

    def test_factory_and_cli_register_both_algorithms(self):
        expected = {
            "expected_sarsa": DifferentialExpectedSarsa,
            "expected_sarsa_tidbd": DifferentialExpectedSarsaTIDBD,
        }
        for algorithm, agent_type in expected.items():
            with self.subTest(algorithm=algorithm):
                config = self.config(algorithm)
                features = DiscreteStateActionFeatures(config.environment)
                agent = create_agent(features, config.agent, seed=3)
                self.assertIsInstance(agent, agent_type)
                self.assertEqual(agent.algorithm_name, algorithm)
                self.assertIn(algorithm, ALGORITHMS)
                self.assertEqual(
                    build_parser().parse_args(["--algorithm", algorithm]).algorithm,
                    algorithm,
                )

    def test_fixed_step_update_uses_epsilon_greedy_expectation(self):
        config, features, agent = self.initialized_agent("expected_sarsa")
        observation, action, reward, next_observation, next_action = self.transition()
        old_trace_index = features.active((2, 2, 4, 4, 3), 2)[0]
        agent.trace[old_trace_index] = 0.5

        values = np.asarray([-2.0, -1.0, 0.0, 1.0, 3.0])
        probabilities = np.asarray([0.04, 0.04, 0.04, 0.04, 0.84])
        expected_bootstrap = float(np.dot(probabilities, values))
        expected_delta = reward - 0.3 + expected_bootstrap - 0.4
        expected_trace = agent.trace.copy() * config.agent.lambda_
        expected_trace[features.active(observation, action)] = 1.0
        expected_weights = agent.weights + agent.alpha * expected_delta * expected_trace

        actual_delta = agent.update(
            observation, action, reward, next_observation, next_action
        )

        self.assertAlmostEqual(expected_bootstrap, 2.44)
        self.assertAlmostEqual(actual_delta, expected_delta)
        self.assertNotAlmostEqual(actual_delta, reward - 0.3 + values[next_action] - 0.4)
        np.testing.assert_allclose(agent.trace, expected_trace, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(agent.weights, expected_weights, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(
            agent.reward_rate, 0.3 + config.agent.reward_rate_step * expected_delta
        )

    def test_tidbd_update_uses_expected_target_and_reference_meta_update(self):
        config, features, agent = self.initialized_agent("expected_sarsa_tidbd")
        observation, action, reward, next_observation, next_action = self.transition()
        active = features.active(observation, action)
        old_trace_index = features.active((2, 2, 4, 4, 3), 2)[0]
        agent.trace[old_trace_index] = 0.5
        agent.h[active] = 0.25
        agent.h[old_trace_index] = -0.1

        expected_delta = reward - 0.3 + 2.44 - 0.4
        expected_trace = agent.trace.copy() * config.agent.lambda_
        expected_trace[active] = 1.0
        expected_beta = agent.beta.copy()
        proposed = expected_beta[active] + config.agent.theta * expected_delta * agent.h[active]
        expected_beta[active] = np.clip(
            proposed, config.agent.beta_min, config.agent.beta_max
        )
        expected_alpha = np.exp(expected_beta)
        expected_weights = agent.weights + expected_alpha * expected_delta * expected_trace
        expected_h = agent.h.copy()
        decay = np.ones_like(expected_h)
        decay[active] = np.maximum(0.0, 1.0 - expected_alpha[active] * expected_trace[active])
        expected_h = expected_h * decay + expected_alpha * expected_delta * expected_trace

        actual_delta = agent.update(
            observation, action, reward, next_observation, next_action
        )

        self.assertAlmostEqual(actual_delta, expected_delta)
        np.testing.assert_allclose(agent.trace, expected_trace, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(agent.beta, expected_beta, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(agent.weights, expected_weights, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(agent.h, expected_h, rtol=0.0, atol=1e-15)

    def test_both_algorithms_restore_exact_continuation(self):
        for algorithm in ("expected_sarsa", "expected_sarsa_tidbd"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as folder:
                trainer = Trainer(self.config(algorithm), base_dir=folder)
                trainer.run_steps(47)
                path = trainer.save(Path(folder) / (algorithm + ".pkl"))
                trainer.run_steps(29)

                restored = Trainer.from_checkpoint(path, base_dir=folder)
                restored.run_steps(29)

                np.testing.assert_array_equal(restored.agent.weights, trainer.agent.weights)
                np.testing.assert_array_equal(restored.agent.trace, trainer.agent.trace)
                if algorithm == "expected_sarsa_tidbd":
                    np.testing.assert_array_equal(restored.agent.beta, trainer.agent.beta)
                    np.testing.assert_array_equal(restored.agent.h, trainer.agent.h)
                self.assertEqual(restored.current_action, trainer.current_action)
                self.assertEqual(
                    restored.environment.state_dict(), trainer.environment.state_dict()
                )


if __name__ == "__main__":
    unittest.main()
