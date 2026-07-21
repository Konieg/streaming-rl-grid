import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.algo import create_agent
from stream_rl_grid.benchmark import run_benchmark
from stream_rl_grid.cli import build_parser
from stream_rl_grid.config import AppConfig
from stream_rl_grid.factorized_features import SparseFactorizedStateActionFeatures
from stream_rl_grid.features import create_features
from stream_rl_grid.trainer import Trainer


class SparseFactorizedFeatureTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.environment.width = 8
        config.environment.height = 8
        config.environment.obstacle_count = 0
        config.environment.obstacle_coordinates = None
        config.environment.profile = "stationary"
        config.environment.start_position = [0, 0]
        config.environment.goal_position = [7, 7]
        config.agent.representation = "sparse-factorized"
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def test_parameter_count_and_disjoint_group_indices(self):
        config = self.config()
        features = SparseFactorizedStateActionFeatures(config.environment)
        self.assertEqual(features.nominal_active_count, 7)
        self.assertEqual(features.size, 1920)

        active = features.active((2, 3, 7, 6, 5), 4)
        self.assertEqual(len(active), 7)
        self.assertEqual(len(set(active.tolist())), 7)
        for index, offset, size in zip(active, features.group_offsets, features.group_sizes):
            self.assertGreaterEqual(index, offset)
            self.assertLess(index, offset + size)

    def test_related_states_share_only_the_expected_factors(self):
        features = SparseFactorizedStateActionFeatures(self.config().environment)
        first = set(features.active((1, 1, 4, 4, 2), 1).tolist())
        translated = set(features.active((2, 2, 5, 5, 2), 1).tolist())
        other_action = set(features.active((1, 1, 4, 4, 2), 3).tolist())

        # Bias, relative displacement, direction, distance, and previous action agree.
        self.assertEqual(len(first & translated), 5)
        self.assertEqual(first & other_action, set())

    def test_value_is_sum_of_active_factor_weights(self):
        config = self.config()
        features = create_features(config.environment, config.agent)
        agent = create_agent(features, config.agent, seed=0)
        observation = (1, 2, 7, 7, 5)
        active = features.active(observation, 2)
        agent.weights[active] = np.arange(1.0, 8.0)
        self.assertEqual(agent.value(observation, 2), 28.0)

    def test_effective_step_size_is_preserved_with_seven_active_features(self):
        config = self.config()
        config.agent.algorithm = "sarsa"
        config.agent.lambda_ = 0.0
        config.agent.effective_initial_step = 0.14
        features = create_features(config.environment, config.agent)
        agent = create_agent(features, config.agent, seed=0)
        observation = (0, 0, 7, 7, 5)
        next_observation = (1, 0, 7, 7, 1)

        delta = agent.update(observation, 1, 2.0, next_observation, 1)

        self.assertEqual(delta, 2.0)
        self.assertAlmostEqual(agent.alpha, 0.02)
        self.assertAlmostEqual(agent.value(observation, 1), 0.28)

    def test_all_model_free_algorithms_remain_finite(self):
        for algorithm in (
            "sarsa",
            "tidbd",
            "true_online_sarsa",
            "adaptive_epsilon_sarsa",
            "expected_sarsa",
            "expected_sarsa_tidbd",
        ):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as folder:
                config = self.config()
                config.agent.algorithm = algorithm
                config.agent.use_tidbd = algorithm in ("tidbd", "expected_sarsa_tidbd")
                trainer = Trainer(config, base_dir=folder)
                snapshot = trainer.run_steps(100)
                self.assertEqual(snapshot["representation"], "sparse-factorized")
                self.assertEqual(snapshot["q_parameter_count"], 1920)
                self.assertTrue(np.all(np.isfinite(trainer.agent.weights)))

    def test_checkpoint_restores_exact_factorized_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config()
            config.agent.algorithm = "sarsa"
            trainer = Trainer(config, base_dir=folder)
            trainer.run_steps(43)
            path = trainer.save(Path(folder) / "factorized.pkl")
            trainer.run_steps(27)
            expected_weights = trainer.agent.weights.copy()
            expected_trace = trainer.agent.trace.copy()

            restored = Trainer.from_checkpoint(path, base_dir=folder)
            restored.run_steps(27)
            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.trace, expected_trace)
            self.assertEqual(restored.current_action, trainer.current_action)
            self.assertEqual(restored.features.state_dict(), trainer.features.state_dict())

    def test_cli_accepts_factorized_representation(self):
        args = build_parser().parse_args(["--representation", "sparse-factorized"])
        self.assertEqual(args.representation, "sparse-factorized")

    def test_benchmark_records_selected_representation(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = run_benchmark(
                ["stationary"], [0], 5, Path(folder), representation="sparse-factorized"
            )
            contents = csv_path.read_text(encoding="utf-8")
            self.assertIn("representation", contents.splitlines()[0])
            self.assertIn("sparse-factorized", contents)


if __name__ == "__main__":
    unittest.main()
