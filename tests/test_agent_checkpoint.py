import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.config import AppConfig
from stream_rl_grid.trainer import Trainer


class AgentAndCheckpointTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.agent.algorithm = "tidbd"
        config.environment.width = 6
        config.environment.height = 5
        config.environment.obstacle_count = 2
        config.environment.obstacle_coordinates = None
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.agent.iht_size = 4096
        config.training.metric_window = 50
        config.training.ui_update_steps = 5
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def test_tile_coder_and_tidbd_remain_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(200)
            self.assertTrue(np.all(np.isfinite(trainer.agent.weights)))
            self.assertTrue(np.all(np.isfinite(trainer.agent.beta)))
            self.assertEqual(trainer.step_count, 200)

    def test_checkpoint_continues_exactly_from_next_action(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(83)
            path = trainer.save(Path(folder) / "exact.pkl")
            expected = []
            for _ in range(31):
                trainer.step_once()
                expected.append((trainer.environment.observation(), trainer.current_action, trainer.last_reward))
            expected_weights = trainer.agent.weights.copy()
            expected_beta = trainer.agent.beta.copy()

            restored = Trainer.from_checkpoint(path, base_dir=folder)
            actual = []
            for _ in range(31):
                restored.step_once()
                actual.append((restored.environment.observation(), restored.current_action, restored.last_reward))
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.beta, expected_beta)
            self.assertEqual(restored.environment.state_dict()["rng_state"], trainer.environment.state_dict()["rng_state"])

    def test_policy_snapshot_is_normalized_without_allocating_visualization_features(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(10)
            used_before = len(trainer.coder.iht.dictionary)
            snapshot = trainer.snapshot()
            used_after = len(trainer.coder.iht.dictionary)
            self.assertEqual(used_after, used_before)
            for row in snapshot["policy_probabilities"]:
                for probabilities in row:
                    if probabilities is not None:
                        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_frozen_policy_matrix_does_not_depend_on_latest_previous_action(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(80)
            first = trainer.snapshot()["policy_probabilities"]
            trainer.environment.previous_action = (trainer.environment.previous_action + 1) % 5
            second = trainer.snapshot()["policy_probabilities"]
            self.assertEqual(first, second)

    def test_frozen_position_policy_mixes_conditional_probabilities_not_q_values(self):
        with tempfile.TemporaryDirectory() as folder:
            config = AppConfig()
            config.agent.algorithm = "sarsa"
            config.training.auto_checkpoint_steps = 1_000_000
            trainer = Trainer(config, base_dir=folder)
            trainer.run_steps(1_000, with_snapshot=False)
            gx, gy = trainer.environment.goal
            grouped = {}
            for observation, count in trainer.agent.observation_counts.items():
                if observation[2:4] == (gx, gy):
                    grouped.setdefault(observation[:2], []).append((observation, count))
            position, conditional = next(
                (position, entries) for position, entries in grouped.items() if len(entries) >= 2
            )
            expected = np.average(
                [trainer.agent.action_probabilities(observation) for observation, _ in conditional],
                weights=[count for _, count in conditional], axis=0,
            )
            matrix = trainer.agent.freeze_policy_matrix(
                trainer.environment.width, trainer.environment.height, trainer.environment.goal
            )
            np.testing.assert_allclose(matrix[position[1], position[0]], expected)

    def test_unseen_policy_states_start_uniform(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            snapshot = trainer.snapshot()
            unseen = snapshot["policy_probabilities"][0][1]
            if unseen is not None:
                np.testing.assert_allclose(unseen, np.full(5, 0.2))

    def test_fixed_step_algorithms_use_shared_interface_and_restore_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            for algorithm in ("q_learning", "q_lambda", "sarsa", "dyna_q"):
                with self.subTest(algorithm=algorithm):
                    config = self.config()
                    config.agent.algorithm = algorithm
                    config.agent.planning_steps = 3
                    trainer = Trainer(config, base_dir=folder, run_id=algorithm)
                    trainer.run_steps(35)
                    self.assertEqual(trainer.agent.algorithm_name, algorithm)
                    self.assertEqual(trainer.snapshot()["algorithm"], algorithm)
                    path = trainer.save(Path(folder) / (algorithm + ".pkl"))
                    trainer.run_steps(20)
                    expected_weights = trainer.agent.weights.copy()
                    expected_reward_rate = trainer.agent.reward_rate
                    expected_agent_state = trainer.agent.state_dict()

                    restored = Trainer.from_checkpoint(path, base_dir=folder)
                    restored.run_steps(20)
                    np.testing.assert_array_equal(restored.agent.weights, expected_weights)
                    self.assertEqual(restored.agent.reward_rate, expected_reward_rate)
                    self.assertEqual(
                        restored.agent.state_dict().get("planning_update_count"),
                        expected_agent_state.get("planning_update_count"),
                    )
                    self.assertEqual(
                        restored.agent.state_dict().get("model"),
                        expected_agent_state.get("model"),
                    )

    def test_legacy_tidbd_flag_maps_to_tidbd_algorithm(self):
        restored = AppConfig.from_dict({"agent": {"use_tidbd": True}})
        self.assertEqual(restored.agent.algorithm, "tidbd")
        restored = AppConfig.from_dict({"agent": {"use_tidbd": False}})
        self.assertEqual(restored.agent.algorithm, "sarsa")

    def test_trainer_applies_live_environment_without_resetting_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(20)
            weights = trainer.agent.weights.copy()
            position = trainer.environment.agent_state
            previous_action = trainer.environment.previous_action
            current_action = trainer.current_action
            rng_state = trainer.agent.rng.bit_generator.state
            snapshot = trainer.apply_environment_configuration({(1, 1), (2, 1)}, (0, 0), (5, 4), "none")
            np.testing.assert_array_equal(trainer.agent.weights, weights)
            self.assertEqual(snapshot["agent_state"], position)
            self.assertEqual(trainer.environment.previous_action, previous_action)
            self.assertEqual(trainer.current_action, current_action)
            self.assertEqual(trainer.agent.rng.bit_generator.state, rng_state)
            self.assertEqual(snapshot["goal"], (5, 4))
            self.assertEqual(snapshot["manual_wind_direction"], "none")

    def test_live_wind_change_does_not_relocate_agent(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(20)
            position = trainer.environment.agent_state
            snapshot = trainer.apply_wind("left", 0.3)
            self.assertEqual(trainer.environment.agent_state, position)
            self.assertEqual(snapshot["manual_wind_direction"], "left")
            self.assertEqual(trainer.environment.config.w_strength, 0.3)

    def test_environment_apply_preserves_policy_when_policy_state_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(40)
            before = trainer.snapshot()
            layout = set(trainer.environment.context_maps[trainer.environment.context_index])
            start = next(
                (x, y) for y in range(trainer.environment.height) for x in range(trainer.environment.width)
                if (x, y) not in layout and (x, y) != trainer.environment.goal
            )
            after = trainer.apply_environment_configuration(
                layout, start, trainer.environment.goal, "none"
            )
            self.assertEqual(after["policy_probabilities"], before["policy_probabilities"])


if __name__ == "__main__":
    unittest.main()
