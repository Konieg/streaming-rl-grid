import math
import unittest

import numpy as np

from stream_rl_grid.environment import ContinualWindyGridWorld

from experiments.phase0.protocol import CONDITIONS, condition_by_name
from experiments.phase0.runner import run_one
from experiments.phase0.tabular import TabularDifferentialLearner, state_action


class Phase0Tests(unittest.TestCase):
    def test_state_action_ignores_previous_action(self) -> None:
        self.assertEqual(
            state_action((1, 2, 4, 0, 0), 3),
            state_action((1, 2, 4, 0, 4), 3),
        )

    def test_fixed_alpha_differential_update(self) -> None:
        learner = TabularDifferentialLearner(fixed_alpha=0.05)
        observation = (0, 4, 4, 0, 5)
        next_observation = (1, 4, 4, 0, 1)
        delta = learner.update(observation, 1, 2.0, next_observation, 1)
        self.assertEqual(delta, 2.0)
        self.assertEqual(learner.value(observation, 1), 0.1)
        self.assertEqual(learner.average_reward, 0.02)

    def test_adaptive_step_size_stays_bounded(self) -> None:
        learner = TabularDifferentialLearner(adaptive=True)
        observation = (0, 4, 4, 0, 5)
        next_observation = (1, 4, 4, 0, 1)
        for _ in range(200):
            learner.update(observation, 1, 10.0, next_observation, 1)
        diagnostics = learner.diagnostics()
        self.assertLessEqual(1e-4, diagnostics["alpha_min"])
        self.assertLessEqual(diagnostics["alpha_min"], diagnostics["alpha_max"])
        self.assertLessEqual(diagnostics["alpha_max"], 0.5)
        self.assertTrue(math.isfinite(diagnostics["q_max_abs"]))

    def test_all_frozen_environment_conditions_construct(self) -> None:
        for condition in CONDITIONS:
            environment = ContinualWindyGridWorld(condition.make_environment_config(seed=0))
            observation, _ = environment.reset(0)
            self.assertEqual(len(observation), 5)

    def test_moving_goal_and_hidden_context_recur(self) -> None:
        moving = ContinualWindyGridWorld(
            condition_by_name("moving_goal").make_environment_config(seed=0)
        )
        moving.reset(0)
        first_goal = moving.goal
        for _ in range(500):
            moving.step(4)
        second_goal = moving.goal
        for _ in range(500):
            moving.step(4)
        self.assertNotEqual(first_goal, second_goal)
        self.assertEqual(moving.goal, first_goal)

        hidden = ContinualWindyGridWorld(
            condition_by_name("hidden_context").make_environment_config(seed=0)
        )
        hidden.reset(0)
        first_map = hidden.active_obstacles
        for _ in range(500):
            hidden.step(4)
        second_map = hidden.active_obstacles
        for _ in range(500):
            hidden.step(4)
        self.assertNotEqual(first_map, second_map)
        self.assertEqual(hidden.active_obstacles, first_map)

    def test_prediction_stream_is_paired_across_step_sizes(self) -> None:
        condition = condition_by_name("seasonal_wind")
        _, slow = run_one(
            "prediction", condition, "slow", {"fixed_alpha": 0.01}, 3, 40
        )
        _, fast = run_one(
            "prediction", condition, "fast", {"fixed_alpha": 0.10}, 3, 40
        )
        np.testing.assert_array_equal(slow["action"], fast["action"])
        np.testing.assert_array_equal(slow["observation"], fast["observation"])

    def test_value_lookup_does_not_mutate_table(self) -> None:
        learner = TabularDifferentialLearner(fixed_alpha=0.05)
        self.assertEqual(learner.value((0, 4, 4, 0, 5), 1), 0.0)
        self.assertEqual(len(learner.q), 0)


if __name__ == "__main__":
    unittest.main()
