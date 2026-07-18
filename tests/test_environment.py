import unittest

from stream_rl_grid.config import EnvironmentConfig
from stream_rl_grid.environment import ContinualWindyGridWorld


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **changes):
        config = EnvironmentConfig(
            width=5,
            height=5,
            obstacle_count=0,
            w_strength=0.0,
            context_maps=[[]],
            seed=3,
        )
        for key, value in changes.items():
            setattr(config, key, value)
        return ContinualWindyGridWorld(config)

    def test_goal_is_rewarded_and_teleported_without_termination(self):
        env = self.make_env()
        env.goal = (2, 2)
        env.agent_state = (1, 2)
        observation, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(reward, env.config.reward_goal)
        self.assertTrue(info["goal_reached"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(env.agent_state, env.goal)
        self.assertEqual(observation[:4], env.observation()[:4])

    def test_default_goal_restart_is_random_and_legal(self):
        obstacles = {(1, 1), (2, 2)}
        env = self.make_env(
            obstacle_count=len(obstacles),
            context_maps=[[list(point) for point in sorted(obstacles)]],
        )
        self.assertEqual(env.config.goal_reached_behavior, "random_agent_restart")
        samples = {env._restart_state() for _ in range(100)}
        self.assertGreater(len(samples), 1)
        self.assertTrue(samples.isdisjoint(obstacles))
        self.assertNotIn(env.goal, samples)

    def test_relocate_target_keeps_agent_on_old_goal(self):
        obstacles = {(1, 1), (2, 2)}
        env = self.make_env(
            obstacle_count=len(obstacles),
            context_maps=[[list(point) for point in sorted(obstacles)]],
            goal_reached_behavior="relocate_target",
            target_move_interval=1,
        )
        old_goal = (3, 3)
        env.goal = old_goal
        env.agent_state = (2, 3)
        observation, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(reward, env.config.reward_goal)
        self.assertTrue(info["goal_reached"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.agent_state, old_goal)
        self.assertNotEqual(env.goal, old_goal)
        self.assertNotIn(env.goal, obstacles)
        self.assertEqual(tuple(observation[:2]), old_goal)
        self.assertEqual(tuple(observation[2:4]), env.goal)
        self.assertIn("target_relocated_after_goal", info["events"])
        self.assertNotIn("goal_moved", info["events"])

    def test_goal_moves_requires_random_agent_restart(self):
        with self.assertRaisesRegex(ValueError, "random_agent_restart"):
            self.make_env(
                goal_moves=True,
                goal_reached_behavior="relocate_target",
            )

    def test_invalid_action_stays_and_receives_collision_reward(self):
        env = self.make_env()
        env.agent_state = (0, 0)
        before = env.agent_state
        _, reward, terminated, _, info = env.step(3)
        self.assertEqual(env.agent_state, before)
        self.assertEqual(reward, env.config.reward_collision)
        self.assertTrue(info["collision"])
        self.assertFalse(terminated)

    def test_stay_action_is_still_affected_by_wind(self):
        env = self.make_env(w_strength=1.0, manual_wind_direction="up")
        env.agent_state = (2, 3)
        env.goal = (4, 4)
        env.step(4)
        self.assertEqual(env.agent_state, (2, 2))

    def test_new_context_obstacle_is_dormant_until_agent_leaves(self):
        env = self.make_env(
            obstacle_count=1,
            obstacle_switches=True,
            num_contexts=2,
            context_switch_interval=2,
            context_maps=[[[4, 0]], [[2, 2]]],
        )
        env.agent_state = (2, 2)
        env.goal = (4, 4)
        env.step(4)
        env.step(4)
        self.assertEqual(env.dormant_obstacle, (2, 2))
        self.assertNotIn((2, 2), env.active_obstacles)
        env.step(1)
        self.assertIsNone(env.dormant_obstacle)

    def test_manual_environment_update_is_immediate_and_persistent(self):
        env = self.make_env(w_strength=1.0)
        env.agent_state = (3, 3)
        env.previous_action = 2
        env.apply_manual_configuration({(2, 2)}, (0, 0), (4, 4), "right")
        self.assertEqual(env.active_obstacles, {(2, 2)})
        self.assertEqual(env.agent_state, (3, 3))
        self.assertEqual(env.previous_action, 2)
        self.assertEqual(env.start_position, (0, 0))
        self.assertEqual(env.goal, (4, 4))
        self.assertEqual(env.wind_vector((2, 2)), (1, 0))
        self.assertEqual(env.config.obstacle_count, 1)

    def test_live_obstacle_on_agent_cell_stays_dormant_until_departure(self):
        env = self.make_env(w_strength=0.0)
        env.agent_state = (2, 2)
        env.goal = (4, 4)
        env.apply_manual_configuration({(2, 2)}, (0, 0), (4, 4), "none")
        self.assertEqual(env.agent_state, (2, 2))
        self.assertEqual(env.dormant_obstacle, (2, 2))
        self.assertNotIn((2, 2), env.active_obstacles)
        env.step(1)
        self.assertIsNone(env.dormant_obstacle)
        self.assertIn((2, 2), env.active_obstacles)

    def test_wind_strength_is_a_displacement_probability(self):
        env = self.make_env(w_strength=0.3, manual_wind_direction="right")
        samples = [env.sample_wind((2, 2)) for _ in range(10_000)]
        rate = sum(sample == (1, 0) for sample in samples) / len(samples)
        self.assertGreater(rate, 0.27)
        self.assertLess(rate, 0.33)

    def test_wind_changes_override_fixed_direction_without_enabling_rewards(self):
        env = self.make_env(
            wind_changes=True,
            reward_changes=False,
            w_strength=1.0,
            manual_wind_direction="left",
            wind_period=1,
        )
        self.assertEqual(env.wind_vector((2, 2)), (0, -1))
        env.agent_state = (2, 2)
        env.goal = (4, 4)
        env.step(4)
        self.assertEqual(env.wind_phase, 1)
        self.assertEqual(env.reward_phase, 0)
        self.assertEqual(env.wind_vector((2, 2)), (1, 0))

    def test_obstacle_switching_generates_additional_context_maps(self):
        env = self.make_env(
            obstacle_count=2,
            obstacle_switches=True,
            num_contexts=3,
            context_maps=[[[0, 3], [2, 2]]],
        )
        self.assertEqual(len(env.context_maps), 3)
        self.assertEqual(env.context_maps[0], {(0, 3), (2, 2)})
        self.assertGreater(len({frozenset(layout) for layout in env.context_maps}), 1)
        self.assertTrue(all(env.goal not in layout for layout in env.context_maps))

    def test_obstacle_only_mode_rejects_a_map_covering_the_fixed_goal(self):
        with self.assertRaisesRegex(ValueError, "fixed goal"):
            self.make_env(
                obstacle_count=1,
                obstacle_switches=True,
                num_contexts=2,
                context_maps=[[[0, 0]], [[4, 4]]],
            )

    def test_default_environment_uses_configured_coordinates(self):
        config = EnvironmentConfig()
        env = ContinualWindyGridWorld(config)
        self.assertEqual(env.context_maps[0], {tuple(point) for point in config.obstacle_coordinates})
        self.assertEqual(env.start_position, tuple(config.start_position))
        self.assertEqual(env.goal, tuple(config.goal_position))

    def test_disabled_switches_do_not_advance_automatic_schedules(self):
        env = self.make_env(
            wind_period=1, reward_period=1, target_move_interval=1,
            context_switch_interval=1, manual_wind_direction="none",
        )
        env.agent_state = (1, 1)
        env.goal = (4, 4)
        env.step(4)
        self.assertEqual(env.wind_phase, 0)
        self.assertEqual(env.reward_phase, 0)
        self.assertEqual(env.context_index, 0)
        self.assertEqual(env.goal, (4, 4))

    def test_all_sixteen_nonstationarity_combinations_are_independent(self):
        for mask in range(16):
            wind_changes = bool(mask & 1)
            goal_moves = bool(mask & 2)
            obstacle_switches = bool(mask & 4)
            reward_changes = bool(mask & 8)
            with self.subTest(mask=mask):
                context_maps = [[], []] if obstacle_switches else [[]]
                env = self.make_env(
                    wind_changes=wind_changes,
                    goal_moves=goal_moves,
                    obstacle_switches=obstacle_switches,
                    reward_changes=reward_changes,
                    num_contexts=2,
                    context_maps=context_maps,
                    wind_period=1,
                    reward_period=1,
                    target_move_interval=1,
                    context_switch_interval=1,
                )
                env.agent_state = (1, 1)
                env.goal = (4, 4)
                old_goal = env.goal
                env.step(4)
                self.assertEqual(env.wind_phase == 1, wind_changes)
                self.assertEqual(env.reward_phase == 1, reward_changes)
                self.assertEqual(env.context_index == 1, obstacle_switches)
                self.assertEqual(env.goal != old_goal, goal_moves)

    def test_reward_changes_without_changing_transition_schedules(self):
        env = self.make_env(reward_changes=True, reward_period=1)
        env.agent_state = (1, 1)
        env.goal = (4, 4)
        env.step(4)
        _, reward, _, _, _ = env.step(4)
        self.assertEqual(reward, env.config.reward_step * 1.1)
        self.assertEqual(env.wind_phase, 0)
        self.assertEqual(env.context_index, 0)
        self.assertEqual(env.goal, (4, 4))


if __name__ == "__main__":
    unittest.main()
