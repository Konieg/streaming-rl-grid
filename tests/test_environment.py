import unittest

from stream_rl_grid.config import EnvironmentConfig, PROFILES
from stream_rl_grid.environment import ContinualWindyGridWorld


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **changes):
        config = EnvironmentConfig(
            width=5,
            height=5,
            obstacle_count=0,
            profile="stationary",
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

    def test_goal_restart_is_random_legal_and_profile_independent(self):
        obstacles = {(1, 1), (2, 2)}
        for profile in PROFILES:
            with self.subTest(profile=profile):
                env = self.make_env(
                    profile=profile,
                    obstacle_count=len(obstacles),
                    num_contexts=1,
                    context_maps=[[list(point) for point in sorted(obstacles)]],
                )
                samples = {env._restart_state() for _ in range(100)}
                self.assertGreater(len(samples), 1)
                self.assertTrue(samples.isdisjoint(obstacles))
                self.assertNotIn(env.goal, samples)

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
            profile="hidden_context",
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

    def test_default_environment_uses_configured_coordinates(self):
        config = EnvironmentConfig()
        env = ContinualWindyGridWorld(config)
        self.assertEqual(env.context_maps[0], {tuple(point) for point in config.obstacle_coordinates})
        self.assertEqual(env.start_position, tuple(config.start_position))
        self.assertEqual(env.goal, tuple(config.goal_position))

    def test_customize_profile_does_not_advance_automatic_schedules(self):
        env = self.make_env(profile="customize", wind_period=1, target_move_interval=1,
                            context_switch_interval=1, manual_wind_direction="none")
        env.agent_state = (1, 1)
        env.goal = (4, 4)
        env.step(4)
        self.assertEqual(env.wind_phase, 0)
        self.assertEqual(env.context_index, 0)
        self.assertEqual(env.goal, (4, 4))


if __name__ == "__main__":
    unittest.main()
