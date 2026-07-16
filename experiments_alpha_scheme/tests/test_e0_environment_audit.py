import unittest

from experiments.tests.fixtures import make_environment


class EnvironmentAuditContractTests(unittest.TestCase):
    def test_seasonal_wind_requires_auto_direction_and_changes_phase(self):
        env = make_environment(
            "seasonal_wind",
            max_wind_strength=1,
            manual_wind_direction="auto",
            wind_period=1,
        )
        env.agent_state = (2, 2)
        wind_before = env.wind_vector(env.agent_state)
        _, _, _, _, info = env.step(4)
        wind_after = env.wind_vector(env.agent_state)

        self.assertNotEqual(wind_before, (0, 0))
        self.assertNotEqual(wind_after, (0, 0))
        self.assertNotEqual(wind_before, wind_after)
        self.assertIn("season:1", info["events"])

    def test_stationary_profile_does_not_advance_schedules(self):
        env = make_environment(
            "stationary", wind_period=1, target_move_interval=1, context_switch_interval=1
        )
        goal_before = env.goal
        env.step(4)
        self.assertEqual(env.wind_phase, 0)
        self.assertEqual(env.context_index, 0)
        self.assertEqual(env.goal, goal_before)


if __name__ == "__main__":
    unittest.main()
