import unittest

from stream_rl_grid.cli import build_parser


class CliConfigurationTests(unittest.TestCase):
    def test_nonstationarity_switches_are_disabled_by_default(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.wind_changes)
        self.assertFalse(args.goal_moves)
        self.assertFalse(args.obstacle_switches)
        self.assertFalse(args.reward_changes)

    def test_nonstationarity_switches_can_be_combined_freely(self):
        args = build_parser().parse_args([
            "--wind-changes", "--obstacle-switches", "--reward-changes",
            "--wind-period", "17", "--reward-period", "23",
        ])
        self.assertTrue(args.wind_changes)
        self.assertFalse(args.goal_moves)
        self.assertTrue(args.obstacle_switches)
        self.assertTrue(args.reward_changes)
        self.assertEqual(args.wind_period, 17)
        self.assertEqual(args.reward_period, 23)


if __name__ == "__main__":
    unittest.main()
