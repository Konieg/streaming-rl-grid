import unittest

from experiments.tests.fixtures import make_environment


class ObservableChangeContractTests(unittest.TestCase):
    def test_goal_move_changes_the_observed_goal_coordinates(self):
        env = make_environment("moving_goal", target_move_interval=1)
        before = env.observation()
        after, _, _, _, info = env.step(4)

        self.assertIn("goal_moved", info["events"])
        self.assertNotEqual(before[2:4], after[2:4])
        self.assertEqual(after[2:4], env.goal)


if __name__ == "__main__":
    unittest.main()
