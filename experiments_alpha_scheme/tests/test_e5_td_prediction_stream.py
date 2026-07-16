import unittest

from experiments.tests.fixtures import fixed_actions, make_environment


class TDPredictionStreamContractTests(unittest.TestCase):
    def test_continuing_stream_has_no_terminal_boundary(self):
        env = make_environment("combined", obstacle_count=0, context_maps=[[], [], []], num_contexts=3,
                               max_wind_strength=1, manual_wind_direction="auto", wind_period=3,
                               target_move_interval=4, context_switch_interval=5)
        for action in fixed_actions(100):
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated)
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
