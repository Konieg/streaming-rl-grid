import unittest

from experiments.tests.fixtures import fixed_actions, make_environment, transition_stream


class StationaryPredictionContractTests(unittest.TestCase):
    def test_fixed_behaviour_stream_is_reproducible(self):
        actions = fixed_actions(40)
        first = transition_stream(make_environment("stationary"), actions)
        second = transition_stream(make_environment("stationary"), actions)

        self.assertEqual(first, second)
        self.assertTrue(all(not terminated and not truncated for _, _, terminated, truncated, _ in first))


if __name__ == "__main__":
    unittest.main()
