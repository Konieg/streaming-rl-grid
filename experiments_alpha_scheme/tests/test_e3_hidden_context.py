import unittest

from experiments.tests.fixtures import make_environment


class HiddenContextContractTests(unittest.TestCase):
    def test_identical_observation_can_have_different_transition_in_each_context(self):
        env = make_environment(
            "hidden_context",
            obstacle_count=1,
            num_contexts=2,
            context_maps=[[[1, 0]], [[0, 1]]],
        )

        def run_in_context(index):
            env.context_index = index
            env.agent_state = (0, 0)
            env.goal = (4, 4)
            env.previous_action = 5
            observation = env.observation()
            next_observation, reward, _, _, info = env.step(1)  # right
            return observation, next_observation, reward, info["collision"]

        first = run_in_context(0)
        second = run_in_context(1)

        self.assertEqual(first[0], second[0])
        self.assertTrue(first[3])
        self.assertFalse(second[3])
        self.assertNotEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
