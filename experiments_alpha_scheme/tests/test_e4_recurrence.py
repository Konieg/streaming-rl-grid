import unittest

from experiments.tests.fixtures import make_environment


class RecurrenceContractTests(unittest.TestCase):
    def test_two_context_schedule_recurs_a_b_a(self):
        env = make_environment(
            "hidden_context",
            obstacle_count=1,
            num_contexts=2,
            context_switch_interval=1,
            context_maps=[[[4, 0]], [[0, 4]]],
        )
        observed_contexts = [env.context_index]
        for _ in range(2):
            env.step(4)
            observed_contexts.append(env.context_index)

        self.assertEqual(observed_contexts, [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
