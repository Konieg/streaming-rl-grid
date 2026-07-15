import unittest

import numpy as np

from stream_rl_grid.algo import (
    DifferentialDynaQ,
    DifferentialQLambda,
    DifferentialQLearning,
    DifferentialSarsa,
)
from stream_rl_grid.config import AgentConfig


class StubCoder:
    size = 8
    nominal_active_count = 1

    def __init__(self):
        self.indices = {}

    def active(self, observation, action, readonly=False):
        del readonly
        key = (tuple(observation), int(action))
        if key not in self.indices:
            self.indices[key] = len(self.indices)
        return np.asarray([self.indices[key]], dtype=np.int64)

    def state_dict(self):
        return {"indices": self.indices.copy()}

    def load_state_dict(self, state):
        self.indices = state["indices"].copy()


class DifferentialAlgorithmTests(unittest.TestCase):
    state = (0, 0, 1, 1, 0)
    next_state = (1, 0, 1, 1, 1)

    def config(self, algorithm):
        config = AgentConfig(
            algorithm=algorithm,
            effective_initial_step=1.0,
            reward_rate_step=0.01,
            planning_steps=0,
        )
        return config

    def initialized_agent(self, agent_type, algorithm):
        coder = StubCoder()
        agent = agent_type(coder, self.config(algorithm), num_actions=2)
        current = coder.active(self.state, 0)[0]
        next_behavior = coder.active(self.next_state, 0)[0]
        next_greedy = coder.active(self.next_state, 1)[0]
        agent.weights[current] = 1.0
        agent.weights[next_behavior] = 2.0
        agent.weights[next_greedy] = 4.0
        return agent, current

    def test_q_learning_bootstraps_from_greedy_action(self):
        agent, current = self.initialized_agent(DifferentialQLearning, "q_learning")
        delta = agent.update(self.state, 0, 3.0, self.next_state, 0)
        self.assertEqual(delta, 6.0)
        self.assertEqual(agent.weights[current], 7.0)
        self.assertEqual(agent.reward_rate, 0.06)

    def test_sarsa_bootstraps_from_behavior_action(self):
        agent, current = self.initialized_agent(DifferentialSarsa, "sarsa")
        delta = agent.update(self.state, 0, 3.0, self.next_state, 0)
        self.assertEqual(delta, 4.0)
        self.assertEqual(agent.weights[current], 5.0)
        self.assertEqual(agent.reward_rate, 0.04)

    def test_watkins_q_lambda_cuts_trace_after_exploratory_action(self):
        agent, _ = self.initialized_agent(DifferentialQLambda, "q_lambda")
        delta = agent.update(self.state, 0, 3.0, self.next_state, 0)
        self.assertEqual(delta, 6.0)
        np.testing.assert_array_equal(agent.trace, np.zeros(agent.coder.size))
        self.assertEqual(agent.trace_cut_count, 1)

    def test_behavior_action_sampling_order_matches_update_type(self):
        for agent_type, algorithm, expected in (
            (DifferentialQLearning, "q_learning", ["update", "select"]),
            (DifferentialSarsa, "sarsa", ["select", "update"]),
        ):
            with self.subTest(algorithm=algorithm):
                agent, _ = self.initialized_agent(agent_type, algorithm)
                events = []
                original_update = agent.update

                def update(*args, **kwargs):
                    events.append("update")
                    return original_update(*args, **kwargs)

                def select_action(observation):
                    del observation
                    events.append("select")
                    return 0

                agent.update = update
                agent.select_action = select_action
                agent.learn_and_select_next(
                    self.state, 0, 3.0, self.next_state
                )
                self.assertEqual(events, expected)

    def test_dyna_q_updates_reward_rate_only_from_real_transition(self):
        agent, _ = self.initialized_agent(DifferentialDynaQ, "dyna_q")
        agent.config.planning_steps = 3
        delta = agent.update(self.state, 0, 3.0, self.next_state, 0)
        self.assertEqual(delta, 6.0)
        self.assertEqual(agent.reward_rate, 0.06)
        self.assertEqual(agent.update_count, 1)
        self.assertEqual(agent.planning_update_count, 3)
        self.assertEqual(len(agent.model), 1)


if __name__ == "__main__":
    unittest.main()
