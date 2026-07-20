import unittest

import numpy as np

from stream_rl_grid.research.agents import DifferentialLinearAgent, ResearchAgentConfig
from stream_rl_grid.research.environment import ContinuingGridMDP, StationaryGridSpec, stationary_ladder
from stream_rl_grid.research.models import EmpiricalModel, OracleModel
from stream_rl_grid.research.oracle import evaluate_policy, solve_average_reward
from stream_rl_grid.research.representations import FeatureConfig, MultiGroupTileCoder
from stream_rl_grid.research.adaptive_models import FactoredGridModel
from stream_rl_grid.research.continual_environment import (
    DynamicContext, DynamicTwoGoalGrid, FrozenDynamicMDP, phase6_scenarios,
)


class ResearchEnvironmentTests(unittest.TestCase):
    def test_goal_is_a_continuing_transition(self):
        mdp = ContinuingGridMDP(StationaryGridSpec("tiny", 3, 3, (), (2, 2)), seed=1)
        state = mdp.state_to_index[(1, 2)]
        mdp.reset(state_index=state)
        next_state, reward, terminated, truncated, info = mdp.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["goal_reached"])
        self.assertEqual(reward, 20.0)
        self.assertIn(next_state, range(len(mdp.states)))

    def test_enumerator_and_sampler_probabilities_match(self):
        mdp = ContinuingGridMDP(stationary_ladder()[2], seed=3)
        state = mdp.state_to_index[(1, 1)]
        outcomes = mdp.transition_distribution(state, 4)
        self.assertAlmostEqual(sum(row[0] for row in outcomes), 1.0)
        self.assertEqual({round(row[0], 8) for row in outcomes}, {0.2, 0.8})


class ResearchOracleTests(unittest.TestCase):
    def test_average_reward_solution_is_self_consistent(self):
        mdp = ContinuingGridMDP(stationary_ladder()[0])
        solution = solve_average_reward(mdp)
        evaluated, distribution = evaluate_policy(mdp, deterministic_policy=solution.greedy_policy)
        self.assertAlmostEqual(solution.gain, evaluated, places=9)
        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        self.assertGreater(solution.gain, mdp.spec.reward_step)


class ResearchFeatureAndAgentTests(unittest.TestCase):
    def make_coder(self, mdp):
        coder = MultiGroupTileCoder(mdp.spec.width, mdp.spec.height, FeatureConfig(iht_size=4096))
        observations = [mdp.observation(state) for state in range(len(mdp.states))]
        coder.preallocate(observations)
        return coder, observations

    def test_multi_group_features_have_expected_groups(self):
        mdp = ContinuingGridMDP(stationary_ladder()[1])
        coder, observations = self.make_coder(mdp)
        indices = coder.active(observations[0], 0, readonly=True)
        self.assertEqual(len(indices), coder.nominal_active_count)
        changed = list(observations[0])
        changed[4] ^= 1
        changed_indices = coder.active(changed, 0, readonly=False)
        self.assertGreater(len(set(indices) ^ set(changed_indices)), 0)
        self.assertEqual(coder.iht.overfull_count, 0)

    def test_planning_zero_is_identical_to_q_learning(self):
        mdp = ContinuingGridMDP(stationary_ladder()[0])
        coder_a, observations = self.make_coder(mdp)
        coder_b, _ = self.make_coder(mdp)
        q = DifferentialLinearAgent(
            coder_a, observations, ResearchAgentConfig("q_learning"), seed=2,
        )
        dyna = DifferentialLinearAgent(
            coder_b, observations,
            ResearchAgentConfig("dyna", planning_steps=0), seed=2,
            model=EmpiricalModel(seed=4),
        )
        transitions = [(0, 1, -1.0, 1), (1, 2, -1.0, 2), (2, 1, 20.0, 0)]
        for transition in transitions:
            q.update_real(*transition)
            dyna.update_real(*transition)
        np.testing.assert_array_equal(q.weights, dyna.weights)
        self.assertEqual(q.reward_rate, dyna.reward_rate)

    def test_oracle_model_distribution_matches_mdp(self):
        mdp = ContinuingGridMDP(stationary_ladder()[2])
        model = OracleModel(mdp)
        for key in ((0, 0), (2, 4), (len(mdp.states) - 1, 3)):
            self.assertAlmostEqual(sum(model.distribution(key).values()), 1.0)


class PhaseSixToNineTests(unittest.TestCase):
    def test_two_goal_environment_remains_continuing(self):
        environment = DynamicTwoGoalGrid(seed=2)
        environment.state_index = environment.state_to_index[(5, 1)]
        next_state, reward, terminated, truncated, info = environment.step(
            1, DynamicContext("test", reward_a=6.0, reward_b=3.0)
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 6.0)
        self.assertEqual(info["goal_id"], "A")
        self.assertIn(next_state, environment.restart_states)

    def test_dynamic_transition_probabilities_sum_to_one(self):
        environment = DynamicTwoGoalGrid()
        context = DynamicContext(
            "mixed", wind_direction="down", wind_probability=0.4,
            upper_block_probability=0.3, lower_block_probability=0.7,
        )
        for state in range(len(environment.states)):
            for action in range(environment.num_actions):
                outcomes = environment.transition_distribution(state, action, context)
                self.assertAlmostEqual(sum(row[0] for row in outcomes), 1.0)

    def test_abrupt_scenarios_are_policy_relevant(self):
        environment = DynamicTwoGoalGrid()
        for scenario in phase6_scenarios()[:2]:
            old = solve_average_reward(
                FrozenDynamicMDP(environment, scenario.context_at(0))
            )
            new = solve_average_reward(
                FrozenDynamicMDP(environment, scenario.context_at(scenario.change_steps[0]))
            )
            self.assertGreater(
                float(np.mean(old.greedy_policy != new.greedy_policy)), 0.20
            )

    def test_factored_model_is_normalized_and_tracks_reward_head(self):
        environment = DynamicTwoGoalGrid()
        model = FactoredGridModel(environment, learning_rate=0.1, seed=4)
        before = model.reward_a
        state = environment.state_to_index[(5, 1)]
        model.observe_with_info(
            state, 1, 6.0, environment.restart_states[0],
            {"goal_id": "A", "goal_reached": True, "collision": False},
        )
        self.assertGreater(model.reward_a, before)
        self.assertAlmostEqual(sum(model.distribution((state, 1)).values()), 1.0)


if __name__ == "__main__":
    unittest.main()
