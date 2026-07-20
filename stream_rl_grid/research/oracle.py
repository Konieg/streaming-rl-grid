"""Exact average-reward planning and policy evaluation for finite MDPs."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .environment import ContinuingGridMDP


@dataclass(frozen=True)
class AverageRewardSolution:
    gain: float
    bias: np.ndarray
    q_values: np.ndarray
    greedy_policy: np.ndarray
    iterations: int
    residual: float


def solve_average_reward(
    mdp: ContinuingGridMDP,
    tolerance: float = 1e-12,
    max_iterations: int = 200_000,
    reference_state: int = 0,
) -> AverageRewardSolution:
    """Solve a communicating finite MDP using relative value iteration."""
    rewards, transitions = mdp.expected_reward_and_transition()
    h = np.zeros(len(mdp.states), dtype=np.float64)
    gain = 0.0
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        raw_q = rewards + np.einsum("san,n->sa", transitions, h)
        unnormalized = raw_q.max(axis=1)
        gain = float(unnormalized[reference_state] - h[reference_state])
        new_h = unnormalized - unnormalized[reference_state]
        difference = new_h - h
        residual = float(difference.max() - difference.min())
        h = new_h
        if residual < tolerance:
            break
    else:
        raise RuntimeError("relative value iteration did not converge")
    q_values = rewards - gain + np.einsum("san,n->sa", transitions, h)
    greedy = np.argmax(q_values, axis=1).astype(np.int64)
    evaluated_gain, _ = evaluate_policy(mdp, deterministic_policy=greedy)
    return AverageRewardSolution(
        gain=float(evaluated_gain),
        bias=h,
        q_values=q_values,
        greedy_policy=greedy,
        iterations=iteration,
        residual=residual,
    )


def policy_matrix_from_q(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    n_states, n_actions = q_values.shape
    policy = np.full((n_states, n_actions), epsilon / n_actions, dtype=np.float64)
    for state, row in enumerate(q_values):
        best = np.flatnonzero(np.isclose(row, row.max(), rtol=1e-12, atol=1e-12))
        policy[state, best] += (1.0 - epsilon) / len(best)
    return policy


def evaluate_policy(
    mdp: ContinuingGridMDP,
    policy_matrix: Optional[np.ndarray] = None,
    deterministic_policy: Optional[np.ndarray] = None,
) -> tuple:
    rewards, transitions = mdp.expected_reward_and_transition()
    n_states, n_actions = rewards.shape
    if policy_matrix is None:
        if deterministic_policy is None:
            raise ValueError("a policy is required")
        policy_matrix = np.zeros((n_states, n_actions), dtype=np.float64)
        policy_matrix[np.arange(n_states), np.asarray(deterministic_policy, dtype=np.int64)] = 1.0
    policy_matrix = np.asarray(policy_matrix, dtype=np.float64)
    if policy_matrix.shape != (n_states, n_actions):
        raise ValueError("policy shape mismatch")
    transition_policy = np.einsum("sa,san->sn", policy_matrix, transitions)
    reward_policy = np.einsum("sa,sa->s", policy_matrix, rewards)
    # Solve pi P = pi and sum(pi)=1 directly.  Power iteration is simple but
    # becomes unnecessarily expensive for slowly mixing epsilon-greedy policies
    # when diagnostics are recorded every 100 real transitions.
    system = transition_policy.T - np.eye(n_states, dtype=np.float64)
    rhs = np.zeros(n_states, dtype=np.float64)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    try:
        distribution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        distribution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    distribution[np.abs(distribution) < 1e-14] = 0.0
    if np.min(distribution) < -1e-10:
        raise RuntimeError("fixed-policy stationary distribution is invalid")
    distribution = np.maximum(distribution, 0.0)
    distribution /= distribution.sum()
    return float(distribution @ reward_policy), distribution


def policy_goal_rate(mdp: ContinuingGridMDP, policy_matrix: np.ndarray) -> float:
    _, distribution = evaluate_policy(mdp, policy_matrix=policy_matrix)
    goal_probability = np.zeros((len(mdp.states), mdp.num_actions), dtype=np.float64)
    for state in range(len(mdp.states)):
        for action in range(mdp.num_actions):
            goal_probability[state, action] = sum(
                probability for probability, _, _, goal, _ in mdp.transition_distribution(state, action) if goal
            )
    return float(np.einsum("s,sa,sa->", distribution, policy_matrix, goal_probability))
