"""Tabular differential TD(0)/Sarsa(0) learner for Phase 0.

This implementation is independent of the repository's existing algorithm modules.
"""

import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Tuple

import numpy as np

from .protocol import (
    ALPHA_MAX,
    ALPHA_MIN,
    INITIAL_ADAPTIVE_ALPHA,
    META_STEP,
    REWARD_RATE_STEP,
)


Observation = Tuple[int, int, int, int, int]
StateAction = Tuple[int, int, int, int, int]


def state_action(observation: Observation, action: int) -> StateAction:
    """Use (agent x/y, goal x/y, action); deliberately ignore previous_action."""
    x, y, goal_x, goal_y, _ = observation
    return int(x), int(y), int(goal_x), int(goal_y), int(action)


class TabularDifferentialLearner:
    """Differential one-step learner with fixed or per-table-entry step sizes."""

    def __init__(self, fixed_alpha: float = None, adaptive: bool = False):
        if adaptive == (fixed_alpha is not None):
            raise ValueError("Choose exactly one of fixed_alpha or adaptive=True.")
        if fixed_alpha is not None and fixed_alpha <= 0.0:
            raise ValueError("fixed_alpha must be positive.")
        self.fixed_alpha = fixed_alpha
        self.adaptive = adaptive
        self.average_reward = 0.0
        self.q: DefaultDict[StateAction, float] = defaultdict(float)
        self._initial_beta = math.log(INITIAL_ADAPTIVE_ALPHA)
        self.beta: DefaultDict[StateAction, float] = defaultdict(lambda: self._initial_beta)
        self.h: DefaultDict[StateAction, float] = defaultdict(float)

    def value(self, observation: Observation, action: int) -> float:
        return self.q.get(state_action(observation, action), 0.0)

    def values(self, observation: Observation, num_actions: int) -> np.ndarray:
        return np.asarray([self.value(observation, action) for action in range(num_actions)])

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        next_action: int,
    ) -> float:
        current = state_action(observation, action)
        successor = state_action(next_observation, next_action)
        delta = reward - self.average_reward + self.q[successor] - self.q[current]
        self.average_reward += REWARD_RATE_STEP * delta

        if self.adaptive:
            beta = self.beta[current] + META_STEP * delta * self.h[current]
            beta = min(math.log(ALPHA_MAX), max(math.log(ALPHA_MIN), beta))
            self.beta[current] = beta
            alpha = math.exp(beta)
            self.q[current] += alpha * delta

            decay = max(0.0, 1.0 - alpha)
            self.h[current] = self.h[current] * decay + alpha * delta
        else:
            self.q[current] += float(self.fixed_alpha) * delta

        if not math.isfinite(self.q[current]) or abs(self.q[current]) > 1e6:
            raise FloatingPointError("Tabular value became numerically unstable.")
        return float(delta)

    def alpha_values(self) -> Iterable[float]:
        if self.adaptive:
            keys = set(self.q) | set(self.beta)
            return (math.exp(self.beta[key]) for key in keys)
        return (float(self.fixed_alpha),)

    def diagnostics(self) -> Dict[str, float]:
        q_values = np.asarray(list(self.q.values()) or [0.0], dtype=float)
        alphas = np.asarray(list(self.alpha_values()), dtype=float)
        return {
            "average_reward_estimate": float(self.average_reward),
            "num_table_entries": int(len(self.q)),
            "q_l2_norm": float(np.linalg.norm(q_values)),
            "q_max_abs": float(np.max(np.abs(q_values))),
            "alpha_p10": float(np.percentile(alphas, 10)),
            "alpha_median": float(np.median(alphas)),
            "alpha_p90": float(np.percentile(alphas, 90)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
        }
