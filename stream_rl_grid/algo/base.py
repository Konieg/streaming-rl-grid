"""Common interface and policy utilities for grid-control algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from ..config import AgentConfig
from ..tile_coder import DualTileCoder


StateKey = Tuple[int, int, int, int]


class BaseControlAgent(ABC):
    """Unified interface used by Trainer, checkpoints, and the GUI."""

    algorithm_name = "base"

    def __init__(self, coder: DualTileCoder, config: AgentConfig, seed: int = 0):
        self.coder = coder
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(coder.size, dtype=np.float64)
        self.reward_rate = 0.0
        self.update_count = 0
        self.last_delta = 0.0
        self.visited_observations = set()
        self.policy_probability_matrix = None
        self.policy_goal = None

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    def value(self, observation: Sequence[int], action: int, readonly: bool = False) -> float:
        active = self.coder.active(observation, action, readonly=readonly)
        return float(self.weights[active].sum())

    def action_values(self, observation: Sequence[int], readonly: bool = False) -> np.ndarray:
        return np.asarray(
            [self.value(observation, action, readonly=readonly) for action in range(5)], dtype=np.float64
        )

    def probabilities_from_values(self, values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))
        probabilities = np.full(5, self.config.epsilon / 5.0, dtype=np.float64)
        probabilities[best] += (1.0 - self.config.epsilon) / len(best)
        return probabilities

    def action_probabilities(self, observation: Sequence[int], readonly: bool = True) -> np.ndarray:
        return self.probabilities_from_values(self.action_values(observation, readonly=readonly))

    def select_action(self, observation: Sequence[int]) -> int:
        observation = tuple(int(value) for value in observation)
        self.visited_observations.add(observation)
        probabilities = self.action_probabilities(observation, readonly=False)
        return int(self.rng.choice(5, p=probabilities))

    def freeze_policy_matrix(self, width: int, height: int, goal: Sequence[int]) -> np.ndarray:
        """Freeze the current epsilon-greedy evaluation policy for every grid position.

        The environment includes previous_action in its Markov state. For a stable 2-D GUI
        view, values are averaged over the previous-action values actually observed at each
        (position, goal) state. Completely unseen states retain the uniform random policy.
        """
        gx, gy = int(goal[0]), int(goal[1])
        previous_by_state: Dict[StateKey, list] = {}
        for x, y, seen_gx, seen_gy, previous_action in self.visited_observations:
            previous_by_state.setdefault((x, y, seen_gx, seen_gy), []).append(previous_action)

        matrix = np.full((height, width, 5), 0.2, dtype=np.float64)
        for y in range(height):
            for x in range(width):
                previous_actions = previous_by_state.get((x, y, gx, gy), [])
                if not previous_actions:
                    continue
                values = np.mean(
                    [self.action_values((x, y, gx, gy, previous), readonly=True)
                     for previous in sorted(set(previous_actions))],
                    axis=0,
                )
                matrix[y, x] = self.probabilities_from_values(values)
        self.policy_probability_matrix = matrix
        self.policy_goal = (gx, gy)
        return matrix.copy()

    def _common_state_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "weights": self.weights.copy(),
            "reward_rate": self.reward_rate,
            "update_count": self.update_count,
            "last_delta": self.last_delta,
            "rng_state": self.rng.bit_generator.state,
            "visited_observations": sorted(self.visited_observations),
            "policy_probability_matrix": None if self.policy_probability_matrix is None
            else self.policy_probability_matrix.copy(),
            "policy_goal": self.policy_goal,
            "coder": self.coder.state_dict(),
        }

    def _load_common_state(self, state: Dict[str, Any]) -> None:
        weights = np.asarray(state["weights"], dtype=np.float64)
        if weights.shape != (self.coder.size,):
            raise ValueError("Checkpoint weights have an incompatible shape.")
        self.weights = weights.copy()
        self.reward_rate = float(state["reward_rate"])
        self.update_count = int(state["update_count"])
        self.last_delta = float(state["last_delta"])
        self.rng.bit_generator.state = state["rng_state"]
        self.visited_observations = {
            tuple(int(value) for value in observation)
            for observation in state.get("visited_observations", [])
        }
        policy = state.get("policy_probability_matrix")
        self.policy_probability_matrix = None if policy is None else np.asarray(policy, dtype=np.float64).copy()
        goal = state.get("policy_goal")
        self.policy_goal = None if goal is None else tuple(goal)
        self.coder.load_state_dict(state["coder"])

    @abstractmethod
    def update(self, observation, action, reward, next_observation, next_action) -> float:
        raise NotImplementedError

    @abstractmethod
    def step_size_summary(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError
