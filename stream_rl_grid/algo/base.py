"""Shared linear epsilon-greedy interface for differential TD control."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..config import AgentConfig


Observation = Tuple[int, int, int, int, int]
StateKey = Tuple[int, int, int, int]


class BaseControlAgent(ABC):
    """Common representation and behavior policy for this project's TD agents."""

    algorithm_name = "base"
    display_name = "Base TD control"
    extra_config_fields = ()
    samples_next_action_before_update = False

    def __init__(
        self,
        coder,
        config: "AgentConfig",
        seed: int = 0,
        num_actions: int = 5,
    ):
        if int(num_actions) < 1:
            raise ValueError("num_actions must be positive.")
        self.coder = coder
        self.config = config
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(coder.size, dtype=np.float64)
        self.reward_rate = 0.0
        self.update_count = 0
        self.last_delta = 0.0
        self.visited_observations = set()
        self.observation_counts: Dict[Observation, int] = {}
        self.policy_probability_matrix = None
        self.policy_goal = None

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    def value(self, observation: Sequence[int], action: int, readonly: bool = False) -> float:
        indices, values = self.feature_values(observation, action, readonly=readonly)
        return float(np.dot(self.weights[indices], values))

    def feature_values(self, observation, action, readonly: bool = False):
        if hasattr(self.coder, "feature_values"):
            return self.coder.feature_values(observation, action, readonly=readonly)
        indices = self.coder.active(observation, action, readonly=readonly)
        return indices, np.ones(indices.shape, dtype=np.float64)

    def value_from_features(self, indices, values) -> float:
        return float(np.dot(self.weights[indices], values))

    def semi_gradient_update(self, indices, values, scale: float) -> None:
        self.weights[indices] += float(scale) * values

    def action_values(self, observation: Sequence[int], readonly: bool = False) -> np.ndarray:
        return np.asarray(
            [self.value(observation, action, readonly=readonly) for action in range(self.num_actions)],
            dtype=np.float64,
        )

    @staticmethod
    def greedy_actions(values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))

    def probabilities_from_values(self, values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        best = self.greedy_actions(values)
        probabilities = np.full(
            self.num_actions,
            self.config.epsilon / self.num_actions,
            dtype=np.float64,
        )
        probabilities[best] += (1.0 - self.config.epsilon) / len(best)
        return probabilities

    def action_probabilities(self, observation: Sequence[int], readonly: bool = True) -> np.ndarray:
        return self.probabilities_from_values(self.action_values(observation, readonly=readonly))

    def select_action(self, observation: Sequence[int]) -> int:
        observation = tuple(int(value) for value in observation)
        self.visited_observations.add(observation)
        self.observation_counts[observation] = self.observation_counts.get(observation, 0) + 1
        probabilities = self.action_probabilities(observation, readonly=False)
        return int(self.rng.choice(self.num_actions, p=probabilities))

    def learn_and_select_next(
        self,
        observation,
        action,
        reward,
        next_observation,
    ) -> Tuple[float, int]:
        """Learn from one transition and sample behavior at the correct algorithmic time."""
        if self.samples_next_action_before_update:
            next_action = self.select_action(next_observation)
            delta = self.update(
                observation, action, reward, next_observation, next_action
            )
        else:
            delta = self.update(
                observation, action, reward, next_observation, None
            )
            next_action = self.select_action(next_observation)
        return float(delta), int(next_action)

    def freeze_policy_matrix(self, width: int, height: int, goal: Sequence[int]) -> np.ndarray:
        """Freeze the current epsilon-greedy evaluation policy for every grid position.

        The environment includes previous_action in its Markov state. For a stable 2-D GUI
        view, conditional action-probability vectors are mixed using their empirical visit
        frequencies. Completely unseen states retain the uniform random policy.
        """
        gx, gy = int(goal[0]), int(goal[1])
        observations_by_state: Dict[StateKey, list] = {}
        for observation, count in self.observation_counts.items():
            x, y, seen_gx, seen_gy = observation[:4]
            observations_by_state.setdefault((x, y, seen_gx, seen_gy), []).append((observation, count))

        matrix = np.full(
            (height, width, self.num_actions),
            1.0 / self.num_actions,
            dtype=np.float64,
        )
        for y in range(height):
            for x in range(width):
                conditional = observations_by_state.get((x, y, gx, gy), [])
                if not conditional:
                    continue
                matrix[y, x] = np.average(
                    [self.action_probabilities(observation, readonly=True)
                     for observation, _ in conditional],
                    weights=[count for _, count in conditional],
                    axis=0,
                )
        self.policy_probability_matrix = matrix
        self.policy_goal = (gx, gy)
        return matrix.copy()

    def _common_state_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "num_actions": self.num_actions,
            "weights": self.weights.copy(),
            "reward_rate": self.reward_rate,
            "update_count": self.update_count,
            "last_delta": self.last_delta,
            "rng_state": self.rng.bit_generator.state,
            "visited_observations": sorted(self.visited_observations),
            "observation_counts": self.observation_counts.copy(),
            "policy_probability_matrix": None if self.policy_probability_matrix is None
            else self.policy_probability_matrix.copy(),
            "policy_goal": self.policy_goal,
            "coder": self.coder.state_dict(),
        }

    def _load_common_state(self, state: Dict[str, Any]) -> None:
        saved_algorithm = state.get("algorithm")
        if saved_algorithm == "sarsa_lambda":
            saved_algorithm = "sarsa"
        if saved_algorithm is not None and saved_algorithm != self.algorithm_name:
            raise ValueError(
                "Checkpoint algorithm %r cannot be loaded by %r."
                % (saved_algorithm, self.algorithm_name)
            )
        if int(state.get("num_actions", self.num_actions)) != self.num_actions:
            raise ValueError("Checkpoint action count is incompatible.")
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
        raw_counts = state.get("observation_counts")
        if raw_counts is None:
            self.observation_counts = {observation: 1 for observation in self.visited_observations}
        else:
            self.observation_counts = {
                tuple(int(value) for value in observation): int(count)
                for observation, count in raw_counts.items()
            }
        policy = state.get("policy_probability_matrix")
        self.policy_probability_matrix = None if policy is None else np.asarray(policy, dtype=np.float64).copy()
        goal = state.get("policy_goal")
        self.policy_goal = None if goal is None else tuple(goal)
        self.coder.load_state_dict(state["coder"])

    def fixed_step_size(self) -> float:
        denominator = float(
            getattr(self.coder, "step_size_denominator", self.coder.nominal_active_count)
        )
        return self.config.effective_initial_step / denominator

    def fixed_step_size_summary(self) -> Dict[str, float]:
        alpha = float(getattr(self, "alpha", self.fixed_step_size()))
        return {
            "alpha_min": alpha,
            "alpha_mean": alpha,
            "alpha_max": alpha,
            "beta_clip_count": 0.0,
        }

    def record_real_update(self, delta: float) -> float:
        """Update average reward and counters after one real environment transition."""
        self.reward_rate += self.config.reward_rate_step * float(delta)
        self.update_count += 1
        self.last_delta = float(delta)
        self.check_finite()
        return self.last_delta

    def check_finite(self) -> None:
        if not np.all(np.isfinite(self.weights)) or not np.isfinite(self.reward_rate):
            raise FloatingPointError("NaN or Inf detected in the learning state.")

    def diagnostics(self) -> Dict[str, float]:
        result = dict(self.step_size_summary())
        result.update({
            "update_count": float(self.update_count),
            "planning_update_count": 0.0,
            "model_size": 0.0,
        })
        return result

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
