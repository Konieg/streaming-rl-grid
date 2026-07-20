"""Differential linear control agents used by the stationary research pipeline."""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Deque, Optional, Tuple

import numpy as np

from .models import BaseModel


@dataclass(frozen=True)
class ResearchAgentConfig:
    method: str
    epsilon: float = 0.05
    effective_step_size: float = 0.1
    reward_rate_step: float = 0.005
    lambda_: float = 0.8
    planning_steps: int = 0
    planning_step_scale: float = 0.5
    replay_capacity: int = 20_000
    planning_strategy: str = "uniform"
    priority_threshold: float = 1e-4
    priority_batch_size: int = 32


class DifferentialLinearAgent:
    """Q-learning, SARSA(lambda), replay, or stochastic-model Dyna."""

    def __init__(
        self,
        coder,
        observations,
        config: ResearchAgentConfig,
        seed: int = 0,
        model: Optional[BaseModel] = None,
        num_actions: int = 5,
    ):
        if config.method not in {"q_learning", "sarsa_lambda", "replay_q", "dyna"}:
            raise ValueError("unknown research method: %s" % config.method)
        if config.method == "dyna" and model is None:
            raise ValueError("Dyna requires a model")
        self.coder = coder
        self.observations = tuple(tuple(int(x) for x in obs) for obs in observations)
        self.features = tuple(
            tuple(coder.active(observation, action, readonly=True) for action in range(num_actions))
            for observation in self.observations
        )
        self.config = config
        self.model = model
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(coder.size, dtype=np.float64)
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.reward_rate = 0.0
        self.real_updates = 0
        self.planning_updates = 0
        self.alpha_real = config.effective_step_size / coder.nominal_active_count
        self.alpha_plan = self.alpha_real * config.planning_step_scale
        self.replay: Deque[Tuple[int, int, float, int]] = deque(maxlen=config.replay_capacity)
        self._priority_heap = []
        self._priority_counter = 0
        self._priority_best = {}

    def values(self, state: int) -> np.ndarray:
        return np.asarray([
            self.weights[self.features[int(state)][action]].sum()
            for action in range(self.num_actions)
        ])

    def value(self, state: int, action: int) -> float:
        indices = self.features[int(state)][int(action)]
        return float(self.weights[indices].sum())

    def policy(self, state: int) -> np.ndarray:
        values = self.values(state)
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))
        probabilities = np.full(self.num_actions, self.config.epsilon / self.num_actions)
        probabilities[best] += (1.0 - self.config.epsilon) / len(best)
        return probabilities

    def select_action(self, state: int) -> int:
        return int(self.rng.choice(self.num_actions, p=self.policy(state)))

    def update_real(
        self, state: int, action: int, reward: float, next_state: int,
        next_action: Optional[int] = None, info=None,
    ) -> float:
        if self.config.method == "sarsa_lambda":
            if next_action is None:
                raise ValueError("SARSA(lambda) needs next_action")
            delta = self._sarsa_lambda_update(state, action, reward, next_state, next_action)
        else:
            delta = self._q_update(state, action, reward, next_state, self.alpha_real)
        self.reward_rate += self.config.reward_rate_step * delta
        self.real_updates += 1
        if self.config.method == "dyna":
            if hasattr(self.model, "observe_with_info"):
                changed_keys = self.model.observe_with_info(
                    state, action, reward, next_state, info or {}
                )
            else:
                changed_keys = self.model.observe(state, action, reward, next_state)
            if changed_keys is None:
                changed_keys = [(int(state), int(action))]
            if self.config.planning_strategy in ("prioritized", "mixed"):
                changed_keys = self._subsample_keys(changed_keys)
                for key in changed_keys:
                    self._push_priority(key, self._model_bellman_error(key))
            self._plan_from_model()
        elif self.config.method == "replay_q":
            self.replay.append((int(state), int(action), float(reward), int(next_state)))
            self._plan_from_replay()
        if not np.all(np.isfinite(self.weights)) or not np.isfinite(self.reward_rate):
            raise FloatingPointError("non-finite research agent state")
        return float(delta)

    def _q_update(self, state: int, action: int, reward: float, next_state: int, alpha: float) -> float:
        indices = self.features[int(state)][int(action)]
        current = float(self.weights[indices].sum())
        delta = float(reward - self.reward_rate + self.values(next_state).max() - current)
        self.weights[indices] += alpha * delta
        return delta

    def _sarsa_lambda_update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> float:
        indices = self.features[int(state)][int(action)]
        current = float(self.weights[indices].sum())
        delta = float(reward - self.reward_rate + self.value(next_state, next_action) - current)
        self.trace *= self.config.lambda_
        self.trace[indices] = 1.0
        self.weights += self.alpha_real * delta * self.trace
        return delta

    def _plan_from_model(self) -> None:
        if len(self.model) == 0:
            return
        if self.config.planning_strategy == "prioritized":
            self._prioritized_planning()
            return
        priority_updates = 0
        if self.config.planning_strategy == "mixed":
            priority_updates = 1
            self._prioritized_planning(priority_updates)
        for _ in range(self.config.planning_steps - priority_updates):
            state, action = self.model.sample_key()
            reward, next_state = self.model.sample((state, action))
            self._q_update(state, action, reward, next_state, self.alpha_plan)
            self.planning_updates += 1

    def _model_bellman_error(self, key) -> float:
        if key not in getattr(self.model, "_key_set", set(self.model.keys)):
            return 0.0
        state, action = key
        target = 0.0
        for (reward, next_state), probability in self.model.distribution(key).items():
            target += probability * (
                reward - self.reward_rate + self.values(next_state).max()
            )
        return abs(float(target - self.value(state, action)))

    def _push_priority(self, key, priority: float) -> None:
        if not np.isfinite(priority) or priority < self.config.priority_threshold:
            return
        key = tuple(key)
        if float(priority) <= self._priority_best.get(key, 0.0):
            return
        self._priority_best[key] = float(priority)
        self._priority_counter += 1
        heapq.heappush(
            self._priority_heap,
            (-float(priority), self._priority_counter, key),
        )

    def _prioritized_planning(self, count=None) -> None:
        updates = self.config.planning_steps if count is None else int(count)
        for _ in range(updates):
            if not self._priority_heap:
                break
            negative_priority, _, key = heapq.heappop(self._priority_heap)
            priority = -negative_priority
            if priority < self._priority_best.get(key, 0.0) - 1e-15:
                continue
            self._priority_best.pop(key, None)
            state, action = key
            self._expected_model_update(key, self.alpha_plan)
            self.planning_updates += 1
            # Propagate the changed value to modeled predecessors.  The models
            # are tiny (<=210 keys), so an exact scan is clearer and safer than
            # maintaining a stale predecessor cache under forgetting.
            if hasattr(self.model, "predecessors"):
                predecessors = self.model.predecessors(state)
            else:
                predecessors = []
                for predecessor in self.model.keys:
                    distribution = self.model.distribution(predecessor)
                    if any(
                        outcome[1] == state and probability > 0.0
                        for outcome, probability in distribution.items()
                    ):
                        predecessors.append(predecessor)
            for predecessor in self._subsample_keys(predecessors):
                self._push_priority(
                    predecessor, self._model_bellman_error(predecessor)
                )

    def _expected_model_update(self, key, alpha: float) -> float:
        state, action = key
        indices = self.features[int(state)][int(action)]
        current = float(self.weights[indices].sum())
        target = 0.0
        for (reward, next_state), probability in self.model.distribution(key).items():
            target += probability * (
                reward - self.reward_rate + self.values(next_state).max()
            )
        delta = float(target - current)
        self.weights[indices] += float(alpha) * delta
        return delta

    def _subsample_keys(self, keys):
        keys = tuple(keys)
        limit = int(self.config.priority_batch_size)
        if len(keys) <= limit:
            return keys
        selected = self.rng.choice(len(keys), size=limit, replace=False)
        return tuple(keys[int(index)] for index in selected)

    def _plan_from_replay(self) -> None:
        if not self.replay:
            return
        for _ in range(self.config.planning_steps):
            state, action, reward, next_state = self.replay[int(self.rng.integers(len(self.replay)))]
            self._q_update(state, action, reward, next_state, self.alpha_plan)
            self.planning_updates += 1
