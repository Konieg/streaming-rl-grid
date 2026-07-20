"""Differential Dyna-Q+ with the textbook time-since-tried bonus."""

import math
from typing import Any, Dict

from .dyna_q import DifferentialDynaQ, ModelKey
from .registry import register_agent


@register_agent
class DifferentialDynaQPlus(DifferentialDynaQ):
    """One-step Dyna-Q+ for continuing average-reward control.

    Every action at an observed state is represented in the model. Untried
    actions initially predict a zero-reward self transition, and planning uses
    r + kappa * sqrt(tau), where tau is the number of real environment steps
    since that state-action pair was last tried.
    """

    algorithm_name = "dyna_q_plus"
    display_name = "Differential Dyna-Q+"
    extra_config_fields = ("planning_steps", "dyna_plus_kappa")

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        super().__init__(coder, config, seed, num_actions=num_actions)
        self.real_time = 0
        self.last_real_visit: Dict[ModelKey, int] = {}

    def _ensure_observed_state(self, observation) -> None:
        observation = self._observation_key(observation)
        for action in range(self.num_actions):
            key = (observation, action)
            if key not in self.model:
                self.model[key] = (0.0, observation)
                self.last_real_visit[key] = 0

    def _run_planning_updates(self) -> None:
        keys = tuple(self.model)
        for _ in range(self.config.planning_steps):
            model_key = keys[int(self.rng.integers(len(keys)))]
            model_reward, model_next = self.model[model_key]
            tau = self.real_time - self.last_real_visit[model_key]
            planning_reward = (
                model_reward + self.config.dyna_plus_kappa * math.sqrt(tau)
            )
            model_observation, model_action = model_key
            self._q_learning_delta(
                model_observation, model_action, planning_reward, model_next
            )
            self.planning_update_count += 1

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        del next_action
        observation = self._observation_key(observation)
        next_observation = self._observation_key(next_observation)
        action = int(action)
        reward = float(reward)

        self.real_time += 1
        self._ensure_observed_state(observation)
        self._ensure_observed_state(next_observation)
        real_delta = self._q_learning_delta(
            observation, action, reward, next_observation
        )
        key = (observation, action)
        self.model[key] = (reward, next_observation)
        self.last_real_visit[key] = self.real_time
        self.record_real_update(real_delta)
        self._run_planning_updates()
        self.check_finite()
        return real_delta

    def diagnostics(self) -> Dict[str, float]:
        result = super().diagnostics()
        taus = [self.real_time - timestamp for timestamp in self.last_real_visit.values()]
        result.update({
            "dyna_plus_kappa": float(self.config.dyna_plus_kappa),
            "dyna_plus_tau_mean": float(sum(taus) / len(taus)) if taus else 0.0,
            "dyna_plus_tau_max": float(max(taus)) if taus else 0.0,
        })
        return result

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            "real_time": self.real_time,
            "last_real_visit": self.last_real_visit.copy(),
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.real_time = int(state.get("real_time", self.update_count))
        raw_visits = state.get("last_real_visit", {})
        self.last_real_visit = {
            (self._observation_key(observation), int(action)): int(timestamp)
            for (observation, action), timestamp in raw_visits.items()
        }
        for key in self.model:
            self.last_real_visit.setdefault(key, 0)
