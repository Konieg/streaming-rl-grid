"""Fixed-step differential Sarsa(lambda) implementation."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent
from .registry import register_agent


@register_agent
class DifferentialSarsa(BaseControlAgent):
    algorithm_name = "sarsa"
    display_name = "Differential SARSA(λ)"
    extra_config_fields = ("lambda_",)
    samples_next_action_before_update = True

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        config.validate()
        super().__init__(coder, config, seed, num_actions=num_actions)
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.alpha = self.fixed_step_size()

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active, features = self.feature_values(observation, action)
        next_active, next_features = self.feature_values(next_observation, next_action)
        delta = float(
            reward - self.reward_rate
            + self.value_from_features(next_active, next_features)
            - self.value_from_features(active, features)
        )
        self.trace *= self.config.lambda_
        self.trace[active] = features
        self.weights += self.alpha * delta * self.trace
        return self.record_real_update(delta)

    def step_size_summary(self) -> Dict[str, float]:
        return self.fixed_step_size_summary()

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({"trace": self.trace.copy(), "alpha": self.alpha})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        trace = np.asarray(state["trace"], dtype=np.float64)
        if trace.shape != (self.coder.size,):
            raise ValueError("Checkpoint trace has an incompatible shape.")
        self.trace = trace.copy()
        self.alpha = float(state.get("alpha", self.fixed_step_size()))
