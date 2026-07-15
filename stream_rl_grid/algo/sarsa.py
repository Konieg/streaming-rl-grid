"""Fixed-step differential Sarsa(lambda) implementation."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent


class DifferentialSarsa(BaseControlAgent):
    algorithm_name = "sarsa"

    def __init__(self, features, config, seed: int = 0):
        config.validate()
        super().__init__(features, config, seed)
        self.trace = np.zeros(features.size, dtype=np.float64)
        self.alpha = config.effective_initial_step

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active = self.features.active(observation, action)
        next_active = self.features.active(next_observation, next_action)
        delta = float(
            reward - self.reward_rate
            + self.weights[next_active].sum() - self.weights[active].sum()
        )
        self.trace *= self.config.lambda_
        self.trace[active] = 1.0
        self.weights += self.alpha * delta * self.trace
        self.reward_rate += self.config.reward_rate_step * delta
        self.update_count += 1
        self.last_delta = delta
        if not np.all(np.isfinite(self.weights)) or not np.isfinite(self.reward_rate):
            raise FloatingPointError("NaN or Inf detected in the Sarsa learning state.")
        return delta

    def step_size_summary(self) -> Dict[str, float]:
        return {
            "alpha_min": float(self.alpha), "alpha_mean": float(self.alpha),
            "alpha_max": float(self.alpha), "beta_clip_count": 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({"trace": self.trace.copy(), "alpha": self.alpha})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        trace = np.asarray(state["trace"], dtype=np.float64)
        if trace.shape != (self.features.size,):
            raise ValueError("Checkpoint trace has an incompatible shape.")
        self.trace = trace.copy()
        self.alpha = float(state.get("alpha", self.config.effective_initial_step))
