"""Differential Sarsa(lambda) with replacing traces and TIDBD step sizes."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent
from .registry import register_agent


@register_agent
class DifferentialSarsaTIDBD(BaseControlAgent):
    algorithm_name = "tidbd"
    display_name = "Differential SARSA(λ) + TIDBD"
    extra_config_fields = ("lambda_", "theta", "beta_min", "beta_max")
    samples_next_action_before_update = True

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        config.validate()
        super().__init__(coder, config, seed, num_actions=num_actions)
        initial_alpha = config.effective_initial_step / coder.nominal_active_count
        self.beta = np.full(coder.size, np.log(initial_alpha), dtype=np.float64)
        self.h = np.zeros(coder.size, dtype=np.float64)
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.beta_clip_count = 0

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active = self.coder.active(observation, action)
        next_active = self.coder.active(next_observation, next_action)
        delta = float(
            reward - self.reward_rate
            + self.weights[next_active].sum() - self.weights[active].sum()
        )
        self.trace *= self.config.lambda_
        self.trace[active] = 1.0

        proposed_beta = self.beta[active] + self.config.theta * delta * self.h[active]
        clipped_beta = np.clip(proposed_beta, self.config.beta_min, self.config.beta_max)
        self.beta_clip_count += int(np.count_nonzero(clipped_beta != proposed_beta))
        self.beta[active] = clipped_beta
        alpha = np.exp(self.beta)
        self.weights += alpha * delta * self.trace

        decay = np.ones_like(self.h)
        decay[active] = np.maximum(0.0, 1.0 - alpha[active] * self.trace[active])
        self.h = self.h * decay + alpha * delta * self.trace
        return self.record_real_update(delta)

    def step_size_summary(self) -> Dict[str, float]:
        values = np.exp(self.beta)
        return {
            "alpha_min": float(values.min()),
            "alpha_mean": float(values.mean()),
            "alpha_max": float(values.max()),
            "beta_clip_count": float(self.beta_clip_count),
        }

    def check_finite(self) -> None:
        arrays = (self.weights, self.beta, self.h, self.trace)
        if not all(np.all(np.isfinite(array)) for array in arrays) or not np.isfinite(self.reward_rate):
            raise FloatingPointError("NaN or Inf detected in the learning state.")
        if np.max(np.abs(self.weights)) > 1e12 or abs(self.reward_rate) > 1e12:
            raise FloatingPointError("Learning state exceeded the configured numerical safety scale.")

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({
            "beta": self.beta.copy(), "h": self.h.copy(), "trace": self.trace.copy(),
            "beta_clip_count": self.beta_clip_count,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        for name in ("beta", "h", "trace"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != (self.coder.size,):
                raise ValueError("Checkpoint %s has an incompatible shape." % name)
            setattr(self, name, value.copy())
        self.beta_clip_count = int(state["beta_clip_count"])
        self.check_finite()
