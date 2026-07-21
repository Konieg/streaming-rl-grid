"""Differential Sarsa(lambda) with replacing traces and TIDBD step sizes."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent


class DifferentialSarsaTIDBD(BaseControlAgent):
    algorithm_name = "tidbd"

    def __init__(self, features, config, seed: int = 0):
        config.validate()
        super().__init__(features, config, seed)
        initial_alpha = self.initial_per_feature_step_size
        self.beta = np.full(features.size, np.log(initial_alpha), dtype=np.float64)
        self.h = np.zeros(features.size, dtype=np.float64)
        self.trace = np.zeros(features.size, dtype=np.float64)
        self.beta_clip_count = 0

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active = self.features.active(observation, action)
        delta = float(
            reward - self.reward_rate
            + self.bootstrap_value(next_observation, next_action)
            - self.weights[active].sum()
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
        self.reward_rate += self.config.reward_rate_step * delta
        self.update_count += 1
        self.last_delta = delta
        self._check_finite()
        return delta

    def step_size_summary(self) -> Dict[str, float]:
        values = np.exp(self.beta)
        return {
            "alpha_min": float(values.min()),
            "alpha_mean": float(values.mean()),
            "alpha_max": float(values.max()),
            "beta_clip_count": float(self.beta_clip_count),
        }

    def _check_finite(self) -> None:
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
            if value.shape != (self.features.size,):
                raise ValueError("Checkpoint %s has an incompatible shape." % name)
            setattr(self, name, value.copy())
        self.beta_clip_count = int(state["beta_clip_count"])
        self._check_finite()
