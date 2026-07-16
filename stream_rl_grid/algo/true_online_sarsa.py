"""Fixed-step differential True Online Sarsa(lambda) with Dutch traces."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent


class DifferentialTrueOnlineSarsa(BaseControlAgent):
    """True Online Sarsa(lambda) for the continuing average-reward setting.

    The continuing task uses gamma=1.  ``q_old`` is never reset during the
    stream, and the average reward follows the same differential update used
    by the other control agents.
    """

    algorithm_name = "true_online_sarsa"

    def __init__(self, features, config, seed: int = 0):
        config.validate()
        super().__init__(features, config, seed)
        self.trace = np.zeros(features.size, dtype=np.float64)
        self.alpha = config.effective_initial_step
        self.q_old = 0.0

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active = self.features.active(observation, action)
        next_active = self.features.active(next_observation, next_action)
        q = float(self.weights[active].sum())
        q_next = float(self.weights[next_active].sum())
        delta = float(reward - self.reward_rate + q_next - q)

        # Dutch trace for gamma=1:
        # e <- lambda e + (1 - alpha lambda e^T x) x
        trace_dot = float(self.trace[active].sum())
        self.trace *= self.config.lambda_
        self.trace[active] += 1.0 - self.alpha * self.config.lambda_ * trace_dot

        correction = q - self.q_old
        self.weights += self.alpha * (delta + correction) * self.trace
        self.weights[active] -= self.alpha * correction
        self.q_old = q_next

        self.reward_rate += self.config.reward_rate_step * delta
        self.update_count += 1
        self.last_delta = delta
        self._check_finite()
        return delta

    def step_size_summary(self) -> Dict[str, float]:
        return {
            "alpha_min": float(self.alpha),
            "alpha_mean": float(self.alpha),
            "alpha_max": float(self.alpha),
            "beta_clip_count": 0.0,
        }

    def _check_finite(self) -> None:
        if (
            not np.all(np.isfinite(self.weights))
            or not np.all(np.isfinite(self.trace))
            or not np.isfinite(self.reward_rate)
            or not np.isfinite(self.q_old)
        ):
            raise FloatingPointError("NaN or Inf detected in the True Online Sarsa learning state.")
        if np.max(np.abs(self.weights)) > 1e12 or abs(self.reward_rate) > 1e12:
            raise FloatingPointError("Learning state exceeded the configured numerical safety scale.")

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({"trace": self.trace.copy(), "alpha": self.alpha, "q_old": self.q_old})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        trace = np.asarray(state["trace"], dtype=np.float64)
        if trace.shape != (self.features.size,):
            raise ValueError("Checkpoint trace has an incompatible shape.")
        self.trace = trace.copy()
        self.alpha = float(state.get("alpha", self.config.effective_initial_step))
        self.q_old = float(state["q_old"])
        self._check_finite()
