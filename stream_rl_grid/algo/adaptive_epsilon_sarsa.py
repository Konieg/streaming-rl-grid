"""Differential Sarsa(lambda) with TD-error-triggered adaptive exploration."""

from typing import Any, Dict

import numpy as np

from .sarsa import DifferentialSarsa


class DifferentialAdaptiveEpsilonSarsa(DifferentialSarsa):
    """Fixed-step Sarsa whose epsilon rises when smoothed |TD error| is abnormal."""

    algorithm_name = "adaptive_epsilon_sarsa"

    def __init__(self, features, config, seed: int = 0):
        super().__init__(features, config, seed)
        self.td_error_magnitude = float(config.adaptive_epsilon_u_ref)
        self.current_epsilon = float(config.adaptive_epsilon_min)

    @property
    def epsilon(self) -> float:
        return self.current_epsilon

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        # A_{t+1} has already been selected with the previous epsilon.  The new
        # epsilon therefore applies only to the next action not yet selected.
        delta = super().update(observation, action, reward, next_observation, next_action)
        kappa = self.config.adaptive_epsilon_kappa
        self.td_error_magnitude = (
            (1.0 - kappa) * self.td_error_magnitude + kappa * abs(delta)
        )
        excess = max(0.0, self.td_error_magnitude - self.config.adaptive_epsilon_u_ref)
        self.current_epsilon = float(
            np.clip(
                self.config.adaptive_epsilon_min + self.config.adaptive_epsilon_scale * excess,
                self.config.adaptive_epsilon_min,
                self.config.adaptive_epsilon_max,
            )
        )
        if not np.isfinite(self.td_error_magnitude) or not np.isfinite(self.current_epsilon):
            raise FloatingPointError("NaN or Inf detected in the adaptive exploration state.")
        return delta

    def exploration_summary(self) -> Dict[str, float]:
        return {
            "epsilon": float(self.current_epsilon),
            "td_error_magnitude": float(self.td_error_magnitude),
        }

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "td_error_magnitude": self.td_error_magnitude,
                "current_epsilon": self.current_epsilon,
            }
        )
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.td_error_magnitude = float(state["td_error_magnitude"])
        self.current_epsilon = float(state["current_epsilon"])
        if not np.isfinite(self.td_error_magnitude) or not np.isfinite(self.current_epsilon):
            raise FloatingPointError("NaN or Inf detected in the adaptive exploration state.")
        if not self.config.adaptive_epsilon_min <= self.current_epsilon <= self.config.adaptive_epsilon_max:
            raise ValueError("Checkpoint adaptive epsilon is outside the configured bounds.")
