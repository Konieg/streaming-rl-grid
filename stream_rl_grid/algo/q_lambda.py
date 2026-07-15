"""Watkins's differential Q(lambda) with replacing traces."""

from typing import Any, Dict

import numpy as np

from .base import BaseControlAgent
from .registry import register_agent


@register_agent
class DifferentialQLambda(BaseControlAgent):
    algorithm_name = "q_lambda"
    display_name = "Watkins Differential Q(λ)"
    extra_config_fields = ("lambda_",)
    samples_next_action_before_update = True

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        config.validate()
        super().__init__(coder, config, seed, num_actions=num_actions)
        self.alpha = self.fixed_step_size()
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.trace_cut_count = 0

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        active = self.coder.active(observation, action)
        next_values = self.action_values(next_observation, readonly=False)
        next_action_is_greedy = int(next_action) in self.greedy_actions(next_values)
        delta = float(
            reward - self.reward_rate
            + next_values.max() - self.weights[active].sum()
        )

        self.trace *= self.config.lambda_
        self.trace[active] = 1.0
        self.weights += self.alpha * delta * self.trace
        if not next_action_is_greedy:
            self.trace.fill(0.0)
            self.trace_cut_count += 1
        return self.record_real_update(delta)

    def step_size_summary(self) -> Dict[str, float]:
        return self.fixed_step_size_summary()

    def diagnostics(self) -> Dict[str, float]:
        result = super().diagnostics()
        result["trace_cut_count"] = float(self.trace_cut_count)
        return result

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({
            "alpha": self.alpha,
            "trace": self.trace.copy(),
            "trace_cut_count": self.trace_cut_count,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        trace = np.asarray(state["trace"], dtype=np.float64)
        if trace.shape != (self.coder.size,):
            raise ValueError("Checkpoint trace has an incompatible shape.")
        self.trace = trace.copy()
        self.alpha = float(state.get("alpha", self.fixed_step_size()))
        self.trace_cut_count = int(state.get("trace_cut_count", 0))
        self.check_finite()
