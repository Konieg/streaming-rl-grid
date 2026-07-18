"""Differential Dyna-Q with Watkins traces on the real experience stream."""

from typing import Any, Dict

import numpy as np

from .dyna_q import DifferentialDynaQ
from .registry import register_agent


@register_agent
class DifferentialDynaQLambda(DifferentialDynaQ):
    """Dyna-Q(lambda) with one-step planning over unordered model samples.

    Eligibility traces follow the temporally contiguous real stream. Planning samples
    are drawn independently from the model and therefore retain ordinary one-step
    Q-learning updates instead of sharing an artificial trace across unrelated samples.
    """

    algorithm_name = "dyna_q_lambda"
    display_name = "Differential Dyna-Q(lambda)"
    extra_config_fields = ("lambda_", "planning_steps")
    samples_next_action_before_update = True

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        super().__init__(coder, config, seed=seed, num_actions=num_actions)
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.trace_cut_count = 0

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        observation = self._observation_key(observation)
        next_observation = self._observation_key(next_observation)
        active, features = self.feature_values(observation, action)
        next_values = self.action_values(next_observation, readonly=False)
        next_action_is_greedy = int(next_action) in self.greedy_actions(next_values)
        real_delta = float(
            reward - self.reward_rate + next_values.max()
            - self.value_from_features(active, features)
        )

        self.trace *= self.config.lambda_
        self.trace[active] = features
        self.weights += self.alpha * real_delta * self.trace
        if not next_action_is_greedy:
            self.trace.fill(0.0)
            self.trace_cut_count += 1

        self.model[(observation, int(action))] = (float(reward), next_observation)
        self.record_real_update(real_delta)
        self._run_planning_updates()
        self.check_finite()
        return real_delta

    def diagnostics(self) -> Dict[str, float]:
        result = super().diagnostics()
        result["trace_cut_count"] = float(self.trace_cut_count)
        return result

    def check_finite(self) -> None:
        super().check_finite()
        if not np.all(np.isfinite(self.trace)):
            raise FloatingPointError("NaN or Inf detected in the eligibility trace.")

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            "trace": self.trace.copy(),
            "trace_cut_count": self.trace_cut_count,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        trace = np.asarray(state["trace"], dtype=np.float64)
        if trace.shape != (self.coder.size,):
            raise ValueError("Checkpoint trace has an incompatible shape.")
        self.trace = trace.copy()
        self.trace_cut_count = int(state.get("trace_cut_count", 0))
        self.check_finite()
