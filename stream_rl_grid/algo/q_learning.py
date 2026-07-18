"""One-step differential semi-gradient Q-learning."""

from typing import Any, Dict

from .base import BaseControlAgent
from .registry import register_agent


@register_agent
class DifferentialQLearning(BaseControlAgent):
    algorithm_name = "q_learning"
    display_name = "Differential Q-learning"

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        config.validate()
        super().__init__(coder, config, seed, num_actions=num_actions)
        self.alpha = self.fixed_step_size()

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        del next_action  # Behavior is epsilon-greedy; the off-policy target is greedy.
        active, features = self.feature_values(observation, action)
        next_value = float(self.action_values(next_observation, readonly=False).max())
        delta = float(
            reward - self.reward_rate + next_value
            - self.value_from_features(active, features)
        )
        self.semi_gradient_update(active, features, self.alpha * delta)
        return self.record_real_update(delta)

    def step_size_summary(self) -> Dict[str, float]:
        return self.fixed_step_size_summary()

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state["alpha"] = self.alpha
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        self.alpha = float(state.get("alpha", self.fixed_step_size()))
        self.check_finite()
