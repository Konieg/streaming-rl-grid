"""Differential Dyna-Q with a latest-transition model."""

from typing import Any, Dict, Sequence, Tuple

from .base import BaseControlAgent, Observation
from .registry import register_agent


ModelKey = Tuple[Observation, int]
ModelValue = Tuple[float, Observation]


@register_agent
class DifferentialDynaQ(BaseControlAgent):
    algorithm_name = "dyna_q"
    display_name = "Differential Dyna-Q"
    extra_config_fields = ("planning_steps",)

    def __init__(self, coder, config, seed: int = 0, num_actions: int = 5):
        config.validate()
        super().__init__(coder, config, seed, num_actions=num_actions)
        self.alpha = self.fixed_step_size()
        self.model: Dict[ModelKey, ModelValue] = {}
        self.planning_update_count = 0

    @staticmethod
    def _observation_key(observation: Sequence[int]) -> Observation:
        return tuple(int(value) for value in observation)

    def _q_learning_delta(self, observation, action, reward, next_observation) -> float:
        active, features = self.feature_values(observation, action)
        next_value = float(self.action_values(next_observation, readonly=False).max())
        delta = float(
            reward - self.reward_rate + next_value
            - self.value_from_features(active, features)
        )
        self.semi_gradient_update(active, features, self.alpha * delta)
        return delta

    def update(self, observation, action, reward, next_observation, next_action) -> float:
        del next_action
        observation = self._observation_key(observation)
        next_observation = self._observation_key(next_observation)

        real_delta = self._q_learning_delta(
            observation, int(action), float(reward), next_observation
        )
        self.model[(observation, int(action))] = (float(reward), next_observation)
        self.record_real_update(real_delta)

        # The average-reward estimate is updated only from real stream transitions.
        # Planning uses the current estimate but does not count model samples again.
        keys = tuple(self.model)
        for _ in range(self.config.planning_steps):
            model_key = keys[int(self.rng.integers(len(keys)))]
            model_reward, model_next = self.model[model_key]
            model_observation, model_action = model_key
            self._q_learning_delta(
                model_observation, model_action, model_reward, model_next
            )
            self.planning_update_count += 1
        self.check_finite()
        return real_delta

    def step_size_summary(self) -> Dict[str, float]:
        return self.fixed_step_size_summary()

    def diagnostics(self) -> Dict[str, float]:
        result = super().diagnostics()
        result.update({
            "planning_update_count": float(self.planning_update_count),
            "model_size": float(len(self.model)),
        })
        return result

    def state_dict(self) -> Dict[str, Any]:
        state = self._common_state_dict()
        state.update({
            "alpha": self.alpha,
            "model": self.model.copy(),
            "planning_update_count": self.planning_update_count,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_common_state(state)
        self.alpha = float(state.get("alpha", self.fixed_step_size()))
        self.model = {
            (self._observation_key(observation), int(action)):
            (float(reward), self._observation_key(next_observation))
            for (observation, action), (reward, next_observation)
            in state.get("model", {}).items()
        }
        self.planning_update_count = int(state.get("planning_update_count", 0))
        self.check_finite()
