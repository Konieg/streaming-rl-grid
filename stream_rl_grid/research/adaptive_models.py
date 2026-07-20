"""Dynamic oracle and change-aware factored models for Phases 7--9."""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .continual_environment import DynamicContext, DynamicTwoGoalGrid
from .models import BaseModel, ModelKey, Outcome


WIND_NAMES = ("none", "up", "right", "down", "left")


class DynamicOracleModel(BaseModel):
    name = "dynamic_oracle"

    def __init__(self, environment: DynamicTwoGoalGrid, seed: int = 0):
        super().__init__(seed)
        self.environment = environment
        self.context = DynamicContext("initial")
        for state in range(len(environment.states)):
            for action in range(environment.num_actions):
                self._register(state, action)

    def set_context(self, context: DynamicContext) -> None:
        self.context = context

    def observe(self, state: int, action: int, reward: float, next_state: int):
        del state, action, reward, next_state
        return ()

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        state, action = key
        result: Dict[Outcome, float] = {}
        for probability, next_state, reward, _, _, _ in self.environment.transition_distribution(
            state, action, self.context
        ):
            outcome = (float(reward), int(next_state))
            result[outcome] = result.get(outcome, 0.0) + probability
        return result

    def sample(self, key: ModelKey) -> Outcome:
        distribution = self.distribution(key)
        outcomes = list(distribution)
        probabilities = np.asarray([distribution[outcome] for outcome in outcomes])
        return outcomes[int(self.rng.choice(len(outcomes), p=probabilities))]


class FactoredGridModel(BaseModel):
    """Stable grid geometry plus adaptive wind, edge, and reward factors.

    The coordinate/action mechanics and fixed wall layout are structural prior.
    Current wind, corridor availability, and goal rewards are inferred online.
    """

    name = "factored"

    def __init__(
        self,
        environment: DynamicTwoGoalGrid,
        learning_rate: float = 0.05,
        surprise_adaptive: bool = False,
        seed: int = 0,
    ):
        super().__init__(seed)
        self.environment = environment
        self.learning_rate = float(learning_rate)
        self.surprise_adaptive = bool(surprise_adaptive)
        self.wind = {name: (0.80 if name == "none" else 0.05) for name in WIND_NAMES}
        self.upper_block_probability = 0.05
        self.lower_block_probability = 0.05
        self.reward_a = 4.0
        self.reward_b = 4.0
        self.reward_step = environment.reward_step
        self.reward_collision = environment.reward_collision
        self.last_surprise = 0.0
        self.last_effective_rate = self.learning_rate
        self.update_magnitudes = {
            "wind": 0.0, "upper_block": 0.0, "lower_block": 0.0,
            "reward_a": 0.0, "reward_b": 0.0,
        }
        self.revision = 0
        for state in range(len(environment.states)):
            for action in range(environment.num_actions):
                self._register(state, action)
        self._potential_predecessors = self._build_potential_predecessors()

    def observe(self, state: int, action: int, reward: float, next_state: int):
        return self.observe_with_info(state, action, reward, next_state, {})

    def observe_with_info(
        self, state: int, action: int, reward: float, next_state: int, info,
    ):
        key = (int(state), int(action))
        surprise = self._surprise(key, float(reward), int(next_state))
        rate = self._effective_rate(surprise)
        self.last_surprise = surprise
        self.last_effective_rate = rate
        changed = False
        for factor in self.update_magnitudes:
            self.update_magnitudes[factor] = 0.0

        goal_id = info.get("goal_id")
        if goal_id == "A":
            old = self.reward_a
            self.reward_a += rate * (float(reward) - self.reward_a)
            self.update_magnitudes["reward_a"] = abs(self.reward_a - old)
            changed = True
        elif goal_id == "B":
            old = self.reward_b
            self.reward_b += rate * (float(reward) - self.reward_b)
            self.update_magnitudes["reward_b"] = abs(self.reward_b - old)
            changed = True

        before = self.environment.states[int(state)]
        dx, dy = self.environment_action(action)
        intended = (before[0] + dx, before[1] + dy) if (dx, dy) != (0, 0) else before
        intended_legal = self.environment._legal(intended)
        edge = frozenset((before, intended))
        if intended_legal and edge in (
            self.environment.upper_edge, self.environment.lower_edge
        ):
            blocked_observation = 1.0 if info.get("collision", False) else 0.0
            if edge == self.environment.upper_edge:
                old = self.upper_block_probability
                self.upper_block_probability += rate * (
                    blocked_observation - self.upper_block_probability
                )
                self.update_magnitudes["upper_block"] = abs(
                    self.upper_block_probability - old
                )
            else:
                old = self.lower_block_probability
                self.lower_block_probability += rate * (
                    blocked_observation - self.lower_block_probability
                )
                self.update_magnitudes["lower_block"] = abs(
                    self.lower_block_probability - old
                )
            changed = True

        wind_observation = self._infer_wind(
            state, action, next_state, info, intended, intended_legal
        )
        if wind_observation is not None:
            old_wind = dict(self.wind)
            for name in WIND_NAMES:
                target = 1.0 if name == wind_observation else 0.0
                self.wind[name] += rate * (target - self.wind[name])
            total = sum(self.wind.values())
            self.wind = {name: max(0.0, value) / total for name, value in self.wind.items()}
            self.update_magnitudes["wind"] = sum(
                abs(self.wind[name] - old_wind[name]) for name in WIND_NAMES
            ) / 2.0
            changed = True

        if changed:
            self.revision += 1
            # A changed global factor can alter predictions at every state.
            return tuple(self.keys)
        return (key,)

    @staticmethod
    def environment_action(action: int):
        from .environment import ACTIONS
        return ACTIONS[int(action)]

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        state, action = key
        result: Dict[Outcome, float] = {}
        for probability, next_state, reward, _, _, _ in self.environment.transition_distribution_from_factors(
            state, action,
            reward_a=self.reward_a, reward_b=self.reward_b,
            wind_distribution=self.wind,
            upper_block_probability=self.upper_block_probability,
            lower_block_probability=self.lower_block_probability,
        ):
            outcome = (float(reward), int(next_state))
            result[outcome] = result.get(outcome, 0.0) + probability
        return result

    def sample(self, key: ModelKey) -> Outcome:
        distribution = self.distribution(key)
        outcomes = list(distribution)
        probabilities = np.asarray([distribution[outcome] for outcome in outcomes])
        return outcomes[int(self.rng.choice(len(outcomes), p=probabilities))]

    def predecessors(self, state: int):
        return self._potential_predecessors[int(state)]

    def diagnostics(self) -> Dict[str, float]:
        result = {
            "estimated_wind_none": self.wind["none"],
            "estimated_wind_up": self.wind["up"],
            "estimated_wind_right": self.wind["right"],
            "estimated_wind_down": self.wind["down"],
            "estimated_wind_left": self.wind["left"],
            "estimated_upper_block": self.upper_block_probability,
            "estimated_lower_block": self.lower_block_probability,
            "estimated_reward_a": self.reward_a,
            "estimated_reward_b": self.reward_b,
            "surprise": self.last_surprise,
            "effective_model_rate": self.last_effective_rate,
        }
        result.update({
            "update_" + name: value
            for name, value in self.update_magnitudes.items()
        })
        return result

    def _surprise(self, key: ModelKey, reward: float, next_state: int) -> float:
        distribution = self.distribution(key)
        transition_probability = sum(
            probability for (_, predicted_next), probability in distribution.items()
            if predicted_next == int(next_state)
        )
        expected_reward = sum(
            probability * predicted_reward
            for (predicted_reward, _), probability in distribution.items()
        )
        reward_likelihood = np.exp(-0.5 * ((reward - expected_reward) / 0.5) ** 2)
        likelihood = max(1e-12, transition_probability * reward_likelihood)
        return float(-np.log(likelihood))

    def _effective_rate(self, surprise: float) -> float:
        if not self.surprise_adaptive:
            return self.learning_rate
        minimum, maximum = 0.01, 0.30
        gate = 1.0 / (1.0 + np.exp(-1.5 * (float(surprise) - 2.5)))
        return float(minimum + (maximum - minimum) * gate)

    def _infer_wind(
        self, state, action, next_state, info, intended, intended_legal,
    ) -> Optional[str]:
        if info.get("collision", False) or info.get("goal_reached", False):
            return None
        if not intended_legal:
            return None
        actual = self.environment.states[int(next_state)]
        residual = (actual[0] - intended[0], actual[1] - intended[1])
        mapping = {
            (0, 0): "none", (0, -1): "up", (1, 0): "right",
            (0, 1): "down", (-1, 0): "left",
        }
        return mapping.get(residual)

    def _build_potential_predecessors(self):
        wind = {name: 0.2 for name in WIND_NAMES}
        result = {state: set() for state in range(len(self.environment.states))}
        for key in self.keys:
            for _, next_state, _, _, _, _ in self.environment.transition_distribution_from_factors(
                key[0], key[1], reward_a=4.0, reward_b=4.0,
                wind_distribution=wind,
                upper_block_probability=0.5,
                lower_block_probability=0.5,
            ):
                result[next_state].add(key)
        return {state: tuple(sorted(keys)) for state, keys in result.items()}
