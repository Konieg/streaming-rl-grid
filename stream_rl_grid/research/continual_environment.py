"""Structured non-stationary continuing grids for research Phases 6--9."""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .environment import ACTIONS, ACTION_NAMES, Coord, WIND


@dataclass(frozen=True)
class DynamicContext:
    name: str
    reward_a: float = 6.0
    reward_b: float = 3.0
    wind_direction: str = "right"
    wind_probability: float = 0.0
    upper_block_probability: float = 0.0
    lower_block_probability: float = 0.0

    def validate(self) -> None:
        if self.wind_direction not in ("none", "up", "right", "down", "left"):
            raise ValueError("unknown dynamic wind direction")
        for value in (
            self.wind_probability,
            self.upper_block_probability,
            self.lower_block_probability,
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("context probabilities must lie in [0, 1]")

    def key(self) -> Tuple[object, ...]:
        return (
            self.name,
            round(float(self.reward_a), 8), round(float(self.reward_b), 8),
            self.wind_direction, round(float(self.wind_probability), 8),
            round(float(self.upper_block_probability), 8),
            round(float(self.lower_block_probability), 8),
        )


@dataclass(frozen=True)
class ContinualScenario:
    name: str
    total_steps: int
    schedule: str
    change_steps: Tuple[int, ...]
    local_window: Optional[Tuple[int, int]] = None
    drift_period: int = 4_000

    def context_at(self, step: int) -> DynamicContext:
        step = int(step)
        if self.schedule == "loca_reward":
            if step < self.change_steps[0]:
                return DynamicContext("reward_A_high", reward_a=6.0, reward_b=4.0)
            return DynamicContext("reward_B_high", reward_a=1.0, reward_b=4.0)
        if self.schedule == "obstacle_abrupt":
            if step < self.change_steps[0]:
                return DynamicContext("upper_open", reward_a=6.0, reward_b=4.0)
            return DynamicContext(
                "upper_blocked", reward_a=6.0, reward_b=4.0,
                upper_block_probability=1.0,
            )
        if self.schedule == "wind_drift":
            phase = (step % self.drift_period) / float(self.drift_period)
            triangular = 1.0 - abs(2.0 * phase - 1.0)
            probability = 0.80 * triangular
            return DynamicContext(
                "wind_triangular", reward_a=6.0, reward_b=4.0,
                wind_direction="down", wind_probability=probability,
            )
        if self.schedule == "recurring_composition":
            segment = min(step // 4_000, 4)
            contexts = (
                DynamicContext(
                    "train_A_right", reward_a=6.0, reward_b=3.0,
                    wind_direction="right", wind_probability=0.45,
                ),
                DynamicContext(
                    "train_B_left", reward_a=3.0, reward_b=6.0,
                    wind_direction="left", wind_probability=0.45,
                ),
                DynamicContext(
                    "recur_A_right", reward_a=6.0, reward_b=3.0,
                    wind_direction="right", wind_probability=0.45,
                ),
                DynamicContext(
                    "heldout_A_left", reward_a=6.0, reward_b=3.0,
                    wind_direction="left", wind_probability=0.45,
                ),
                DynamicContext(
                    "recur_B_left", reward_a=3.0, reward_b=6.0,
                    wind_direction="left", wind_probability=0.45,
                ),
            )
            return contexts[segment]
        raise ValueError("unknown schedule: %s" % self.schedule)

    def diagnostic_valid(self, step: int) -> bool:
        if self.local_window is None:
            return True
        start, end = self.local_window
        return not start <= int(step) < end

    def phase_name(self, step: int) -> str:
        if self.local_window is not None:
            start, end = self.local_window
            if step < start:
                return "prechange"
            if step < end:
                return "local_observation"
            return "choice_test"
        return self.context_at(step).name


def phase6_scenarios() -> Tuple[ContinualScenario, ...]:
    return (
        ContinualScenario(
            "loca_reward", 20_000, "loca_reward", (10_000, 10_500),
            local_window=(10_000, 10_500),
        ),
        ContinualScenario(
            "obstacle_abrupt", 20_000, "obstacle_abrupt", (10_000,),
        ),
        ContinualScenario(
            "wind_drift", 20_000, "wind_drift", (4_000, 8_000, 12_000, 16_000),
            drift_period=4_000,
        ),
        ContinualScenario(
            "recurring_composition", 20_000, "recurring_composition",
            (4_000, 8_000, 12_000, 16_000),
        ),
    )


class DynamicTwoGoalGrid:
    """Two-corridor continuing grid with reward, wind, and edge factors."""

    width = 7
    height = 7
    num_actions = len(ACTIONS)
    goal_a: Coord = (6, 1)
    goal_b: Coord = (6, 5)
    pseudo_goal: Coord = (6, 3)
    upper_edge = frozenset(((2, 1), (3, 1)))
    lower_edge = frozenset(((2, 5), (3, 5)))
    reward_step = -0.05
    reward_collision = -0.25

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.obstacles = frozenset((3, y) for y in range(7) if y not in (1, 5))
        self.goals = frozenset((self.goal_a, self.goal_b))
        self.states = tuple(
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.obstacles and (x, y) not in self.goals
        )
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.restart_states = tuple(
            self.state_to_index[(0, y)] for y in range(self.height)
        )
        self.restart_distribution = np.zeros(len(self.states), dtype=np.float64)
        self.restart_distribution[list(self.restart_states)] = 1.0 / len(self.restart_states)
        self.local_states = tuple(
            index for index, (x, y) in enumerate(self.states)
            if x >= 4 and y <= 2
        )
        self.state_index = self.restart_states[0]
        self.step_count = 0
        self.spec = type("DynamicSpecView", (), {
            "width": self.width, "height": self.height,
            "name": "two_goal_dynamic",
        })()

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state_index = int(self.rng.choice(self.restart_states))
        self.step_count = 0
        return self.state_index

    def set_common_state(self) -> int:
        self.state_index = self.state_to_index[(0, 3)]
        return self.state_index

    def set_local_state(self) -> int:
        self.state_index = int(self.rng.choice(self.local_states))
        return self.state_index

    def observation(self, state_index: Optional[int] = None) -> Tuple[int, int, int, int, int]:
        index = self.state_index if state_index is None else int(state_index)
        x, y = self.states[index]
        gx, gy = self.pseudo_goal
        return x, y, gx, gy, self.local_wall_mask(index)

    def local_wall_mask(self, state_index: int) -> int:
        x, y = self.states[int(state_index)]
        mask = 0
        for bit, (dx, dy) in enumerate(ACTIONS[:4]):
            if not self._legal((x + dx, y + dy)):
                mask |= 1 << bit
        return mask

    def true_wind_distribution(self, context: DynamicContext) -> Dict[str, float]:
        context.validate()
        if context.wind_direction == "none" or context.wind_probability == 0.0:
            return {
                "none": 1.0, "up": 0.0, "right": 0.0,
                "down": 0.0, "left": 0.0,
            }
        result = {
            "none": 1.0 - context.wind_probability,
            "up": 0.0, "right": 0.0, "down": 0.0, "left": 0.0,
        }
        result[context.wind_direction] = context.wind_probability
        return result

    def transition_distribution(
        self, state_index: int, action: int, context: DynamicContext,
    ) -> Tuple[Tuple[float, int, float, bool, bool, Optional[str]], ...]:
        return self.transition_distribution_from_factors(
            state_index, action,
            reward_a=context.reward_a, reward_b=context.reward_b,
            wind_distribution=self.true_wind_distribution(context),
            upper_block_probability=context.upper_block_probability,
            lower_block_probability=context.lower_block_probability,
        )

    def transition_distribution_from_factors(
        self,
        state_index: int,
        action: int,
        reward_a: float,
        reward_b: float,
        wind_distribution: Mapping[str, float],
        upper_block_probability: float,
        lower_block_probability: float,
    ) -> Tuple[Tuple[float, int, float, bool, bool, Optional[str]], ...]:
        state = self.states[int(state_index)]
        action = int(action)
        aggregate: Dict[Tuple[int, float, bool, bool, Optional[str]], float] = {}
        for wind_name, wind_probability in wind_distribution.items():
            if wind_probability <= 0.0:
                continue
            wind = WIND[wind_name]
            intended = state
            action_collision = False
            dx, dy = ACTIONS[action]
            if (dx, dy) != (0, 0):
                proposed = (state[0] + dx, state[1] + dy)
                if not self._legal(proposed):
                    action_collision = True
                else:
                    intended = proposed
            if action_collision:
                branches = ((1.0, state, True),)
            else:
                block_probability = self._edge_block_probability(
                    state, intended, upper_block_probability, lower_block_probability
                )
                branches = (
                    (block_probability, state, True),
                    (1.0 - block_probability, intended, False),
                )
            for block_branch, candidate, collision in branches:
                probability = float(wind_probability) * float(block_branch)
                if probability <= 0.0:
                    continue
                if not collision and wind != (0, 0):
                    proposed = (candidate[0] + wind[0], candidate[1] + wind[1])
                    if not self._legal(proposed):
                        collision = True
                        candidate = state
                    else:
                        wind_block = self._edge_block_probability(
                            candidate, proposed,
                            upper_block_probability, lower_block_probability,
                        )
                        if wind_block > 0.0:
                            # Split the branch once more for a stochastic blocked edge.
                            self._add_outcome(
                                aggregate, probability * wind_block, state,
                                self.reward_collision, False, True, None,
                            )
                            probability *= 1.0 - wind_block
                            if probability <= 0.0:
                                continue
                        candidate = proposed
                if collision:
                    self._add_outcome(
                        aggregate, probability, state,
                        self.reward_collision, False, True, None,
                    )
                elif candidate in self.goals:
                    goal_id = "A" if candidate == self.goal_a else "B"
                    reward = float(reward_a if goal_id == "A" else reward_b)
                    for restart, restart_probability in enumerate(self.restart_distribution):
                        if restart_probability:
                            self._add_outcome(
                                aggregate, probability * float(restart_probability),
                                self.states[restart], reward, True, False, goal_id,
                            )
                else:
                    self._add_outcome(
                        aggregate, probability, candidate,
                        self.reward_step, False, False, None,
                    )
        outcomes = tuple(
            (probability, next_state, reward, goal, collision, goal_id)
            for (next_state, reward, goal, collision, goal_id), probability
            in sorted(aggregate.items(), key=lambda item: repr(item[0]))
        )
        if not np.isclose(sum(row[0] for row in outcomes), 1.0):
            raise RuntimeError("dynamic transition probabilities do not sum to one")
        return outcomes

    def step(self, action: int, context: DynamicContext):
        before = self.state_index
        outcomes = self.transition_distribution(before, action, context)
        probabilities = np.asarray([row[0] for row in outcomes], dtype=np.float64)
        selected = outcomes[int(self.rng.choice(len(outcomes), p=probabilities))]
        _, next_state, reward, goal, collision, goal_id = selected
        self.state_index = int(next_state)
        self.step_count += 1
        info = {
            "goal_reached": bool(goal), "goal_id": goal_id,
            "collision": bool(collision), "continued": True,
            "state_before": before, "state_after": self.state_index,
            "context": context.name,
        }
        return self.state_index, float(reward), False, False, info

    def expected_reward_and_transition(self, context: DynamicContext):
        n_states = len(self.states)
        rewards = np.zeros((n_states, self.num_actions), dtype=np.float64)
        transitions = np.zeros((n_states, self.num_actions, n_states), dtype=np.float64)
        for state in range(n_states):
            for action in range(self.num_actions):
                for probability, next_state, reward, _, _, _ in self.transition_distribution(
                    state, action, context
                ):
                    transitions[state, action, next_state] += probability
                    rewards[state, action] += probability * reward
        return rewards, transitions

    def _add_outcome(
        self, aggregate, probability, candidate, reward, goal, collision, goal_id,
    ) -> None:
        if probability <= 0.0:
            return
        next_index = self.state_to_index[candidate]
        key = (next_index, float(reward), bool(goal), bool(collision), goal_id)
        aggregate[key] = aggregate.get(key, 0.0) + float(probability)

    def _edge_block_probability(
        self, before: Coord, after: Coord,
        upper_probability: float, lower_probability: float,
    ) -> float:
        edge = frozenset((before, after))
        if edge == self.upper_edge:
            return float(upper_probability)
        if edge == self.lower_edge:
            return float(lower_probability)
        return 0.0

    def _legal(self, state: Coord) -> bool:
        x, y = state
        return (
            0 <= x < self.width and 0 <= y < self.height
            and state not in self.obstacles
        )


class FrozenDynamicMDP:
    """Adapter exposing one dynamic context as a stationary finite MDP."""

    def __init__(self, environment: DynamicTwoGoalGrid, context: DynamicContext):
        self.environment = environment
        self.context = context
        self.states = environment.states
        self.num_actions = environment.num_actions
        self.spec = environment.spec

    def expected_reward_and_transition(self):
        return self.environment.expected_reward_and_transition(self.context)

    def transition_distribution(self, state: int, action: int):
        return tuple(
            (probability, next_state, reward, goal, collision)
            for probability, next_state, reward, goal, collision, _
            in self.environment.transition_distribution(state, action, self.context)
        )
