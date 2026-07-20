"""Finite continuing grid MDPs used by the stationary diagnostics.

The goal transition is not a terminal transition.  It emits ``reward_goal`` and
immediately places the agent at a uniformly sampled legal non-goal state.  This
makes both the simulator and the exact enumerator describe the same continuing
average-reward problem.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]
ACTIONS: Tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
ACTION_NAMES = ("up", "right", "down", "left", "stay")
WIND = {
    "none": (0, 0),
    "up": (0, -1),
    "right": (1, 0),
    "down": (0, 1),
    "left": (-1, 0),
}


@dataclass(frozen=True)
class StationaryGridSpec:
    name: str
    width: int
    height: int
    obstacles: Tuple[Coord, ...]
    goal: Coord
    reward_goal: float = 20.0
    reward_step: float = -1.0
    reward_collision: float = -2.0
    wind_direction: str = "none"
    wind_probability: float = 0.0

    def validate(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("grid dimensions must be at least two")
        if self.wind_direction not in WIND:
            raise ValueError("unknown wind direction: %s" % self.wind_direction)
        if not 0.0 <= self.wind_probability <= 1.0:
            raise ValueError("wind_probability must lie in [0, 1]")
        obstacles = set(self.obstacles)
        if len(obstacles) != len(self.obstacles):
            raise ValueError("obstacles must be unique")
        cells = {(x, y) for y in range(self.height) for x in range(self.width)}
        if self.goal not in cells or self.goal in obstacles:
            raise ValueError("goal must be a legal cell")
        if not obstacles <= cells:
            raise ValueError("obstacle outside grid")
        free = cells - obstacles
        if len(free) < 2:
            raise ValueError("at least two free cells are required")
        frontier = [next(iter(free))]
        seen = set(frontier)
        while frontier:
            x, y = frontier.pop()
            for dx, dy in ACTIONS[:4]:
                nxt = (x + dx, y + dy)
                if nxt in free and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        if seen != free:
            raise ValueError("free cells must be connected")


class ContinuingGridMDP:
    """Simulator plus exact transition kernel for a stationary continuing grid."""

    num_actions = len(ACTIONS)

    def __init__(self, spec: StationaryGridSpec, seed: int = 0):
        spec.validate()
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.obstacles = frozenset(spec.obstacles)
        self.states: Tuple[Coord, ...] = tuple(
            (x, y)
            for y in range(spec.height)
            for x in range(spec.width)
            if (x, y) not in self.obstacles and (x, y) != spec.goal
        )
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.restart_distribution = np.full(len(self.states), 1.0 / len(self.states))
        self.state_index = 0
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, state_index: Optional[int] = None) -> Tuple[int, Dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state_index is None:
            self.state_index = int(self.rng.integers(len(self.states)))
        else:
            if not 0 <= int(state_index) < len(self.states):
                raise ValueError("invalid state_index")
            self.state_index = int(state_index)
        self.step_count = 0
        return self.state_index, {"continued": True}

    def observation(self, state_index: Optional[int] = None) -> Tuple[int, int, int, int, int]:
        index = self.state_index if state_index is None else int(state_index)
        x, y = self.states[index]
        gx, gy = self.spec.goal
        return x, y, gx, gy, self.local_wall_mask(index)

    def local_wall_mask(self, state_index: int) -> int:
        x, y = self.states[int(state_index)]
        mask = 0
        for bit, (dx, dy) in enumerate(ACTIONS[:4]):
            if not self._legal((x + dx, y + dy)):
                mask |= 1 << bit
        return mask

    def transition_distribution(self, state_index: int, action: int) -> Tuple[Tuple[float, int, float, bool, bool], ...]:
        """Return ``(probability, next_state, reward, goal, collision)`` outcomes."""
        state_index = int(state_index)
        action = int(action)
        if not 0 <= state_index < len(self.states) or not 0 <= action < self.num_actions:
            raise ValueError("invalid state or action")
        wind_probability = self.spec.wind_probability if self.spec.wind_direction != "none" else 0.0
        branches = [(1.0 - wind_probability, (0, 0))]
        if wind_probability > 0.0:
            branches.append((wind_probability, WIND[self.spec.wind_direction]))
        aggregate: Dict[Tuple[int, float, bool, bool], float] = {}
        for probability, wind in branches:
            if probability <= 0.0:
                continue
            raw_next, reward, reached_goal, collision = self._deterministic_outcome(
                self.states[state_index], action, wind
            )
            if reached_goal:
                for next_index, restart_probability in enumerate(self.restart_distribution):
                    key = (next_index, reward, True, collision)
                    aggregate[key] = aggregate.get(key, 0.0) + probability * float(restart_probability)
            else:
                next_index = self.state_to_index[raw_next]
                key = (next_index, reward, False, collision)
                aggregate[key] = aggregate.get(key, 0.0) + probability
        outcomes = tuple(
            (probability, next_index, reward, goal, collision)
            for (next_index, reward, goal, collision), probability in sorted(aggregate.items())
        )
        if not np.isclose(sum(row[0] for row in outcomes), 1.0):
            raise RuntimeError("transition probabilities do not sum to one")
        return outcomes

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, object]]:
        outcomes = self.transition_distribution(self.state_index, action)
        probabilities = np.asarray([row[0] for row in outcomes], dtype=np.float64)
        selected = outcomes[int(self.rng.choice(len(outcomes), p=probabilities))]
        _, next_index, reward, goal, collision = selected
        before = self.state_index
        self.state_index = int(next_index)
        self.step_count += 1
        info = {
            "goal_reached": bool(goal),
            "collision": bool(collision),
            "continued": True,
            "state_before": before,
            "state_after": self.state_index,
        }
        return self.state_index, float(reward), False, False, info

    def expected_reward_and_transition(self) -> Tuple[np.ndarray, np.ndarray]:
        n_states = len(self.states)
        rewards = np.zeros((n_states, self.num_actions), dtype=np.float64)
        transitions = np.zeros((n_states, self.num_actions, n_states), dtype=np.float64)
        for state in range(n_states):
            for action in range(self.num_actions):
                for probability, next_state, reward, _, _ in self.transition_distribution(state, action):
                    transitions[state, action, next_state] += probability
                    rewards[state, action] += probability * reward
        return rewards, transitions

    def _deterministic_outcome(self, before: Coord, action: int, wind: Coord) -> Tuple[Coord, float, bool, bool]:
        candidate = before
        collision = False
        dx, dy = ACTIONS[int(action)]
        if (dx, dy) != (0, 0):
            proposed = (candidate[0] + dx, candidate[1] + dy)
            if not self._legal(proposed):
                collision = True
            else:
                candidate = proposed
        if not collision and wind != (0, 0):
            proposed = (candidate[0] + wind[0], candidate[1] + wind[1])
            if not self._legal(proposed):
                collision = True
            else:
                candidate = proposed
        if collision:
            return before, float(self.spec.reward_collision), False, True
        if candidate == self.spec.goal:
            return candidate, float(self.spec.reward_goal), True, False
        return candidate, float(self.spec.reward_step), False, False

    def _legal(self, state: Coord) -> bool:
        x, y = state
        return (
            0 <= x < self.spec.width
            and 0 <= y < self.spec.height
            and state not in self.obstacles
        )


def stationary_ladder() -> Tuple[StationaryGridSpec, ...]:
    """A compact four-level competence ladder; the last level is the main task."""
    wall_a = tuple((2, y) for y in range(5) if y != 2)
    wall_b = tuple((3, y) for y in range(7) if y != 4)
    main_obstacles = tuple(
        sorted(
            {(2, y) for y in range(8) if y != 2}
            | {(5, y) for y in range(1, 9) if y != 6}
        )
    )
    return (
        StationaryGridSpec("A_open5", 5, 5, (), (4, 4)),
        StationaryGridSpec("B_wall5", 5, 5, wall_a, (4, 4)),
        StationaryGridSpec(
            "C_windy7", 7, 7, wall_b, (6, 6),
            wind_direction="right", wind_probability=0.20,
        ),
        StationaryGridSpec(
            "D_corridor9", 9, 9, main_obstacles, (8, 8),
            wind_direction="right", wind_probability=0.20,
        ),
    )
