"""A continuing windy grid world with structured non-stationarity.

The environment deliberately never terminates. Reaching the goal produces a reward and
teleports the agent to a random legal non-goal cell as part of the same continuing MDP.
"""

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import EnvironmentConfig


Coord = Tuple[int, int]
ACTIONS: Tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
ACTION_NAMES: Tuple[str, ...] = ("up", "right", "down", "left", "stay")
NO_ACTION = len(ACTIONS)
WIND_DIRECTIONS: Tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
WIND_BY_NAME = dict(zip(("up", "right", "down", "left"), WIND_DIRECTIONS))


class ContinualWindyGridWorld:
    """Five-action continuing grid world with hidden, structured context changes."""

    def __init__(self, config: EnvironmentConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.context_maps = self._prepare_context_maps(config.context_maps)
        self.goal_path = self._prepare_goal_path(config.goal_path)
        self.step_count = 0
        self.context_index = 0
        self.wind_phase = 0
        self.reward_phase = 0
        self.goal_path_index = 0
        self.goal_path_direction = 1
        self.dormant_obstacle: Optional[Coord] = None
        self.goal: Coord = self._initial_goal()
        self.start_position: Coord = self._initial_start()
        self.agent_state: Coord = self.start_position
        self.previous_action = NO_ACTION
        self.last_events: List[str] = []

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def active_obstacles(self) -> Set[Coord]:
        obstacles = set(self.context_maps[self.context_index])
        if self.dormant_obstacle is not None:
            obstacles.discard(self.dormant_obstacle)
        return obstacles

    def observation(self) -> Tuple[int, int, int, int, int]:
        return (
            self.agent_state[0],
            self.agent_state[1],
            self.goal[0],
            self.goal[1],
            self.previous_action,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, int, int, int, int], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.context_index = 0
        self.wind_phase = 0
        self.reward_phase = 0
        self.goal_path_index = 0
        self.goal_path_direction = 1
        self.dormant_obstacle = None
        self.goal = self._initial_goal()
        self.start_position = self._initial_start()
        self.agent_state = self.start_position
        self.previous_action = NO_ACTION
        self.last_events = ["reset"]
        return self.observation(), self._info(False, False, (0, 0))

    def step(self, action: int):
        if action < 0 or action >= len(ACTIONS):
            raise ValueError("Invalid action index: %r" % action)

        before = self.agent_state
        collision = False
        candidate = before
        dx, dy = ACTIONS[action]
        if (dx, dy) != (0, 0):
            proposed = (candidate[0] + dx, candidate[1] + dy)
            if not self._is_legal(proposed):
                collision = True
            else:
                candidate = proposed

        wind = self.sample_wind(before)
        if not collision:
            wx, wy = wind
            unit = (int(np.sign(wx)), int(np.sign(wy)))
            for _ in range(abs(wx) + abs(wy)):
                proposed = (candidate[0] + unit[0], candidate[1] + unit[1])
                if not self._is_legal(proposed):
                    collision = True
                    break
                candidate = proposed

        if collision:
            candidate = before
            reward = self._phase_reward("collision")
        else:
            reward = self._phase_reward("goal" if candidate == self.goal else "step")

        reached_goal = not collision and candidate == self.goal
        self.agent_state = candidate
        if self.dormant_obstacle is not None and self.agent_state != self.dormant_obstacle:
            self.dormant_obstacle = None

        goal_event = None
        relocate_target_after_schedules = False
        if reached_goal:
            if self.config.goal_reached_behavior == "relocate_target":
                relocate_target_after_schedules = True
                goal_event = "target_relocated_after_goal"
            else:
                self.agent_state = self._restart_state()
                goal_event = "agent_restarted_after_goal"
            if self.dormant_obstacle is not None and self.agent_state != self.dormant_obstacle:
                self.dormant_obstacle = None

        self.previous_action = action
        self.step_count += 1
        self.last_events = []
        self._advance_schedules(skip_goal_move=relocate_target_after_schedules)
        if relocate_target_after_schedules:
            self._relocate_goal_randomly()
        if goal_event is not None:
            self.last_events.insert(0, goal_event)
        info = self._info(collision, reached_goal, wind)
        info["state_before"] = before
        info["state_after_dynamics"] = candidate
        return self.observation(), float(reward), False, False, info

    def wind_vector(self, state: Coord) -> Coord:
        """Return the currently selected unit wind direction (without sampling)."""
        del state
        manual = self.config.manual_wind_direction
        if self.config.w_strength == 0.0:
            return (0, 0)
        if self.config.wind_changes:
            return WIND_DIRECTIONS[self.wind_phase]
        if manual == "none":
            return (0, 0)
        return WIND_BY_NAME[manual]

    def sample_wind(self, state: Coord) -> Coord:
        """Sample one extra wind displacement using w_strength as its probability."""
        direction = self.wind_vector(state)
        if direction == (0, 0) or self.config.w_strength <= 0.0:
            return (0, 0)
        if self.config.w_strength >= 1.0 or self.rng.random() < self.config.w_strength:
            return direction
        return (0, 0)

    def apply_manual_configuration(
        self,
        obstacles: Iterable[Coord],
        start: Coord,
        goal: Coord,
        wind_direction: str,
        replace_maps: bool = True,
    ) -> None:
        """Atomically edit the environment without resetting the continuing agent state."""
        layout = {(int(x), int(y)) for x, y in obstacles}
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if wind_direction not in WIND_BY_NAME and wind_direction != "none":
            raise ValueError("Unknown wind direction: %s" % wind_direction)
        if len(layout) > self.width * self.height - 2:
            raise ValueError("Obstacle count leaves fewer than two legal cells.")
        if any(not self._in_bounds(point) for point in layout | {start, goal}):
            raise ValueError("All obstacle, start, and goal coordinates must be inside the grid.")
        if start == goal:
            raise ValueError("Start and goal coordinates must differ.")
        if start in layout or goal in layout:
            raise ValueError("Start and goal coordinates cannot be obstacles.")
        if not self.free_cells_connected(layout):
            raise ValueError("Obstacle map disconnects the legal cells.")

        map_count = self.config.num_contexts if self.config.obstacle_switches else 1
        self.config.obstacle_count = len(layout)
        self.config.obstacle_coordinates = [list(point) for point in sorted(layout)]
        self.config.start_position = list(start)
        self.config.goal_position = list(goal)
        self.config.manual_wind_direction = wind_direction
        if replace_maps:
            self.context_maps = self._expand_context_maps([layout], map_count)
            self.context_index = 0
        self.config.context_maps = [
            [list(point) for point in sorted(context)] for context in self.context_maps
        ]
        self.start_position = start
        self.goal = goal
        # A live edit must not teleport the agent or change the policy slice selected by
        # previous_action. If its current cell becomes blocked, use the same dormant-cell
        # rule as a scheduled context switch: the obstacle activates after the agent leaves.
        self.dormant_obstacle = self.agent_state if self.agent_state in self.context_maps[self.context_index] else None
        self.last_events = ["manual_environment_update"]

    def _advance_schedules(self, skip_goal_move: bool = False) -> None:
        if self.config.wind_changes and self._schedule_due(
            self.config.wind_start_step, self.config.wind_period
        ):
            self.wind_phase = (self.wind_phase + 1) % len(WIND_DIRECTIONS)
            self.last_events.append("wind_phase:%d" % self.wind_phase)

        if self.config.reward_changes and self._schedule_due(
            self.config.reward_start_step, self.config.reward_period
        ):
            self.reward_phase = (self.reward_phase + 1) % len(WIND_DIRECTIONS)
            self.last_events.append("reward_phase:%d" % self.reward_phase)

        if (
            not skip_goal_move
            and self.config.goal_moves
            and self._schedule_due(
                self.config.target_move_start_step,
                self.config.target_move_interval,
            )
        ):
            old_goal = self.goal
            self._move_goal_to_next_legal_waypoint()
            if self.goal != old_goal:
                self.last_events.append("goal_moved")

        if (
            self.config.obstacle_switches
            and self._schedule_due(
                self.config.context_switch_start_step,
                self.config.context_switch_interval,
            )
        ):
            self.context_index = (self.context_index + 1) % len(self.context_maps)
            raw_obstacles = self.context_maps[self.context_index]
            self.dormant_obstacle = self.agent_state if self.agent_state in raw_obstacles else None
            if self.goal in self.active_obstacles:
                if not self.config.goal_moves:
                    raise RuntimeError(
                        "A fixed goal cannot be covered by a switching obstacle map."
                    )
                self._move_goal_to_next_legal_waypoint()
            self.last_events.append("context:%d" % self.context_index)

    def _schedule_due(self, start_step: Optional[int], period: int) -> bool:
        first = int(period if start_step is None else start_step)
        return self.step_count >= first and (self.step_count - first) % int(period) == 0

    def _move_goal_to_next_legal_waypoint(self) -> None:
        n = len(self.goal_path)
        for _ in range(max(1, 2 * n)):
            next_index = self.goal_path_index + self.goal_path_direction
            if next_index >= n or next_index < 0:
                self.goal_path_direction *= -1
                next_index = self.goal_path_index + self.goal_path_direction
            self.goal_path_index = next_index
            candidate = self.goal_path[self.goal_path_index]
            if candidate not in self.active_obstacles and candidate != self.agent_state:
                self.goal = candidate
                return

    def _phase_reward(self, event: str) -> float:
        base = {
            "goal": self.config.reward_goal,
            "collision": self.config.reward_collision,
            "step": self.config.reward_step,
        }[event]
        if not self.config.reward_changes:
            return base
        multipliers = {
            "goal": (1.0, 0.75, 1.25, 0.9),
            "collision": (1.0, 1.2, 0.8, 1.0),
            "step": (1.0, 1.1, 0.9, 1.0),
        }
        return base * multipliers[event][self.reward_phase]

    def _prepare_context_maps(self, raw_maps: Optional[List[List[List[int]]]]) -> List[Set[Coord]]:
        expected_count = self.config.num_contexts if self.config.obstacle_switches else 1
        if not raw_maps and self.config.obstacle_coordinates:
            coordinates = {(int(p[0]), int(p[1])) for p in self.config.obstacle_coordinates}
            if len(coordinates) == self.config.obstacle_count:
                raw_maps = [[list(point) for point in sorted(coordinates)]]
        if raw_maps:
            maps = [{(int(p[0]), int(p[1])) for p in layout} for layout in raw_maps]
            if len(maps) == 1 and expected_count > 1:
                maps = self._expand_context_maps(maps, expected_count)
            if len(maps) != expected_count:
                raise ValueError("Expected %d context map(s), received %d." % (expected_count, len(maps)))
            for layout in maps:
                self._validate_obstacles(layout)
            return maps
        return [self._generate_connected_obstacles() for _ in range(expected_count)]

    def _expand_context_maps(
        self, initial_maps: Sequence[Set[Coord]], expected_count: int
    ) -> List[Set[Coord]]:
        """Keep authored maps and generate the remaining switching contexts."""
        maps = [set(layout) for layout in initial_maps]
        attempts = 0
        while len(maps) < expected_count:
            candidate = self._generate_connected_obstacles()
            attempts += 1
            if candidate not in maps or attempts >= 100:
                maps.append(candidate)
                attempts = 0
        return maps

    def _generate_connected_obstacles(self) -> Set[Coord]:
        cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        if not self.config.goal_moves and self.config.goal_position is not None:
            fixed_goal = tuple(int(value) for value in self.config.goal_position)
            cells.remove(fixed_goal)
        for _ in range(10_000):
            if self.config.obstacle_count == 0:
                return set()
            indices = self.rng.choice(len(cells), size=self.config.obstacle_count, replace=False)
            obstacles = {cells[int(i)] for i in indices}
            if self.free_cells_connected(obstacles):
                return obstacles
        raise ValueError("Could not generate a connected map; reduce obstacle_count.")

    def _validate_obstacles(self, obstacles: Set[Coord]) -> None:
        if len(obstacles) != self.config.obstacle_count:
            raise ValueError("Each context map must contain exactly obstacle_count cells.")
        if any(not self._in_bounds(p) for p in obstacles):
            raise ValueError("Obstacle lies outside the grid.")
        if not self.config.goal_moves and self.config.goal_position is not None:
            fixed_goal = tuple(int(value) for value in self.config.goal_position)
            if fixed_goal in obstacles:
                raise ValueError(
                    "A switching obstacle map cannot cover the fixed goal."
                )
        if not self.free_cells_connected(obstacles):
            raise ValueError("Obstacle map disconnects the legal cells.")

    def free_cells_connected(self, obstacles: Iterable[Coord]) -> bool:
        blocked = set(obstacles)
        free = {(x, y) for y in range(self.height) for x in range(self.width)} - blocked
        if not free:
            return False
        start = next(iter(free))
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ACTIONS[:4]:
                nxt = (x + dx, y + dy)
                if nxt in free and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen == free

    def _prepare_goal_path(self, raw_path: Optional[List[List[int]]]) -> List[Coord]:
        if raw_path:
            path = [(int(p[0]), int(p[1])) for p in raw_path]
            if not path or any(not self._in_bounds(p) for p in path):
                raise ValueError("Every goal waypoint must be inside the grid.")
            return path
        path: List[Coord] = []
        for y in range(self.height):
            xs: Sequence[int] = range(self.width) if y % 2 == 0 else range(self.width - 1, -1, -1)
            path.extend((x, y) for x in xs)
        return path

    def _first_legal_goal(self) -> Coord:
        obstacles = (
            set().union(*self.context_maps)
            if self.config.obstacle_switches and not self.config.goal_moves
            else self.context_maps[0]
        )
        for i in range(len(self.goal_path) - 1, -1, -1):
            if self.goal_path[i] not in obstacles:
                self.goal_path_index = i
                return self.goal_path[i]
        raise ValueError("No legal goal cell exists.")

    def _initial_goal(self) -> Coord:
        if self.config.goal_position is not None:
            goal = tuple(int(v) for v in self.config.goal_position)
            relevant_maps = (
                self.context_maps
                if self.config.obstacle_switches and not self.config.goal_moves
                else [self.active_obstacles]
            )
            if any(goal in obstacles for obstacles in relevant_maps):
                raise ValueError("Goal coordinate cannot be an obstacle.")
            if goal in self.goal_path:
                self.goal_path_index = self.goal_path.index(goal)
            return goal
        return self._first_legal_goal()

    def _initial_start(self) -> Coord:
        if self.config.start_position is not None:
            start = tuple(int(v) for v in self.config.start_position)
            if start == self.goal or start in self.active_obstacles:
                raise ValueError("Start coordinate must be a legal non-goal cell.")
            return start
        return self._random_legal_state(exclude={self.goal})

    def _restart_state(self) -> Coord:
        """Sample a fresh legal non-goal position after a rewarded goal transition."""
        return self._random_legal_state(exclude={self.goal})

    def _relocate_goal_randomly(self) -> None:
        """Move the target while leaving the agent on the previously rewarded target cell."""
        self.goal = self._random_legal_state(
            exclude={self.agent_state},
            legal_in_all_contexts=(
                self.config.obstacle_switches and not self.config.goal_moves
            ),
        )
        if self.goal in self.goal_path:
            self.goal_path_index = self.goal_path.index(self.goal)

    def _random_legal_state(
        self, exclude: Optional[Set[Coord]] = None,
        legal_in_all_contexts: bool = False,
    ) -> Coord:
        excluded = set() if exclude is None else set(exclude)
        obstacles = (
            set().union(*self.context_maps)
            if legal_in_all_contexts else self.active_obstacles
        )
        legal = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in obstacles and (x, y) not in excluded
        ]
        if not legal:
            raise RuntimeError("No legal state is available for relocation.")
        return legal[int(self.rng.integers(len(legal)))]

    def _in_bounds(self, state: Coord) -> bool:
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height

    def _is_legal(self, state: Coord) -> bool:
        return self._in_bounds(state) and state not in self.active_obstacles

    def _info(self, collision: bool, reached_goal: bool, wind: Coord) -> Dict[str, Any]:
        return {
            "collision": collision,
            "goal_reached": reached_goal,
            "wind": wind,
            "wind_phase": self.wind_phase,
            "reward_phase": self.reward_phase,
            "context_index": self.context_index,
            "events": list(self.last_events),
            "global_step": self.step_count,
            "dormant_obstacle": self.dormant_obstacle,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "context_index": self.context_index,
            "wind_phase": self.wind_phase,
            "reward_phase": self.reward_phase,
            "goal_path_index": self.goal_path_index,
            "goal_path_direction": self.goal_path_direction,
            "dormant_obstacle": self.dormant_obstacle,
            "goal": self.goal,
            "start_position": self.start_position,
            "agent_state": self.agent_state,
            "previous_action": self.previous_action,
            "context_maps": [sorted(m) for m in self.context_maps],
            "goal_path": list(self.goal_path),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self.context_index = int(state["context_index"])
        self.wind_phase = int(state["wind_phase"])
        self.reward_phase = int(state["reward_phase"])
        self.goal_path_index = int(state["goal_path_index"])
        self.goal_path_direction = int(state["goal_path_direction"])
        self.dormant_obstacle = None if state["dormant_obstacle"] is None else tuple(state["dormant_obstacle"])
        self.goal = tuple(state["goal"])
        self.agent_state = tuple(state["agent_state"])
        self.start_position = tuple(state.get("start_position", self.config.start_position or self.agent_state))
        self.previous_action = int(state["previous_action"])
        self.context_maps = [{tuple(p) for p in layout} for layout in state["context_maps"]]
        self.goal_path = [tuple(p) for p in state["goal_path"]]
        self.rng.bit_generator.state = state["rng_state"]
        self.last_events = ["checkpoint_loaded"]
