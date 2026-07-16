"""Frozen Phase 0 protocol built only on the project environment API."""

from dataclasses import dataclass
from typing import Dict, Tuple

from stream_rl_grid.config import EnvironmentConfig


NUM_ACTIONS = 5
FIXED_ALPHAS: Tuple[float, ...] = (0.01, 0.05, 0.10)
INITIAL_ADAPTIVE_ALPHA = 0.05
REWARD_RATE_STEP = 0.01
META_STEP = 0.01
ALPHA_MIN = 1e-4
ALPHA_MAX = 0.5
EPSILON = 0.1
SWITCH_INTERVAL = 500
DEFAULT_STEPS = 3_000
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
METRIC_WINDOW = 100
RECOVERY_SMOOTHING = 25
RECOVERY_TOLERANCE = 0.10
PROBE_INTERVAL = 100
PROBE_STEPS = 100

BASE_OBSTACLES = [[0, 3], [2, 2], [4, 3]]
CONTEXT_MAPS = [
    [[0, 3], [2, 2], [4, 3]],
    [[0, 1], [2, 3], [4, 1]],
]
GOAL_PATH = [[4, 0], [0, 0]]
START = [0, 4]
INITIAL_GOAL = [4, 0]


@dataclass(frozen=True)
class Condition:
    name: str

    def make_environment_config(self, seed: int) -> EnvironmentConfig:
        common = dict(
            width=5,
            height=5,
            obstacle_count=3,
            seed=seed,
            reward_goal=10.0,
            reward_collision=-5.0,
            reward_step=-1.0,
            start_position=list(START),
            goal_position=list(INITIAL_GOAL),
        )
        if self.name == "stationary":
            return EnvironmentConfig(
                profile="stationary",
                obstacle_coordinates=BASE_OBSTACLES,
                manual_wind_direction="none",
                w_strength=0.0,
                **common,
            )
        if self.name == "seasonal_wind":
            return EnvironmentConfig(
                profile="seasonal_wind",
                obstacle_coordinates=BASE_OBSTACLES,
                manual_wind_direction="auto",
                w_strength=0.3,
                wind_period=SWITCH_INTERVAL,
                **common,
            )
        if self.name == "hidden_context":
            return EnvironmentConfig(
                profile="hidden_context",
                num_contexts=2,
                obstacle_coordinates=None,
                context_maps=CONTEXT_MAPS,
                context_switch_interval=SWITCH_INTERVAL,
                manual_wind_direction="none",
                w_strength=0.0,
                **common,
            )
        if self.name == "moving_goal":
            return EnvironmentConfig(
                profile="moving_goal",
                obstacle_coordinates=BASE_OBSTACLES,
                goal_path=GOAL_PATH,
                target_move_interval=SWITCH_INTERVAL,
                manual_wind_direction="none",
                w_strength=0.0,
                **common,
            )
        raise ValueError("Unknown Phase 0 condition: %s" % self.name)


CONDITIONS: Tuple[Condition, ...] = tuple(
    Condition(name)
    for name in ("stationary", "seasonal_wind", "hidden_context", "moving_goal")
)


def condition_by_name(name: str) -> Condition:
    for condition in CONDITIONS:
        if condition.name == name:
            return condition
    raise ValueError("Unknown Phase 0 condition: %s" % name)


def frozen_protocol() -> Dict[str, object]:
    """Return the protocol fields that are written beside every result set."""
    return {
        "conditions": [condition.name for condition in CONDITIONS],
        "fixed_alphas": list(FIXED_ALPHAS),
        "initial_adaptive_alpha": INITIAL_ADAPTIVE_ALPHA,
        "reward_rate_step": REWARD_RATE_STEP,
        "meta_step": META_STEP,
        "adaptive_alpha_bounds": [ALPHA_MIN, ALPHA_MAX],
        "epsilon": EPSILON,
        "switch_interval": SWITCH_INTERVAL,
        "metric_window": METRIC_WINDOW,
        "recovery_smoothing": RECOVERY_SMOOTHING,
        "recovery_tolerance": RECOVERY_TOLERANCE,
        "probe_interval": PROBE_INTERVAL,
        "probe_steps": PROBE_STEPS,
    }

