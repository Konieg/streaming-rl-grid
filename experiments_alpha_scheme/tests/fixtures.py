"""Small deterministic fixtures shared by experiment-contract tests."""

from typing import Iterable, List

from stream_rl_grid.config import EnvironmentConfig
from stream_rl_grid.environment import ContinualWindyGridWorld


def make_environment(profile: str = "stationary", **changes) -> ContinualWindyGridWorld:
    config = EnvironmentConfig(
        width=5,
        height=5,
        obstacle_count=0,
        profile=profile,
        seed=17,
        max_wind_strength=0,
        context_maps=[[]],
        start_position=[0, 0],
        goal_position=[4, 4],
    )
    for name, value in changes.items():
        setattr(config, name, value)
    return ContinualWindyGridWorld(config)


def fixed_actions(count: int) -> List[int]:
    """A deterministic behaviour-policy stream independent of any learner."""
    return [(3 * step + 1) % 5 for step in range(int(count))]


def transition_stream(environment: ContinualWindyGridWorld, actions: Iterable[int]):
    """Collect only immutable comparable transition fields for contract checks."""
    result = []
    for action in actions:
        observation, reward, terminated, truncated, info = environment.step(action)
        result.append((observation, reward, terminated, truncated, tuple(info["events"])))
    return result
