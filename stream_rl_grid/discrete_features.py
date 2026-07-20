"""Exact one-hot features for the finite grid state-action space."""

from typing import Any, Dict, Sequence

import numpy as np

from .config import EnvironmentConfig


class DiscreteStateActionFeatures:
    """Map every observable state-action pair to one deterministic table index.

    The observation is ``(x, y, goal_x, goal_y, previous_action)``.  The
    previous-action component has one extra value for the initial ``no action``
    marker, while the candidate action has the five environment actions.
    """

    action_count = 5
    previous_action_count = action_count + 1
    representation_name = "tabular-one-hot"

    def __init__(self, env_config: EnvironmentConfig):
        self.width = int(env_config.width)
        self.height = int(env_config.height)

    @property
    def nominal_active_count(self) -> int:
        return 1

    @property
    def size(self) -> int:
        position_count = self.width * self.height
        return position_count * position_count * self.previous_action_count * self.action_count

    def active(self, observation: Sequence[int], action: int, readonly: bool = False) -> np.ndarray:
        del readonly  # The mapping is immutable; evaluation never allocates state.
        if len(observation) != 5:
            raise ValueError("Observation must contain x, y, goal_x, goal_y, and previous_action.")
        x, y, goal_x, goal_y, previous_action = (int(value) for value in observation)
        action = int(action)
        if not (0 <= x < self.width and 0 <= goal_x < self.width):
            raise ValueError("Observation x coordinates are outside the configured grid.")
        if not (0 <= y < self.height and 0 <= goal_y < self.height):
            raise ValueError("Observation y coordinates are outside the configured grid.")
        if not 0 <= previous_action < self.previous_action_count:
            raise ValueError("Observation previous_action is invalid: %r" % previous_action)
        if not 0 <= action < self.action_count:
            raise ValueError("Action is invalid: %r" % action)

        index = x
        index = index * self.height + y
        index = index * self.width + goal_x
        index = index * self.height + goal_y
        index = index * self.previous_action_count + previous_action
        index = index * self.action_count + action
        return np.asarray([index], dtype=np.int64)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "representation": self.representation_name,
            "width": self.width,
            "height": self.height,
            "action_count": self.action_count,
            "previous_action_count": self.previous_action_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state != self.state_dict():
            raise ValueError("Checkpoint discrete state-action representation is incompatible.")
