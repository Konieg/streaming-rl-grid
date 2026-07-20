"""Sparse binary factorization of the observable state-action value function."""

from typing import Any, Dict, Sequence

import numpy as np

from .config import EnvironmentConfig


class SparseFactorizedStateActionFeatures:
    """Share value parameters across complementary discrete state factors.

    Every state-action pair activates one feature in each of seven disjoint
    groups: action bias, agent position, goal position, relative displacement,
    relative direction, Manhattan distance, and previous-action interaction.
    """

    representation_name = "sparse-factorized"
    action_count = 5
    previous_action_count = action_count + 1
    group_names = (
        "action_bias",
        "position_action",
        "goal_action",
        "relative_displacement_action",
        "relative_direction_action",
        "distance_action",
        "previous_action_interaction",
    )

    def __init__(self, env_config: EnvironmentConfig):
        self.width = int(env_config.width)
        self.height = int(env_config.height)
        position_count = self.width * self.height
        relative_count = (2 * self.width - 1) * (2 * self.height - 1)
        distance_count = self.width + self.height - 1
        self.group_sizes = (
            self.action_count,
            position_count * self.action_count,
            position_count * self.action_count,
            relative_count * self.action_count,
            9 * self.action_count,
            distance_count * self.action_count,
            self.previous_action_count * self.action_count,
        )
        offsets = []
        offset = 0
        for size in self.group_sizes:
            offsets.append(offset)
            offset += size
        self.group_offsets = tuple(offsets)
        self._size = offset

    @property
    def nominal_active_count(self) -> int:
        return len(self.group_names)

    @property
    def size(self) -> int:
        return self._size

    def active(self, observation: Sequence[int], action: int, readonly: bool = False) -> np.ndarray:
        del readonly
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

        position = x * self.height + y
        goal_position = goal_x * self.height + goal_y
        dx = goal_x - x
        dy = goal_y - y
        relative = (dx + self.width - 1) * (2 * self.height - 1) + (dy + self.height - 1)
        direction = (int(np.sign(dx)) + 1) * 3 + (int(np.sign(dy)) + 1)
        distance = abs(dx) + abs(dy)

        local_indices = (
            action,
            position * self.action_count + action,
            goal_position * self.action_count + action,
            relative * self.action_count + action,
            direction * self.action_count + action,
            distance * self.action_count + action,
            previous_action * self.action_count + action,
        )
        return np.asarray(
            [offset + index for offset, index in zip(self.group_offsets, local_indices)],
            dtype=np.int64,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "representation": self.representation_name,
            "version": 1,
            "width": self.width,
            "height": self.height,
            "action_count": self.action_count,
            "previous_action_count": self.previous_action_count,
            "group_names": list(self.group_names),
            "group_sizes": list(self.group_sizes),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state != self.state_dict():
            raise ValueError("Checkpoint sparse factorized representation is incompatible.")
