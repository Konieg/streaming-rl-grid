"""Sparse multi-group tile coding for research experiments."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from ..tile_coder import IndexHashTable, tiles


@dataclass(frozen=True)
class FeatureConfig:
    # The initial 10/10/5 resolution was too fine on a 9x9 grid (4,950 used
    # features for 330 decisions), while the 3/3/2, 4/4/2 alternative caused
    # excessive online interference.  This validated middle resolution keeps
    # the proposed 8/8/4 active groups but increases spatial sharing.
    position_tilings: int = 8
    relative_tilings: int = 8
    joint_tilings: int = 4
    position_tiles: int = 4
    relative_tiles: int = 4
    joint_tiles: int = 3
    include_local_geometry: bool = True
    iht_size: int = 8192


class MultiGroupTileCoder:
    """Absolute, relative, joint, local-geometry, and action-bias groups."""

    representation_name = "multi_group_tile_coding"

    def __init__(self, width: int, height: int, config: FeatureConfig = FeatureConfig()):
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.iht = IndexHashTable(config.iht_size)

    @property
    def size(self) -> int:
        return self.iht.size

    @property
    def nominal_active_count(self) -> int:
        return (
            self.config.position_tilings
            + self.config.relative_tilings
            + self.config.joint_tilings
            + int(self.config.include_local_geometry)
            + 1
        )

    @property
    def step_size_denominator(self) -> float:
        return float(self.nominal_active_count)

    def active(self, observation: Sequence[int], action: int, readonly: bool = False) -> np.ndarray:
        x, y, gx, gy, wall_mask = [int(value) for value in observation]
        position = self._position(x, y, self.config.position_tiles)
        relative = self._relative(gx - x, gy - y, self.config.relative_tiles)
        joint_position = self._position(x, y, self.config.joint_tiles)
        joint_relative = self._relative(gx - x, gy - y, self.config.joint_tiles)
        active = []
        active.extend(tiles(
            self.iht, self.config.position_tilings, position,
            ("position", int(action)), readonly=readonly,
        ))
        active.extend(tiles(
            self.iht, self.config.relative_tilings, relative,
            ("relative", int(action)), readonly=readonly,
        ))
        if self.config.joint_tilings:
            active.extend(tiles(
                self.iht, self.config.joint_tilings,
                joint_position + joint_relative,
                ("joint", int(action)), readonly=readonly,
            ))
        if self.config.include_local_geometry:
            local = self.iht.get_index(("local", wall_mask, int(action)), readonly=readonly)
            if local >= 0:
                active.append(local)
        bias = self.iht.get_index(("bias", int(action)), readonly=readonly)
        if bias >= 0:
            active.append(bias)
        return np.unique(np.asarray(active, dtype=np.int64))

    def feature_values(self, observation: Sequence[int], action: int, readonly: bool = False):
        indices = self.active(observation, action, readonly=readonly)
        return indices, np.ones(len(indices), dtype=np.float64)

    def preallocate(self, observations: Sequence[Sequence[int]], num_actions: int = 5) -> None:
        for observation in observations:
            for action in range(num_actions):
                self.active(observation, action, readonly=False)

    def state_dict(self) -> Dict[str, object]:
        return {"iht": self.iht.state_dict()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.iht.load_state_dict(state["iht"])

    def _position(self, x: int, y: int, tiles_per_dimension: int) -> list:
        return [
            x / max(1, self.width - 1) * tiles_per_dimension,
            y / max(1, self.height - 1) * tiles_per_dimension,
        ]

    def _relative(self, dx: int, dy: int, tiles_per_dimension: int) -> list:
        return [
            (dx / max(1, self.width - 1) + 1.0) * 0.5 * tiles_per_dimension,
            (dy / max(1, self.height - 1) + 1.0) * 0.5 * tiles_per_dimension,
        ]
