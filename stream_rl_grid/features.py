"""Selectable linear feature representations for action-value learning."""

from typing import Sequence, Tuple

import numpy as np

from .config import AgentConfig, EnvironmentConfig
from .tile_coder import DualTileCoder


TILE_CODING = "tile_coding"
HANDCRAFTED_LFA = "handcrafted_lfa"
FEATURE_REPRESENTATIONS = (TILE_CODING, HANDCRAFTED_LFA)
FEATURE_REPRESENTATION_LABELS = {
    TILE_CODING: "Tile coding",
    HANDCRAFTED_LFA: "Hand-crafted linear features (D=55)",
}

NUM_ACTIONS = 5
BASES_PER_ACTION = 11
BASE_DIMENSION = BASES_PER_ACTION * NUM_ACTIONS


class GridFeatureEncoder:
    """The collaborator's unit-norm D=55 action-value representation.

    Each action owns an independent block of eleven weights.  The observation's
    previous-action component is intentionally ignored.
    """

    dimension = BASE_DIMENSION
    size = BASE_DIMENSION
    nominal_active_count = 1
    step_size_denominator = 1.0
    representation_name = HANDCRAFTED_LFA

    def __init__(self, width: int, height: int):
        if width < 2 or height < 2:
            raise ValueError("Grid dimensions must be at least two.")
        self.width = int(width)
        self.height = int(height)
        groups = []
        names = []
        basis_names = (
            "bias", "x", "y", "x2", "y2", "xy",
            "dx", "dy", "dx2", "dy2", "dxdy",
        )
        for action in range(NUM_ACTIONS):
            groups.extend(["absolute_position"] * 6 + ["relative_goal"] * 5)
            names.extend("action_%d/%s" % (action, name) for name in basis_names)
        self.groups = np.asarray(groups, dtype="U32")
        self.names = tuple(names)

    def spatial_basis(self, observation: Sequence[int]) -> np.ndarray:
        x, y, goal_x, goal_y = (int(value) for value in observation[:4])
        x_scaled = 2.0 * x / (self.width - 1) - 1.0
        y_scaled = 2.0 * y / (self.height - 1) - 1.0
        dx = (x - goal_x) / (self.width - 1)
        dy = (y - goal_y) / (self.height - 1)
        basis = np.asarray(
            [
                1.0, x_scaled, y_scaled, x_scaled * x_scaled,
                y_scaled * y_scaled, x_scaled * y_scaled,
                dx, dy, dx * dx, dy * dy, dx * dy,
            ],
            dtype=np.float64,
        )
        return basis / np.linalg.norm(basis)

    def encode(self, observation: Sequence[int], action: int) -> np.ndarray:
        if not 0 <= int(action) < NUM_ACTIONS:
            raise ValueError("action must be in [0, 4]")
        features = np.zeros(BASE_DIMENSION, dtype=np.float64)
        start = int(action) * BASES_PER_ACTION
        features[start:start + BASES_PER_ACTION] = self.spatial_basis(observation)
        return features

    def feature_values(
        self, observation: Sequence[int], action: int, readonly: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        del readonly
        start = int(action) * BASES_PER_ACTION
        values = self.spatial_basis(observation)
        indices = np.arange(start, start + BASES_PER_ACTION, dtype=np.int64)
        nonzero = values != 0.0
        return indices[nonzero], values[nonzero]

    def active(self, observation: Sequence[int], action: int, readonly: bool = False) -> np.ndarray:
        """Return nonzero indices for diagnostics and compatibility helpers."""
        indices, _ = self.feature_values(observation, action, readonly=readonly)
        return indices

    def state_dict(self):
        return {
            "representation": self.representation_name,
            "width": self.width,
            "height": self.height,
        }

    def load_state_dict(self, state) -> None:
        if int(state.get("width", self.width)) != self.width or int(
            state.get("height", self.height)
        ) != self.height:
            raise ValueError("Checkpoint feature encoder grid dimensions do not match.")
        representation = state.get("representation", self.representation_name)
        if representation != self.representation_name:
            raise ValueError("Checkpoint feature representation does not match.")


def create_feature_representation(
    env_config: EnvironmentConfig, agent_config: AgentConfig
):
    if agent_config.feature_representation == TILE_CODING:
        return DualTileCoder(env_config, agent_config)
    if agent_config.feature_representation == HANDCRAFTED_LFA:
        return GridFeatureEncoder(env_config.width, env_config.height)
    raise ValueError(
        "Unknown feature representation: %s" % agent_config.feature_representation
    )
