"""The single explicit feature representation used by Phases 1--3."""

from typing import Sequence

import numpy as np


NUM_ACTIONS = 5
BASES_PER_ACTION = 11
BASE_DIMENSION = BASES_PER_ACTION * NUM_ACTIONS
NUISANCE_DIMENSION = 16
SELECTION_DIMENSION = BASE_DIMENSION + NUISANCE_DIMENSION


class GridFeatureEncoder:
    """Unit-norm D=55 action-value features specified in the experiment plan."""

    dimension = BASE_DIMENSION

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


class NuisanceFeatureEncoder(GridFeatureEncoder):
    """D=71 extension with a controlled independent one-hot nuisance group."""

    dimension = SELECTION_DIMENSION

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.groups = np.concatenate(
            [self.groups, np.full(NUISANCE_DIMENSION, "nuisance", dtype="U32")]
        )
        self.names = self.names + tuple(
            "nuisance/%d" % index for index in range(NUISANCE_DIMENSION)
        )

    def encode(
        self, observation: Sequence[int], action: int, nuisance_index: int
    ) -> np.ndarray:
        if not 0 <= int(nuisance_index) < NUISANCE_DIMENSION:
            raise ValueError("nuisance_index must be in [0, 15]")
        features = np.zeros(self.dimension, dtype=np.float64)
        features[:BASE_DIMENSION] = super().encode(observation, action)
        features[BASE_DIMENSION + int(nuisance_index)] = 1.0
        return features / np.sqrt(2.0)
