"""Selectable linear feature representations for action-value learning."""

from typing import Sequence, Tuple

import numpy as np

from .config import AgentConfig, EnvironmentConfig
from .tile_coder import DualTileCoder


TILE_CODING = "tile_coding"
HANDCRAFTED_LFA = "handcrafted_lfa"
HANDCRAFTED_LFA_NUISANCE = "handcrafted_lfa_nuisance"
FEATURE_REPRESENTATIONS = (
    TILE_CODING,
    HANDCRAFTED_LFA,
    HANDCRAFTED_LFA_NUISANCE,
)
FEATURE_REPRESENTATION_LABELS = {
    TILE_CODING: "Tile coding",
    HANDCRAFTED_LFA: "Hand-crafted linear features (D=55)",
    HANDCRAFTED_LFA_NUISANCE: "Hand-crafted + nuisance features (D=71)",
}

NUM_ACTIONS = 5
BASES_PER_ACTION = 11
BASE_DIMENSION = BASES_PER_ACTION * NUM_ACTIONS
NUISANCE_DIMENSION = 16
SELECTION_DIMENSION = BASE_DIMENSION + NUISANCE_DIMENSION


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


class NuisanceFeatureEncoder(GridFeatureEncoder):
    """Collaborator's D=71 extension with independent one-hot nuisance input.

    The trainer appends the current nuisance index to the ordinary five-value
    environment observation.  Sharing that index across candidate actions is
    important: it makes it an observed input at one stream time, rather than a
    fresh random draw for every value-function query.
    """

    dimension = SELECTION_DIMENSION
    size = SELECTION_DIMENSION
    representation_name = HANDCRAFTED_LFA_NUISANCE
    requires_nuisance = True

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.groups = np.concatenate(
            [self.groups, np.full(NUISANCE_DIMENSION, "nuisance", dtype="U32")]
        )
        self.names = self.names + tuple(
            "nuisance/%d" % index for index in range(NUISANCE_DIMENSION)
        )

    @staticmethod
    def _nuisance_index(observation: Sequence[int]) -> int:
        if len(observation) < 6:
            raise ValueError(
                "D=71 features require a nuisance index appended to the observation."
            )
        index = int(observation[5])
        if not 0 <= index < NUISANCE_DIMENSION:
            raise ValueError("nuisance index must be in [0, 15]")
        return index

    def encode(self, observation: Sequence[int], action: int) -> np.ndarray:
        features = np.zeros(self.dimension, dtype=np.float64)
        features[:BASE_DIMENSION] = super().encode(observation, action)
        features[BASE_DIMENSION + self._nuisance_index(observation)] = 1.0
        return features / np.sqrt(2.0)

    def feature_values(
        self, observation: Sequence[int], action: int, readonly: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        del readonly
        main_indices, main_values = super().feature_values(observation, action)
        nuisance_index = BASE_DIMENSION + self._nuisance_index(observation)
        indices = np.concatenate(
            [main_indices, np.asarray([nuisance_index], dtype=np.int64)]
        )
        values = np.concatenate([main_values, np.asarray([1.0])]) / np.sqrt(2.0)
        return indices, values


def create_feature_representation(
    env_config: EnvironmentConfig, agent_config: AgentConfig
):
    if agent_config.feature_representation == TILE_CODING:
        return DualTileCoder(env_config, agent_config)
    if agent_config.feature_representation == HANDCRAFTED_LFA:
        return GridFeatureEncoder(env_config.width, env_config.height)
    if agent_config.feature_representation == HANDCRAFTED_LFA_NUISANCE:
        return NuisanceFeatureEncoder(env_config.width, env_config.height)
    raise ValueError(
        "Unknown feature representation: %s" % agent_config.feature_representation
    )
