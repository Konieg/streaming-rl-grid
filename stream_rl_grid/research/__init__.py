"""Validated research stack for continuing average-reward experiments."""

from .environment import ContinuingGridMDP, StationaryGridSpec
from .oracle import AverageRewardSolution, solve_average_reward
from .representations import MultiGroupTileCoder

__all__ = [
    "AverageRewardSolution",
    "ContinuingGridMDP",
    "MultiGroupTileCoder",
    "StationaryGridSpec",
    "solve_average_reward",
]
