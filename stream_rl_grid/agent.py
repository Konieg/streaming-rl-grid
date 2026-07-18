"""Backward-compatible imports for algorithms moved to :mod:`stream_rl_grid.algo`."""

from .algo import (
    BaseControlAgent,
    DifferentialDynaQ,
    DifferentialDynaQLambda,
    DifferentialQLambda,
    DifferentialQLearning,
    DifferentialSarsa,
    DifferentialSarsaTIDBD,
    create_agent,
)

__all__ = [
    "BaseControlAgent",
    "DifferentialDynaQ",
    "DifferentialDynaQLambda",
    "DifferentialQLambda",
    "DifferentialQLearning",
    "DifferentialSarsa",
    "DifferentialSarsaTIDBD",
    "create_agent",
]
