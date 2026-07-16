"""Backward-compatible imports for algorithms moved to :mod:`stream_rl_grid.algo`."""

from .algo import (
    BaseControlAgent,
    DifferentialAdaptiveEpsilonSarsa,
    DifferentialSarsa,
    DifferentialSarsaTIDBD,
    DifferentialTrueOnlineSarsa,
    create_agent,
)

__all__ = [
    "BaseControlAgent",
    "DifferentialAdaptiveEpsilonSarsa",
    "DifferentialSarsa",
    "DifferentialSarsaTIDBD",
    "DifferentialTrueOnlineSarsa",
    "create_agent",
]
