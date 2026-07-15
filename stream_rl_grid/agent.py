"""Backward-compatible imports for algorithms moved to :mod:`stream_rl_grid.algo`."""

from .algo import BaseControlAgent, DifferentialSarsa, DifferentialSarsaTIDBD, create_agent

__all__ = ["BaseControlAgent", "DifferentialSarsa", "DifferentialSarsaTIDBD", "create_agent"]
