"""Algorithm factory for the shared training interface."""

from .base import BaseControlAgent
from .sarsa import DifferentialSarsa
from .tidbd import DifferentialSarsaTIDBD


def create_agent(coder, config, seed=0):
    if config.algorithm == "tidbd":
        return DifferentialSarsaTIDBD(coder, config, seed=seed)
    if config.algorithm == "sarsa":
        return DifferentialSarsa(coder, config, seed=seed)
    raise ValueError("Unknown training algorithm: %s" % config.algorithm)


__all__ = ["BaseControlAgent", "DifferentialSarsa", "DifferentialSarsaTIDBD", "create_agent"]
