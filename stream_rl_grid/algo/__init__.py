"""Algorithm factory for the shared training interface."""

from .base import BaseControlAgent
from .sarsa import DifferentialSarsa
from .tidbd import DifferentialSarsaTIDBD


def create_agent(features, config, seed=0):
    if config.algorithm == "tidbd":
        return DifferentialSarsaTIDBD(features, config, seed=seed)
    if config.algorithm == "sarsa":
        return DifferentialSarsa(features, config, seed=seed)
    raise ValueError("Unknown training algorithm: %s" % config.algorithm)


__all__ = ["BaseControlAgent", "DifferentialSarsa", "DifferentialSarsaTIDBD", "create_agent"]
