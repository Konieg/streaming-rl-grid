"""Registered tile-coded TD-control algorithms."""

from .base import BaseControlAgent
from .registry import algorithm_config_fields, algorithm_labels, available_algorithms, create_agent

# Importing concrete modules performs their local registry declarations.
from .dyna_q import DifferentialDynaQ
from .dyna_q_plus import DifferentialDynaQPlus
from .dyna_q_lambda import DifferentialDynaQLambda
from .q_lambda import DifferentialQLambda
from .q_learning import DifferentialQLearning
from .sarsa import DifferentialSarsa
from .tidbd import DifferentialSarsaTIDBD


ALGORITHMS = available_algorithms()
ALGORITHM_LABELS = algorithm_labels()
ALGORITHM_CONFIG_FIELDS = algorithm_config_fields()


__all__ = [
    "ALGORITHMS",
    "ALGORITHM_CONFIG_FIELDS",
    "ALGORITHM_LABELS",
    "BaseControlAgent",
    "DifferentialDynaQ",
    "DifferentialDynaQPlus",
    "DifferentialDynaQLambda",
    "DifferentialQLambda",
    "DifferentialQLearning",
    "DifferentialSarsa",
    "DifferentialSarsaTIDBD",
    "create_agent",
]
