"""Differential Expected Sarsa(lambda) with fixed or TIDBD step sizes."""

from typing import Sequence

import numpy as np

from .sarsa import DifferentialSarsa
from .tidbd import DifferentialSarsaTIDBD


class ExpectedPolicyBootstrapMixin:
    """Replace the sampled Sarsa bootstrap with its epsilon-greedy expectation."""

    def bootstrap_value(self, next_observation: Sequence[int], next_action: int) -> float:
        del next_action
        values = self.action_values(next_observation, readonly=False)
        probabilities = self.probabilities_from_values(values)
        return float(np.dot(probabilities, values))


class DifferentialExpectedSarsa(ExpectedPolicyBootstrapMixin, DifferentialSarsa):
    """Fixed-step differential Expected Sarsa(lambda) with replacing traces."""

    algorithm_name = "expected_sarsa"


class DifferentialExpectedSarsaTIDBD(ExpectedPolicyBootstrapMixin, DifferentialSarsaTIDBD):
    """Differential Expected Sarsa(lambda) with replacing traces and TIDBD."""

    algorithm_name = "expected_sarsa_tidbd"
