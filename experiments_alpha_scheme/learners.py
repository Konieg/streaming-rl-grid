"""Trace-free linear differential TD/Sarsa learner shared by Phases 1--3."""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from experiments.phase0.protocol import (
    ALPHA_MAX,
    ALPHA_MIN,
    INITIAL_ADAPTIVE_ALPHA,
    META_STEP,
    REWARD_RATE_STEP,
)


class LinearDifferentialLearner:
    """Linear semi-gradient learner with fixed or per-feature TIDBD step sizes."""

    def __init__(
        self,
        dimension: int,
        groups: Sequence[str],
        fixed_alpha: Optional[float] = None,
        adaptive: bool = False,
    ):
        if adaptive == (fixed_alpha is not None):
            raise ValueError("Choose exactly one of fixed_alpha or adaptive=True.")
        if fixed_alpha is not None and fixed_alpha <= 0.0:
            raise ValueError("fixed_alpha must be positive.")
        self.dimension = int(dimension)
        self.groups = np.asarray(groups)
        if self.groups.shape != (self.dimension,):
            raise ValueError("groups must have one entry per feature")
        self.fixed_alpha = fixed_alpha
        self.adaptive = bool(adaptive)
        self.weights = np.zeros(self.dimension, dtype=np.float64)
        self.average_reward = 0.0
        self.beta = np.full(
            self.dimension, math.log(INITIAL_ADAPTIVE_ALPHA), dtype=np.float64
        )
        self.h = np.zeros(self.dimension, dtype=np.float64)
        self.cumulative_abs_update = np.zeros(self.dimension, dtype=np.float64)

    @property
    def alphas(self) -> np.ndarray:
        if self.adaptive:
            return np.exp(self.beta)
        return np.full(self.dimension, float(self.fixed_alpha), dtype=np.float64)

    def value(self, features: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        weights = self.weights if mask is None else self.weights * mask
        return float(np.dot(weights, features))

    def update(
        self, features: np.ndarray, reward: float, next_features: np.ndarray
    ) -> Dict[str, float]:
        delta = float(
            reward - self.average_reward
            + self.value(next_features) - self.value(features)
        )
        self.average_reward += REWARD_RATE_STEP * delta
        if self.adaptive:
            self.beta += META_STEP * delta * features * self.h
            np.clip(self.beta, math.log(ALPHA_MIN), math.log(ALPHA_MAX), out=self.beta)
            alphas = np.exp(self.beta)
            update = alphas * delta * features
            self.weights += update
            self.h = (
                self.h * np.maximum(0.0, 1.0 - alphas * features * features)
                + alphas * delta * features
            )
        else:
            update = float(self.fixed_alpha) * delta * features
            self.weights += update
        self.cumulative_abs_update += np.abs(update)
        if not np.all(np.isfinite(self.weights)) or np.max(np.abs(self.weights)) > 1e6:
            raise FloatingPointError("Linear weights became numerically unstable.")
        return {"delta": delta, "update_energy": float(np.dot(update, update))}

    def diagnostics(self) -> Dict[str, object]:
        alphas = self.alphas
        return {
            "weight_l2_norm": float(np.linalg.norm(self.weights)),
            "weight_max_abs": float(np.max(np.abs(self.weights))),
            "average_reward_estimate": float(self.average_reward),
            "alpha_p10": float(np.percentile(alphas, 10)),
            "alpha_median": float(np.median(alphas)),
            "alpha_p90": float(np.percentile(alphas, 90)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "beta_lower_bound_count": int(np.count_nonzero(self.beta <= math.log(ALPHA_MIN))),
            "beta_upper_bound_count": int(np.count_nonzero(self.beta >= math.log(ALPHA_MAX))),
            "alpha_by_group": {
                str(group): {
                    "p10": float(np.percentile(alphas[self.groups == group], 10)),
                    "median": float(np.median(alphas[self.groups == group])),
                    "p90": float(np.percentile(alphas[self.groups == group], 90)),
                }
                for group in np.unique(self.groups)
            },
            "cumulative_abs_update_by_group": {
                str(group): float(np.sum(self.cumulative_abs_update[self.groups == group]))
                for group in np.unique(self.groups)
            },
        }
