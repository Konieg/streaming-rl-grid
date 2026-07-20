"""Factory for state-action value representations."""

from .config import REPRESENTATIONS, AgentConfig, EnvironmentConfig
from .discrete_features import DiscreteStateActionFeatures
from .factorized_features import SparseFactorizedStateActionFeatures


def create_features(env_config: EnvironmentConfig, agent_config: AgentConfig):
    if agent_config.representation == "tabular-one-hot":
        return DiscreteStateActionFeatures(env_config)
    if agent_config.representation == "sparse-factorized":
        return SparseFactorizedStateActionFeatures(env_config)
    raise ValueError("Unknown value representation: %s" % agent_config.representation)


__all__ = [
    "REPRESENTATIONS",
    "DiscreteStateActionFeatures",
    "SparseFactorizedStateActionFeatures",
    "create_features",
]
