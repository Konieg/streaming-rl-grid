"""Lightweight registry for the project's tile-coded TD-control agents."""

from typing import Dict, Type


_AGENT_TYPES: Dict[str, Type] = {}


def register_agent(agent_type):
    name = str(agent_type.algorithm_name)
    if not name or name == "base":
        raise ValueError("A concrete agent must define a non-base algorithm_name.")
    if name in _AGENT_TYPES:
        raise ValueError("Duplicate algorithm registration: %s" % name)
    _AGENT_TYPES[name] = agent_type
    return agent_type


def available_algorithms():
    return tuple(_AGENT_TYPES)


def algorithm_labels():
    return {name: agent_type.display_name for name, agent_type in _AGENT_TYPES.items()}


def algorithm_config_fields():
    return {
        name: tuple(agent_type.extra_config_fields)
        for name, agent_type in _AGENT_TYPES.items()
    }


def create_agent(coder, config, seed=0, num_actions=5):
    try:
        agent_type = _AGENT_TYPES[config.algorithm]
    except KeyError as exc:
        raise ValueError("Unknown training algorithm: %s" % config.algorithm) from exc
    return agent_type(coder, config, seed=seed, num_actions=num_actions)
