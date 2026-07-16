"""Shared D=55/D=71 streaming runner for Phases 1--3."""

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stream_rl_grid.environment import ContinualWindyGridWorld

from experiments.features import (
    NUISANCE_DIMENSION,
    GridFeatureEncoder,
    NuisanceFeatureEncoder,
)
from experiments.learners import LinearDifferentialLearner
from experiments.phase0.metrics import summarize_trace
from experiments.phase0.protocol import (
    CONDITIONS,
    EPSILON,
    FIXED_ALPHAS,
    NUM_ACTIONS,
    PROBE_INTERVAL,
    PROBE_STEPS,
    Condition,
    frozen_protocol,
)
from experiments.result_schema import (
    aggregate_run_records,
    make_result_bundle,
    make_run_record,
    write_result_bundle,
)


Mechanism = Tuple[str, Dict[str, object]]


def fixed_mechanisms() -> Tuple[Mechanism, ...]:
    return tuple(
        ("fixed_%.2f" % alpha, {"fixed_alpha": alpha}) for alpha in FIXED_ALPHAS
    )


def adaptive_mechanism() -> Tuple[Mechanism, ...]:
    return (("adaptive", {"adaptive": True}),)


def all_mechanisms() -> Tuple[Mechanism, ...]:
    return fixed_mechanisms() + adaptive_mechanism()


def _mode_label(environment: ContinualWindyGridWorld, condition: Condition) -> str:
    if condition.name == "seasonal_wind":
        return "wind_%d" % environment.wind_phase
    if condition.name == "hidden_context":
        return "context_%d" % environment.context_index
    if condition.name == "moving_goal":
        return "goal_%d_%d" % environment.goal
    return "stationary"


def _encode(
    encoder: GridFeatureEncoder,
    observation: Sequence[int],
    action: int,
    nuisance: Optional[int],
) -> np.ndarray:
    if isinstance(encoder, NuisanceFeatureEncoder):
        if nuisance is None:
            raise ValueError("D=71 encoding requires a nuisance index")
        return encoder.encode(observation, action, nuisance)
    return encoder.encode(observation, action)


def _action_values(
    learner: LinearDifferentialLearner,
    encoder: GridFeatureEncoder,
    observation: Sequence[int],
    nuisance: Optional[int],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(
        [
            learner.value(_encode(encoder, observation, action, nuisance), mask)
            for action in range(NUM_ACTIONS)
        ]
    )


def _uniform_action(rng: np.random.Generator) -> int:
    return int(rng.integers(NUM_ACTIONS))


def _greedy_action(
    learner: LinearDifferentialLearner,
    encoder: GridFeatureEncoder,
    observation: Sequence[int],
    nuisance: Optional[int],
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
) -> int:
    values = _action_values(learner, encoder, observation, nuisance, mask)
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(candidates[int(rng.integers(len(candidates)))])


def _epsilon_greedy_action(
    learner: LinearDifferentialLearner,
    encoder: GridFeatureEncoder,
    observation: Sequence[int],
    nuisance: Optional[int],
    rng: np.random.Generator,
) -> int:
    if rng.random() < EPSILON:
        return _uniform_action(rng)
    return _greedy_action(learner, encoder, observation, nuisance, rng)


def _policy_entropy(
    learner: LinearDifferentialLearner,
    encoder: GridFeatureEncoder,
    observation: Sequence[int],
    nuisance: Optional[int],
) -> float:
    values = _action_values(learner, encoder, observation, nuisance)
    greedy = np.flatnonzero(np.isclose(values, np.max(values)))
    probabilities = np.full(NUM_ACTIONS, EPSILON / NUM_ACTIONS)
    probabilities[greedy] += (1.0 - EPSILON) / len(greedy)
    return float(-np.sum(probabilities * np.log(probabilities)))


def evaluate_frozen(
    learner: LinearDifferentialLearner,
    encoder: GridFeatureEncoder,
    condition: Condition,
    task: str,
    seed: int,
    mask: Optional[np.ndarray] = None,
    burn_in_steps: int = 0,
) -> Dict[str, float]:
    """Evaluate without modifying learner, environment, or policy state."""
    environment = ContinualWindyGridWorld(condition.make_environment_config(seed))
    observation, _ = environment.reset(seed)
    action_rng = np.random.default_rng(seed + 1)
    nuisance_rng = np.random.default_rng(seed + 2)
    nuisance = (
        int(nuisance_rng.integers(NUISANCE_DIMENSION))
        if isinstance(encoder, NuisanceFeatureEncoder) else None
    )
    for _ in range(int(burn_in_steps)):
        observation, _, _, _, _ = environment.step(4)
        if isinstance(encoder, NuisanceFeatureEncoder):
            nuisance = int(nuisance_rng.integers(NUISANCE_DIMENSION))
    squared_errors, rewards, goals, collisions = [], [], [], []
    action = (
        _uniform_action(action_rng)
        if task == "prediction"
        else _greedy_action(learner, encoder, observation, nuisance, action_rng, mask)
    )
    for _ in range(PROBE_STEPS):
        next_observation, reward, _, _, info = environment.step(action)
        next_nuisance = (
            int(nuisance_rng.integers(NUISANCE_DIMENSION))
            if isinstance(encoder, NuisanceFeatureEncoder) else None
        )
        next_action = (
            _uniform_action(action_rng)
            if task == "prediction"
            else _greedy_action(
                learner, encoder, next_observation, next_nuisance, action_rng, mask
            )
        )
        features = _encode(encoder, observation, action, nuisance)
        next_features = _encode(encoder, next_observation, next_action, next_nuisance)
        delta = (
            reward - learner.average_reward
            + learner.value(next_features, mask) - learner.value(features, mask)
        )
        squared_errors.append(delta * delta)
        rewards.append(reward)
        goals.append(info["goal_reached"])
        collisions.append(info["collision"])
        observation, action, nuisance = next_observation, next_action, next_nuisance
    return {
        "squared_td_error": float(np.mean(squared_errors)),
        "mean_reward": float(np.mean(rewards)),
        "goal_rate_per_1000": float(np.mean(goals) * 1000.0),
        "collision_rate": float(np.mean(collisions)),
    }


def run_one(
    task: str,
    condition: Condition,
    mechanism_name: str,
    learner_kwargs: Mapping[str, object],
    seed: int,
    steps: int,
    nuisance_features: bool = False,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], LinearDifferentialLearner, GridFeatureEncoder]:
    if task not in ("prediction", "control"):
        raise ValueError("task must be prediction or control")
    environment = ContinualWindyGridWorld(condition.make_environment_config(seed))
    observation, _ = environment.reset(seed)
    encoder = (
        NuisanceFeatureEncoder(environment.width, environment.height)
        if nuisance_features else GridFeatureEncoder(environment.width, environment.height)
    )
    learner = LinearDifferentialLearner(
        encoder.dimension, encoder.groups, **dict(learner_kwargs)
    )
    action_rng = np.random.default_rng(seed + 10_000)
    nuisance_rng = np.random.default_rng(seed + 30_000)
    nuisance = int(nuisance_rng.integers(NUISANCE_DIMENSION)) if nuisance_features else None
    action = (
        _uniform_action(action_rng)
        if task == "prediction"
        else _epsilon_greedy_action(learner, encoder, observation, nuisance, action_rng)
    )

    rewards = np.full(steps, np.nan)
    squared_td_errors = np.full(steps, np.nan)
    collisions = np.full(steps, np.nan)
    goals = np.full(steps, np.nan)
    update_energy = np.full(steps, np.nan)
    policy_entropies = np.full(steps, math.log(NUM_ACTIONS))
    alpha_p10 = np.full(steps, np.nan)
    alpha_median = np.full(steps, np.nan)
    alpha_p90 = np.full(steps, np.nan)
    actions = np.full(steps, -1, dtype=np.int8)
    observations = np.full((steps, 4), -1, dtype=np.int16)
    mode_labels: List[str] = []
    change_steps: List[int] = []
    visited = set()
    probe_steps = [0]
    probe_modes = [_mode_label(environment, condition)]
    probe_values = [
        evaluate_frozen(learner, encoder, condition, task, seed + 20_000)
    ]
    stable, failure = True, None

    for step in range(steps):
        mode_labels.append(_mode_label(environment, condition))
        observations[step] = observation[:4]
        actions[step] = action
        visited.add(tuple(observation[:4]) + (action,))
        if task == "control":
            policy_entropies[step] = _policy_entropy(
                learner, encoder, observation, nuisance
            )
        alphas = learner.alphas
        alpha_p10[step], alpha_median[step], alpha_p90[step] = np.percentile(
            alphas, (10, 50, 90)
        )

        next_observation, reward, _, _, info = environment.step(action)
        next_nuisance = (
            int(nuisance_rng.integers(NUISANCE_DIMENSION)) if nuisance_features else None
        )
        next_action = (
            _uniform_action(action_rng)
            if task == "prediction"
            else _epsilon_greedy_action(
                learner, encoder, next_observation, next_nuisance, action_rng
            )
        )
        features = _encode(encoder, observation, action, nuisance)
        next_features = _encode(encoder, next_observation, next_action, next_nuisance)
        try:
            update = learner.update(features, reward, next_features)
        except FloatingPointError as exc:
            stable, failure = False, str(exc)
            break
        rewards[step] = reward
        squared_td_errors[step] = update["delta"] ** 2
        update_energy[step] = update["update_energy"]
        collisions[step] = float(info["collision"])
        goals[step] = float(info["goal_reached"])
        if info["events"]:
            change_steps.append(step + 1)
        observation, action, nuisance = next_observation, next_action, next_nuisance
        if (step + 1) % PROBE_INTERVAL == 0:
            probe_steps.append(step + 1)
            probe_modes.append(_mode_label(environment, condition))
            probe_values.append(
                evaluate_frozen(learner, encoder, condition, task, seed + 20_000)
            )

    completed = int(np.count_nonzero(np.isfinite(rewards)))
    trace_summary = summarize_trace(
        rewards[:completed], squared_td_errors[:completed], collisions[:completed],
        goals[:completed], mode_labels[:completed], policy_entropies[:completed], task,
    )
    diagnostics = learner.diagnostics()
    diagnostics.update(
        {
            "feature_dimension": encoder.dimension,
            "unique_state_actions_visited": len(visited),
            "mean_update_energy": float(np.nanmean(update_energy[:completed])),
        }
    )
    summary: Dict[str, object] = {
        "task": task,
        "condition": condition.name,
        "mechanism": mechanism_name,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed,
        "numerically_stable": stable,
        "failure": failure,
        "change_steps": change_steps,
        **trace_summary,
        "a_probes": [
            {"step": int(probe_step), "training_mode": mode, **value}
            for probe_step, mode, value in zip(probe_steps, probe_modes, probe_values)
        ],
        "diagnostics": diagnostics,
    }
    arrays = {
        "reward": rewards,
        "squared_td_error": squared_td_errors,
        "collision": collisions,
        "goal": goals,
        "update_energy": update_energy,
        "policy_entropy": policy_entropies,
        "alpha_p10": alpha_p10,
        "alpha_median": alpha_median,
        "alpha_p90": alpha_p90,
        "action": actions,
        "observation": observations,
        "mode": np.asarray(mode_labels, dtype="U32"),
        "probe_step": np.asarray(probe_steps, dtype=np.int64),
    }
    return summary, arrays, learner, encoder


def run_matrix(
    phase: str,
    subexperiment: str,
    task: str,
    mechanisms: Sequence[Mechanism],
    steps: int,
    seeds: Sequence[int],
    output: Path,
    conditions: Sequence[Condition] = CONDITIONS,
    nuisance_features: bool = False,
    save_models: bool = False,
) -> List[Dict[str, object]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in conditions:
        for mechanism_name, learner_kwargs in mechanisms:
            for seed in seeds:
                summary, arrays, learner, encoder = run_one(
                    task, condition, mechanism_name, learner_kwargs, int(seed), steps,
                    nuisance_features=nuisance_features,
                )
                run_name = "%s__%s__seed_%d" % (condition.name, mechanism_name, seed)
                trace_name = run_name + ".npz"
                np.savez_compressed(output / trace_name, **arrays)
                model_name = None
                if save_models:
                    model_name = run_name + "__model.npz"
                    np.savez_compressed(
                        output / model_name,
                        weights=learner.weights,
                        beta=learner.beta,
                        h=learner.h,
                        alphas=learner.alphas,
                        groups=encoder.groups,
                        names=np.asarray(encoder.names),
                        average_reward=np.asarray([learner.average_reward]),
                    )
                summary["run_id"] = run_name
                record = make_run_record(summary, trace_name, model_name)
                records.append(record)
                print(
                    "%-15s %-10s seed=%d reward=%7.3f td_mse=%9.4f stable=%s"
                    % (
                        condition.name, mechanism_name, seed,
                        record["metrics"]["mean_reward"],
                        record["metrics"]["online_squared_td_error"],
                        record["status"]["numerically_stable"],
                    )
                )
    protocol = frozen_protocol()
    protocol.update(
        {
            "steps": steps,
            "seeds": [int(seed) for seed in seeds],
            "feature_dimension": 71 if nuisance_features else 55,
            "nuisance_features": 16 if nuisance_features else 0,
            "selected_conditions": [condition.name for condition in conditions],
        }
    )
    bundle = make_result_bundle(
        phase, subexperiment, task, protocol, records, aggregate_run_records(records)
    )
    write_result_bundle(output / "summary.json", bundle)
    return records
