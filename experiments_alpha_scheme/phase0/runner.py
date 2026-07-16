"""Execution and persistence shared by P0.1 and P0.2."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from stream_rl_grid.environment import ContinualWindyGridWorld

from .metrics import summarize_trace
from .protocol import (
    CONDITIONS,
    EPSILON,
    FIXED_ALPHAS,
    NUM_ACTIONS,
    PROBE_INTERVAL,
    PROBE_STEPS,
    Condition,
    frozen_protocol,
)
from .tabular import Observation, TabularDifferentialLearner, state_action
from experiments.result_schema import make_result_bundle, make_run_record, write_result_bundle


def _mechanisms() -> List[Tuple[str, Dict[str, object]]]:
    mechanisms = [
        ("fixed_%.2f" % alpha, {"fixed_alpha": alpha}) for alpha in FIXED_ALPHAS
    ]
    mechanisms.append(("adaptive", {"adaptive": True}))
    return mechanisms


def _uniform_action(rng: np.random.Generator) -> int:
    return int(rng.integers(NUM_ACTIONS))


def _greedy_action(
    learner: TabularDifferentialLearner,
    observation: Observation,
    rng: np.random.Generator,
) -> int:
    values = learner.values(observation, NUM_ACTIONS)
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(candidates[int(rng.integers(len(candidates)))])


def _epsilon_greedy_action(
    learner: TabularDifferentialLearner,
    observation: Observation,
    rng: np.random.Generator,
) -> int:
    if rng.random() < EPSILON:
        return _uniform_action(rng)
    return _greedy_action(learner, observation, rng)


def _epsilon_greedy_entropy(
    learner: TabularDifferentialLearner, observation: Observation
) -> float:
    values = learner.values(observation, NUM_ACTIONS)
    greedy = np.flatnonzero(np.isclose(values, np.max(values)))
    probabilities = np.full(NUM_ACTIONS, EPSILON / NUM_ACTIONS)
    probabilities[greedy] += (1.0 - EPSILON) / len(greedy)
    return float(-np.sum(probabilities * np.log(probabilities)))


def _mode_label(environment: ContinualWindyGridWorld, condition: Condition) -> str:
    if condition.name == "seasonal_wind":
        return "wind_%d" % environment.wind_phase
    if condition.name == "hidden_context":
        return "context_%d" % environment.context_index
    if condition.name == "moving_goal":
        return "goal_%d_%d" % environment.goal
    return "stationary"


def _prediction_probe(
    learner: TabularDifferentialLearner, condition: Condition, seed: int
) -> Dict[str, float]:
    environment = ContinualWindyGridWorld(condition.make_environment_config(seed))
    observation, _ = environment.reset(seed)
    rng = np.random.default_rng(seed + 1)
    action = _uniform_action(rng)
    squared_errors = []
    for _ in range(PROBE_STEPS):
        next_observation, reward, _, _, _ = environment.step(action)
        next_action = _uniform_action(rng)
        delta = (
            reward
            - learner.average_reward
            + learner.value(next_observation, next_action)
            - learner.value(observation, action)
        )
        squared_errors.append(delta * delta)
        observation, action = next_observation, next_action
    return {"squared_td_error": float(np.mean(squared_errors))}


def _control_probe(
    learner: TabularDifferentialLearner, condition: Condition, seed: int
) -> Dict[str, float]:
    environment = ContinualWindyGridWorld(condition.make_environment_config(seed))
    observation, _ = environment.reset(seed)
    rng = np.random.default_rng(seed + 1)
    rewards, goals, collisions = [], [], []
    for _ in range(PROBE_STEPS):
        action = _greedy_action(learner, observation, rng)
        observation, reward, _, _, info = environment.step(action)
        rewards.append(reward)
        goals.append(info["goal_reached"])
        collisions.append(info["collision"])
    return {
        "mean_reward": float(np.mean(rewards)),
        "goal_rate_per_1000": float(np.mean(goals) * 1000.0),
        "collision_rate": float(np.mean(collisions)),
    }


def run_one(
    task: str,
    condition: Condition,
    mechanism_name: str,
    learner_kwargs: Dict[str, object],
    seed: int,
    steps: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    if task not in ("prediction", "control"):
        raise ValueError("task must be prediction or control")
    environment = ContinualWindyGridWorld(condition.make_environment_config(seed))
    observation, _ = environment.reset(seed)
    learner = TabularDifferentialLearner(**learner_kwargs)
    action_rng = np.random.default_rng(seed + 10_000)
    choose_action = _uniform_action if task == "prediction" else None
    action = (
        choose_action(action_rng)
        if task == "prediction"
        else _epsilon_greedy_action(learner, observation, action_rng)
    )

    rewards = np.full(steps, np.nan)
    squared_td_errors = np.full(steps, np.nan)
    collisions = np.full(steps, np.nan)
    goals = np.full(steps, np.nan)
    actions = np.full(steps, -1, dtype=np.int8)
    observations = np.full((steps, 4), -1, dtype=np.int16)
    policy_entropies = np.full(steps, math.log(NUM_ACTIONS))
    mode_labels: List[str] = []
    change_steps: List[int] = []
    visited = set()
    probe_steps = [0]
    probe_modes = [_mode_label(environment, condition)]
    probe_values = [
        (_prediction_probe if task == "prediction" else _control_probe)(
            learner, condition, seed + 20_000
        )
    ]
    stable = True
    failure = None

    for step in range(steps):
        mode_labels.append(_mode_label(environment, condition))
        observations[step] = observation[:4]
        actions[step] = action
        visited.add(state_action(observation, action))
        if task == "control":
            policy_entropies[step] = _epsilon_greedy_entropy(learner, observation)

        next_observation, reward, _, _, info = environment.step(action)
        next_action = (
            _uniform_action(action_rng)
            if task == "prediction"
            else _epsilon_greedy_action(learner, next_observation, action_rng)
        )
        try:
            delta = learner.update(
                observation, action, reward, next_observation, next_action
            )
        except FloatingPointError as exc:
            stable = False
            failure = str(exc)
            break

        rewards[step] = reward
        squared_td_errors[step] = delta * delta
        collisions[step] = float(info["collision"])
        goals[step] = float(info["goal_reached"])
        if info["events"]:
            change_steps.append(step + 1)
        observation, action = next_observation, next_action

        if (step + 1) % PROBE_INTERVAL == 0:
            probe_steps.append(step + 1)
            probe_modes.append(_mode_label(environment, condition))
            probe_values.append(
                (_prediction_probe if task == "prediction" else _control_probe)(
                    learner, condition, seed + 20_000
                )
            )

    completed_steps = int(np.count_nonzero(np.isfinite(rewards)))
    trace_summary = summarize_trace(
        rewards[:completed_steps],
        squared_td_errors[:completed_steps],
        collisions[:completed_steps],
        goals[:completed_steps],
        mode_labels[:completed_steps],
        policy_entropies[:completed_steps],
        task,
    )
    summary: Dict[str, object] = {
        "task": task,
        "condition": condition.name,
        "mechanism": mechanism_name,
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "numerically_stable": stable,
        "failure": failure,
        "unique_state_actions_visited": len(visited),
        "change_steps": change_steps,
        **trace_summary,
        **learner.diagnostics(),
    }
    probe_keys = sorted(probe_values[0])
    summary["a_probes"] = [
        {"step": int(probe_step), "training_mode": probe_mode, **probe_value}
        for probe_step, probe_mode, probe_value in zip(
            probe_steps, probe_modes, probe_values
        )
    ]
    arrays = {
        "reward": rewards,
        "squared_td_error": squared_td_errors,
        "collision": collisions,
        "goal": goals,
        "action": actions,
        "observation": observations,
        "policy_entropy": policy_entropies,
        "mode": np.asarray(mode_labels, dtype="U32"),
        "probe_step": np.asarray(probe_steps, dtype=np.int64),
    }
    for key in probe_keys:
        arrays["probe_%s" % key] = np.asarray(
            [value[key] for value in probe_values], dtype=float
        )
    return summary, arrays


def _aggregate_runs(summaries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["condition"], summary["mechanism"])].append(summary)
    scalar_fields = (
        ("mean_reward", "mean_reward"),
        ("mean_squared_td_error", "online_squared_td_error"),
        ("goal_rate_per_1000", "goal_rate_per_1000"),
        ("collision_rate", "collision_rate"),
        ("mean_policy_entropy", "mean_policy_entropy"),
        ("average_reward_estimate", "average_reward_estimate"),
        ("alpha_median", "alpha_median"),
    )
    aggregates: List[Dict[str, object]] = []
    for (condition, mechanism), runs in sorted(groups.items()):
        aggregate: Dict[str, object] = {
            "condition": condition,
            "mechanism": mechanism,
            "num_runs": len(runs),
            "num_stable_runs": sum(bool(run["numerically_stable"]) for run in runs),
            "metrics": {},
        }
        for source_field, schema_field in scalar_fields:
            values = np.asarray([float(run[source_field]) for run in runs], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            standard_error = (
                0.0 if values.size == 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))
            )
            aggregate["metrics"][schema_field] = {
                "mean": float(np.mean(values)),
                "standard_error": standard_error,
                "ci95": [
                    float(np.mean(values) - 1.96 * standard_error),
                    float(np.mean(values) + 1.96 * standard_error),
                ],
            }
        aggregates.append(aggregate)
    return aggregates


def run_matrix(task: str, steps: int, seeds: Sequence[int], output: Path) -> List[Dict[str, object]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    run_records: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        for mechanism_name, learner_kwargs in _mechanisms():
            for seed in seeds:
                summary, arrays = run_one(
                    task, condition, mechanism_name, learner_kwargs, int(seed), steps
                )
                run_name = "%s__%s__seed_%d" % (
                    condition.name,
                    mechanism_name,
                    seed,
                )
                np.savez_compressed(output / (run_name + ".npz"), **arrays)
                summary["run_id"] = run_name
                summaries.append(summary)
                run_records.append(make_run_record(summary, run_name + ".npz"))
                print(
                    "%-15s %-10s seed=%d reward=%7.3f td_mse=%9.4f stable=%s"
                    % (
                        condition.name,
                        mechanism_name,
                        seed,
                        summary["mean_reward"],
                        summary["mean_squared_td_error"],
                        summary["numerically_stable"],
                    )
                )

    subexperiment = "p0_1_prediction" if task == "prediction" else "p0_2_control"
    protocol = frozen_protocol()
    protocol.update({"steps": steps, "seeds": [int(seed) for seed in seeds]})
    payload = make_result_bundle(
        phase="phase0",
        subexperiment=subexperiment,
        task=task,
        protocol=protocol,
        runs=run_records,
        aggregates=_aggregate_runs(summaries),
    )
    write_result_bundle(output / "summary.json", payload)
    return summaries
