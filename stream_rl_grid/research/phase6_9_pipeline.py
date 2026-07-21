"""Autonomous Phase 6--9 continual-MBRL experiment pipeline."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adaptive_models import DynamicOracleModel, FactoredGridModel
from .agents import DifferentialLinearAgent, ResearchAgentConfig
from .continual_environment import (
    ContinualScenario, DynamicContext, DynamicTwoGoalGrid, FrozenDynamicMDP,
    phase6_scenarios,
)
from .models import EmpiricalModel, ExponentialRecencyModel, LatestTransitionModel
from .oracle import evaluate_policy, policy_matrix_from_q, solve_average_reward
from .pipeline import _mean_ci, _write_csv, _write_json
from .representations import FeatureConfig, MultiGroupTileCoder


PROTOCOL_VERSION = 1
DIAGNOSTIC_INTERVAL = 100
PLANNING_STEPS = 5
METHOD_ORDER = (
    "q_learning", "sarsa_lambda", "latest_dyna", "empirical_dyna",
    "ema_dyna", "prioritized_ema", "cafd_uniform", "cafd_lite",
    "cafd_surprise", "oracle_dyna",
)
METHOD_LABELS = {
    "q_learning": "Q-learning",
    "sarsa_lambda": r"SARSA($\lambda$)",
    "latest_dyna": "Latest Dyna",
    "empirical_dyna": "Empirical Dyna",
    "ema_dyna": "EMA Dyna",
    "prioritized_ema": "Prioritized EMA",
    "cafd_uniform": "Factored Dyna",
    "cafd_lite": "CAFD-Lite",
    "cafd_surprise": "CAFD-Surprise",
    "oracle_dyna": "Oracle Dyna",
}
COLORS = {
    "q_learning": "#4C78A8", "sarsa_lambda": "#F58518",
    "latest_dyna": "#E45756", "empirical_dyna": "#9D755D",
    "ema_dyna": "#72B7B2", "prioritized_ema": "#59A14F",
    "cafd_uniform": "#B79A20", "cafd_lite": "#2E7D32",
    "cafd_surprise": "#7B2CBF", "oracle_dyna": "#B279A2",
}


def _model_spec(method: str, environment: DynamicTwoGoalGrid, seed: int, params):
    if method in ("q_learning", "sarsa_lambda"):
        return None
    if method == "latest_dyna":
        return LatestTransitionModel(seed)
    if method == "empirical_dyna":
        return EmpiricalModel(seed)
    if method in ("ema_dyna", "prioritized_ema"):
        return ExponentialRecencyModel(float(params.get("ema_decay", 0.97)), seed)
    if method in ("cafd_uniform", "cafd_lite"):
        return FactoredGridModel(
            environment, learning_rate=float(params.get("factor_rate", 0.05)),
            surprise_adaptive=False, seed=seed,
        )
    if method == "cafd_surprise":
        return FactoredGridModel(
            environment, learning_rate=float(params.get("factor_rate", 0.05)),
            surprise_adaptive=True, seed=seed,
        )
    if method == "oracle_dyna":
        return DynamicOracleModel(environment, seed)
    raise ValueError("unknown Phase 6--9 method: %s" % method)


def _agent_method(method: str) -> str:
    if method == "q_learning":
        return "q_learning"
    if method == "sarsa_lambda":
        return "sarsa_lambda"
    return "dyna"


def _planning_strategy(method: str) -> str:
    if method == "prioritized_ema":
        return "prioritized"
    if method in ("cafd_lite", "cafd_surprise"):
        return "mixed"
    return "uniform"


def _oracle_solution(environment, context, cache):
    key = context.key()
    if key not in cache:
        frozen = FrozenDynamicMDP(environment, context)
        cache[key] = solve_average_reward(frozen)
    return cache[key]


def _outcome_marginal(distribution):
    transition = {}
    expected_reward = 0.0
    for (reward, next_state), probability in distribution.items():
        transition[next_state] = transition.get(next_state, 0.0) + probability
        expected_reward += probability * reward
    return transition, expected_reward


def dynamic_model_error(model, environment, context):
    if model is None:
        return float("nan"), 0.0
    errors = []
    covered = 0
    for state in range(len(environment.states)):
        for action in range(environment.num_actions):
            key = (state, action)
            true_distribution = {}
            for probability, next_state, reward, _, _, _ in environment.transition_distribution(
                state, action, context
            ):
                outcome = (float(reward), int(next_state))
                true_distribution[outcome] = true_distribution.get(outcome, 0.0) + probability
            if key not in getattr(model, "_key_set", set()):
                errors.append(1.0)
                continue
            covered += 1
            predicted = model.distribution(key)
            true_next, true_reward = _outcome_marginal(true_distribution)
            predicted_next, predicted_reward = _outcome_marginal(predicted)
            support = set(true_next) | set(predicted_next)
            transition_tv = 0.5 * sum(
                abs(true_next.get(value, 0.0) - predicted_next.get(value, 0.0))
                for value in support
            )
            reward_error = min(1.0, abs(predicted_reward - true_reward) / 6.0)
            errors.append(0.8 * transition_tv + 0.2 * reward_error)
    total = len(environment.states) * environment.num_actions
    return float(np.mean(errors)), covered / total


def preferred_goal(context: DynamicContext) -> str:
    """Return the currently higher-reward goal (not a route heuristic)."""
    return "A" if context.reward_a >= context.reward_b else "B"


def _paired_bootstrap_ci(values, repetitions: int = 10_000):
    """Deterministic percentile CI for paired seed-level differences."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(60_909)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _apply_loca_intervention(environment, scenario, step, state, agent):
    if scenario.local_window is None:
        return state, None
    start, end = scenario.local_window
    if start <= step < end:
        new_state = environment.set_local_state()
        return new_state, agent.select_action(new_state)
    if step == end:
        new_state = environment.set_common_state()
        return new_state, agent.select_action(new_state)
    return state, None


def run_dynamic_single(job: Mapping[str, Any]) -> Dict[str, Any]:
    scenario = ContinualScenario(**job["scenario"])
    method = str(job["method"])
    seed = int(job["seed"])
    params = dict(job.get("parameters", {}))
    environment = DynamicTwoGoalGrid(seed=seed + 10_000)
    observations = [environment.observation(state) for state in range(len(environment.states))]
    coder = MultiGroupTileCoder(environment.width, environment.height, FeatureConfig())
    coder.preallocate(observations)
    model = _model_spec(method, environment, seed + 30_000, params)
    config = ResearchAgentConfig(
        method=_agent_method(method), epsilon=0.05,
        effective_step_size=0.1 if method == "sarsa_lambda" else 0.2,
        reward_rate_step=0.005, lambda_=0.8,
        planning_steps=0 if model is None else PLANNING_STEPS,
        planning_step_scale=0.25,
        planning_strategy=_planning_strategy(method),
        priority_threshold=1e-3, priority_batch_size=4,
    )
    agent = DifferentialLinearAgent(
        coder, observations, config, seed=seed + 20_000, model=model,
    )
    state = environment.reset(seed + 10_000)
    action = agent.select_action(state)
    oracle_cache = {}
    interval_reward = 0.0
    interval_goals = {"A": 0, "B": 0}
    interval_collisions = 0
    total_reward = 0.0
    curves = []
    events = []
    start_time = time.perf_counter()
    last_context_key = None
    for step in range(scenario.total_steps):
        state, replacement_action = _apply_loca_intervention(
            environment, scenario, step, state, agent
        )
        if replacement_action is not None:
            action = replacement_action
            events.append({"step": step, "event": "state_distribution_intervention"})
        context = scenario.context_at(step)
        if context.key() != last_context_key:
            events.append({"step": step, "event": "context", "context": context.name})
            last_context_key = context.key()
        if isinstance(model, DynamicOracleModel):
            model.set_context(context)
        next_state, reward, terminated, truncated, info = environment.step(action, context)
        if terminated or truncated:
            raise RuntimeError("continual Phase 6--9 environment terminated")
        next_action = agent.select_action(next_state)
        agent.update_real(
            state, action, reward, next_state,
            next_action if config.method == "sarsa_lambda" else None,
            info=info,
        )
        state, action = next_state, next_action
        total_reward += reward
        interval_reward += reward
        if info["goal_id"] is not None:
            interval_goals[info["goal_id"]] += 1
        interval_collisions += int(info["collision"])
        completed_step = step + 1
        if completed_step % DIAGNOSTIC_INTERVAL == 0:
            diagnostic_context = scenario.context_at(step)
            frozen = FrozenDynamicMDP(environment, diagnostic_context)
            oracle = _oracle_solution(environment, diagnostic_context, oracle_cache)
            q_values = np.vstack([agent.values(s) for s in range(len(environment.states))])
            policy = policy_matrix_from_q(q_values, config.epsilon)
            exact_gain, _ = evaluate_policy(frozen, policy_matrix=policy)
            regret = oracle.gain - exact_gain
            valid = scenario.diagnostic_valid(step)
            model_error, coverage = dynamic_model_error(
                model, environment, diagnostic_context
            )
            goal_total = interval_goals["A"] + interval_goals["B"]
            preferred = preferred_goal(diagnostic_context)
            diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else {}
            curves.append({
                "step": completed_step,
                "phase": scenario.phase_name(step),
                "context": diagnostic_context.name,
                "diagnostic_valid": valid,
                "interval_average_reward": interval_reward / DIAGNOSTIC_INTERVAL,
                "stream_average_reward": total_reward / completed_step,
                "exact_policy_gain": exact_gain,
                "oracle_gain": oracle.gain,
                "dynamic_regret": regret if valid else float("nan"),
                "normalized_dynamic_regret": (
                    regret / max(0.1, abs(oracle.gain)) if valid else float("nan")
                ),
                "goal_a_count": interval_goals["A"],
                "goal_b_count": interval_goals["B"],
                "preferred_goal_fraction": (
                    interval_goals[preferred] / goal_total if goal_total else float("nan")
                ),
                "collision_rate": interval_collisions / DIAGNOSTIC_INTERVAL,
                "model_error": model_error,
                "model_coverage": coverage,
                "true_wind_probability": diagnostic_context.wind_probability,
                "true_upper_block": diagnostic_context.upper_block_probability,
                "true_reward_a": diagnostic_context.reward_a,
                "true_reward_b": diagnostic_context.reward_b,
                **diagnostics,
            })
            interval_reward = 0.0
            interval_goals = {"A": 0, "B": 0}
            interval_collisions = 0

    valid_rows = [row for row in curves if np.isfinite(row["dynamic_regret"])]
    post_start = scenario.local_window[1] if scenario.local_window else scenario.change_steps[0]
    post_500 = [row for row in valid_rows if post_start < row["step"] <= post_start + 500]
    post_2000 = [row for row in valid_rows if post_start < row["step"] <= post_start + 2_000]
    tail = curves[int(0.8 * len(curves)):]
    summary = {
        "scenario": scenario.name, "method": method, "seed": seed,
        "steps": scenario.total_steps, "parameters": params,
        "mean_normalized_dynamic_regret": float(np.nanmean([
            row["normalized_dynamic_regret"] for row in valid_rows
        ])),
        "postchange_regret_500": float(np.nanmean([
            row["dynamic_regret"] for row in post_500
        ])) if post_500 else float("nan"),
        "postchange_regret_2000": float(np.nanmean([
            row["dynamic_regret"] for row in post_2000
        ])) if post_2000 else float("nan"),
        "tail_dynamic_regret": float(np.nanmean([
            row["dynamic_regret"] for row in tail
        ])),
        "tail_model_error": float(np.nanmean([
            row["model_error"] for row in tail
        ])) if any(np.isfinite(row["model_error"]) for row in tail) else float("nan"),
        "tail_preferred_goal_fraction": float(np.nanmean([
            row["preferred_goal_fraction"] for row in tail
        ])) if any(np.isfinite(row["preferred_goal_fraction"]) for row in tail) else float("nan"),
        "stream_average_reward": total_reward / scenario.total_steps,
        "real_updates": agent.real_updates,
        "planning_updates": agent.planning_updates,
        "elapsed_seconds": time.perf_counter() - start_time,
        "curves": curves, "events": events,
    }
    return summary


def _job(scenario, method, seed, parameters=None):
    return {
        "scenario": asdict(scenario), "method": method, "seed": int(seed),
        "parameters": dict(parameters or {}),
    }


def execute_jobs(jobs, output: Path, workers: int):
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    pending = []
    for job in jobs:
        suffix = ""
        if job["parameters"]:
            suffix = "__" + "_".join(
                "%s_%s" % (key, str(value).replace(".", "p"))
                for key, value in sorted(job["parameters"].items())
            )
        path = output / (
            "%s__%s%s__seed_%03d.json"
            % (job["scenario"]["name"], job["method"], suffix, job["seed"])
        )
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                completed.append(json.load(handle))
        else:
            pending.append((job, path))
    print("%s: %d complete, %d pending" % (output.name, len(completed), len(pending)), flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_dynamic_single, job): path for job, path in pending}
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                result = future.result()
                _write_json(path, result)
                completed.append(result)
                if index % 10 == 0 or index == len(pending):
                    print("%s: %d/%d" % (output.name, index, len(pending)), flush=True)
    return completed


def validate_scenarios(output: Path):
    environment = DynamicTwoGoalGrid()
    rows = []
    details = {}
    for scenario in phase6_scenarios():
        times = {0, scenario.total_steps - 1, *scenario.change_steps}
        if scenario.schedule == "wind_drift":
            times.update((2_000, 6_000, 10_000, 14_000, 18_000))
        contexts = []
        for step in sorted(times):
            context = scenario.context_at(step)
            if context.key() not in [item.key() for item in contexts]:
                contexts.append(context)
        solutions = []
        for context in contexts:
            frozen = FrozenDynamicMDP(environment, context)
            solution = solve_average_reward(frozen)
            solutions.append((context, solution))
            rows.append({
                "scenario": scenario.name, "context": context.name,
                "context_key": repr(context.key()), "oracle_gain": solution.gain,
            })
        comparisons = []
        for first, second in zip(solutions, solutions[1:]):
            old_context, old_solution = first
            new_context, new_solution = second
            old_gain, _ = evaluate_policy(
                FrozenDynamicMDP(environment, new_context),
                deterministic_policy=old_solution.greedy_policy,
            )
            comparisons.append({
                "old": old_context.name, "new": new_context.name,
                "policy_disagreement": float(np.mean(
                    old_solution.greedy_policy != new_solution.greedy_policy
                )),
                "old_policy_regret_in_new": new_solution.gain - old_gain,
            })
        details[scenario.name] = comparisons
    _write_csv(output / "scenario_oracles.csv", rows)
    _write_json(output / "scenario_validation.json", details)
    return details


def compact_pilot(output: Path, workers: int):
    base = phase6_scenarios()
    pilot_scenarios = (
        replace(base[0], total_steps=10_000, change_steps=(5_000, 5_500), local_window=(5_000, 5_500)),
        replace(base[1], total_steps=10_000, change_steps=(5_000,)),
        replace(base[2], total_steps=10_000, change_steps=(2_000, 4_000, 6_000, 8_000), drift_period=2_000),
    )
    jobs = []
    for scenario in pilot_scenarios:
        for decay in (0.90, 0.97, 0.99):
            for seed in range(3):
                jobs.append(_job(scenario, "ema_dyna", seed, {"ema_decay": decay}))
        for rate in (0.02, 0.05, 0.10):
            for seed in range(3):
                jobs.append(_job(scenario, "cafd_lite", seed, {"factor_rate": rate}))
    results = execute_jobs(jobs, output / "pilot_runs", workers)
    rows = []
    for family, parameter in (("ema_dyna", "ema_decay"), ("cafd_lite", "factor_rate")):
        values = sorted({result["parameters"][parameter] for result in results if result["method"] == family})
        for value in values:
            selected = [
                result["mean_normalized_dynamic_regret"] for result in results
                if result["method"] == family and result["parameters"][parameter] == value
            ]
            rows.append({
                "family": family, "parameter": parameter, "value": value,
                "n": len(selected), "mean_normalized_dynamic_regret": float(np.mean(selected)),
                "se": float(np.std(selected, ddof=1) / np.sqrt(len(selected))),
            })
    _write_csv(output / "pilot_summary.csv", rows)
    selected = {
        "ema_decay": min(
            (row for row in rows if row["family"] == "ema_dyna"),
            key=lambda row: row["mean_normalized_dynamic_regret"],
        )["value"],
        "factor_rate": min(
            (row for row in rows if row["family"] == "cafd_lite"),
            key=lambda row: row["mean_normalized_dynamic_regret"],
        )["value"],
    }
    _write_json(output / "selected_adaptation_parameters.json", selected)
    return selected


def aggregate_results(results, summary: Path):
    summary.mkdir(parents=True, exist_ok=True)
    # Protocol-v1 originally used an obstacle-specific route heuristic for this
    # auxiliary diagnostic.  Recompute it from the unambiguous higher-reward
    # goal so cached runs remain comparable without changing any training data.
    for run in results:
        interval_rewards = np.asarray([
            point["interval_average_reward"] for point in run["curves"]
        ], dtype=np.float64)
        for index, point in enumerate(run["curves"]):
            point["rolling_average_reward_500"] = (
                float(np.mean(interval_rewards[index - 4:index + 1]))
                if index >= 4 else float("nan")
            )
        for point in run["curves"]:
            goal_total = point["goal_a_count"] + point["goal_b_count"]
            higher_reward_count = (
                point["goal_a_count"]
                if point["true_reward_a"] >= point["true_reward_b"]
                else point["goal_b_count"]
            )
            point["preferred_goal_fraction"] = (
                higher_reward_count / goal_total if goal_total else float("nan")
            )
        tail = run["curves"][int(0.8 * len(run["curves"])):]
        finite = [
            point["preferred_goal_fraction"] for point in tail
            if np.isfinite(point["preferred_goal_fraction"])
        ]
        run["tail_preferred_goal_fraction"] = (
            float(np.mean(finite)) if finite else float("nan")
        )
    metrics = (
        "mean_normalized_dynamic_regret", "postchange_regret_500",
        "postchange_regret_2000", "tail_dynamic_regret",
        "tail_model_error", "tail_preferred_goal_fraction",
        "stream_average_reward", "elapsed_seconds",
    )
    aggregate = []
    paired = []
    ablations = []
    for scenario in phase6_scenarios():
        scenario_results = [row for row in results if row["scenario"] == scenario.name]
        baseline = {row["seed"]: row for row in scenario_results if row["method"] == "q_learning"}
        for method in METHOD_ORDER:
            runs = [row for row in scenario_results if row["method"] == method]
            row = {"scenario": scenario.name, "method": method, "n": len(runs)}
            for metric in metrics:
                values = np.asarray([run[metric] for run in runs], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    mean, se, ci = _mean_ci(values)
                else:
                    mean = se = ci = float("nan")
                row[metric + "_mean"] = mean
                row[metric + "_se"] = se
                row[metric + "_ci95"] = ci
            aggregate.append(row)
            if method != "q_learning":
                by_seed = {run["seed"]: run for run in runs}
                seeds = sorted(set(baseline) & set(by_seed))
                for metric in (
                    "mean_normalized_dynamic_regret", "postchange_regret_500",
                    "postchange_regret_2000", "tail_dynamic_regret",
                ):
                    differences = [
                        by_seed[seed][metric] - baseline[seed][metric]
                        for seed in seeds
                        if np.isfinite(by_seed[seed][metric]) and np.isfinite(baseline[seed][metric])
                    ]
                    mean, se, ci = _mean_ci(differences)
                    bootstrap_low, bootstrap_high = _paired_bootstrap_ci(differences)
                    paired.append({
                        "scenario": scenario.name, "method": method,
                        "metric": metric, "n": len(differences),
                        "difference_vs_q_mean": mean,
                        "difference_vs_q_se": se,
                        "difference_vs_q_ci95": ci,
                        "bootstrap_ci95_low": bootstrap_low,
                        "bootstrap_ci95_high": bootstrap_high,
                        "ci_excludes_zero": bootstrap_low > 0.0 or bootstrap_high < 0.0,
                    })
        by_method_seed = {
            method: {
                row["seed"]: row for row in scenario_results if row["method"] == method
            }
            for method in METHOD_ORDER
        }
        for treatment, control, label in (
            ("ema_dyna", "empirical_dyna", "recency_minus_empirical"),
            ("prioritized_ema", "ema_dyna", "priority_minus_uniform_ema"),
            ("cafd_lite", "cafd_uniform", "priority_minus_uniform_factored"),
            ("cafd_surprise", "cafd_lite", "surprise_minus_fixed_cafd"),
        ):
            seeds = sorted(set(by_method_seed[treatment]) & set(by_method_seed[control]))
            for metric in (
                "mean_normalized_dynamic_regret", "postchange_regret_500",
                "postchange_regret_2000", "tail_dynamic_regret", "tail_model_error",
            ):
                differences = [
                    by_method_seed[treatment][seed][metric]
                    - by_method_seed[control][seed][metric]
                    for seed in seeds
                    if np.isfinite(by_method_seed[treatment][seed][metric])
                    and np.isfinite(by_method_seed[control][seed][metric])
                ]
                if differences:
                    mean, se, ci = _mean_ci(differences)
                    bootstrap_low, bootstrap_high = _paired_bootstrap_ci(differences)
                else:
                    mean = se = ci = float("nan")
                    bootstrap_low = bootstrap_high = float("nan")
                ablations.append({
                    "scenario": scenario.name, "comparison": label,
                    "treatment": treatment, "control": control, "metric": metric,
                    "n": len(differences), "paired_difference_mean": mean,
                    "paired_difference_se": se, "paired_difference_ci95": ci,
                    "bootstrap_ci95_low": bootstrap_low,
                    "bootstrap_ci95_high": bootstrap_high,
                    "ci_excludes_zero": bool(
                        bootstrap_low > 0.0 or bootstrap_high < 0.0
                    ),
                })
    _write_csv(summary / "aggregate_summary.csv", aggregate)
    _write_csv(summary / "paired_vs_q_learning.csv", paired)
    _write_csv(summary / "paired_mechanism_ablations.csv", ablations)

    stepwise = []
    for scenario in phase6_scenarios():
        for method in METHOD_ORDER:
            runs = [
                row for row in results
                if row["scenario"] == scenario.name and row["method"] == method
            ]
            for index, point in enumerate(runs[0]["curves"]):
                row = {
                    "scenario": scenario.name, "method": method,
                    "step": point["step"], "phase": point["phase"], "n": len(runs),
                }
                for metric in (
                    "interval_average_reward", "rolling_average_reward_500",
                    "stream_average_reward",
                    "exact_policy_gain", "oracle_gain", "dynamic_regret",
                    "normalized_dynamic_regret", "model_error",
                    "preferred_goal_fraction", "true_wind_probability",
                    "estimated_wind_down", "surprise", "effective_model_rate",
                ):
                    values = np.asarray([
                        run["curves"][index].get(metric, float("nan")) for run in runs
                    ], dtype=np.float64)
                    values = values[np.isfinite(values)]
                    if values.size:
                        mean, se, ci = _mean_ci(values)
                    else:
                        mean = se = ci = float("nan")
                    row[metric + "_mean"] = mean
                    row[metric + "_ci95"] = ci
                stepwise.append(row)
    _write_csv(summary / "stepwise_summary.csv", stepwise)
    make_plots(results, aggregate, summary)
    conclusion = conclusions_markdown(aggregate, paired, ablations)
    (summary / "STATISTICAL_CONCLUSIONS.md").write_text(conclusion, encoding="utf-8")
    _write_json(
        summary / "analysis.json",
        {"aggregate": aggregate, "paired": paired, "ablations": ablations},
    )
    return aggregate, paired


def make_plots(results, aggregate, summary):
    display_methods = (
        "q_learning", "empirical_dyna", "ema_dyna", "prioritized_ema",
        "cafd_uniform", "cafd_lite", "cafd_surprise", "oracle_dyna",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for axis, scenario in zip(axes.flat, phase6_scenarios()):
        for method in display_methods:
            runs = [row for row in results if row["scenario"] == scenario.name and row["method"] == method]
            steps = np.asarray([point["step"] for point in runs[0]["curves"]])
            values = np.asarray([[point["dynamic_regret"] for point in run["curves"]] for run in runs])
            valid_n = np.sum(np.isfinite(values), axis=0)
            mean = np.divide(
                np.nansum(values, axis=0), valid_n,
                out=np.full(values.shape[1], np.nan), where=valid_n > 0,
            )
            centered = values - mean
            squared = np.nansum(centered * centered, axis=0)
            sample_std = np.sqrt(np.divide(
                squared, valid_n - 1,
                out=np.full(values.shape[1], np.nan), where=valid_n > 1,
            ))
            ci = 1.96 * sample_std / np.sqrt(valid_n)
            axis.plot(steps, mean, label=METHOD_LABELS[method], color=COLORS[method], linewidth=1.4)
            axis.fill_between(steps, mean - ci, mean + ci, color=COLORS[method], alpha=0.09)
        for change in scenario.change_steps:
            axis.axvline(change, color="black", linestyle=":", linewidth=0.8)
        axis.set_title(scenario.name)
        axis.set_xlabel("Real environment steps")
        axis.set_ylabel("Exact dynamic regret")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(summary / "dynamic_regret_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, scenario in zip(axes.flat, phase6_scenarios()):
        rows = [row for row in aggregate if row["scenario"] == scenario.name]
        x = np.arange(len(rows))
        axis.bar(
            x, [row["mean_normalized_dynamic_regret_mean"] for row in rows],
            yerr=[row["mean_normalized_dynamic_regret_ci95"] for row in rows],
            color=[COLORS[row["method"]] for row in rows], capsize=2,
        )
        axis.set_xticks(x, [METHOD_LABELS[row["method"]] for row in rows], rotation=35, ha="right", fontsize=7)
        axis.set_title(scenario.name)
        axis.set_ylabel("Mean normalized dynamic regret")
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(summary / "dynamic_regret_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, scenario in zip(axes.flat, phase6_scenarios()):
        for method in (
            "latest_dyna", "empirical_dyna", "ema_dyna",
            "prioritized_ema", "cafd_uniform", "cafd_lite", "cafd_surprise",
        ):
            runs = [row for row in results if row["scenario"] == scenario.name and row["method"] == method]
            steps = np.asarray([point["step"] for point in runs[0]["curves"]])
            values = np.asarray([[point["model_error"] for point in run["curves"]] for run in runs])
            axis.plot(steps, np.nanmean(values, axis=0), label=METHOD_LABELS[method], color=COLORS[method])
        axis.set_title(scenario.name); axis.set_xlabel("Real environment steps")
        axis.set_ylabel("World-model error"); axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(summary / "model_tracking_curves.png", dpi=180)
    plt.close(fig)

    scenario = next(item for item in phase6_scenarios() if item.name == "wind_drift")
    fig, axis = plt.subplots(figsize=(10, 5))
    reference = [row for row in results if row["scenario"] == scenario.name][0]
    steps = np.asarray([point["step"] for point in reference["curves"]])
    axis.plot(steps, [point["true_wind_probability"] for point in reference["curves"]], color="black", linestyle="--", label="True down-wind probability")
    for method in ("cafd_uniform", "cafd_lite", "cafd_surprise"):
        runs = [row for row in results if row["scenario"] == scenario.name and row["method"] == method]
        values = np.asarray([[point.get("estimated_wind_down", np.nan) for point in run["curves"]] for run in runs])
        axis.plot(steps, np.nanmean(values, axis=0), color=COLORS[method], label=METHOD_LABELS[method])
    axis.set_xlabel("Real environment steps"); axis.set_ylabel("Wind probability")
    axis.grid(alpha=0.2); axis.legend(); fig.tight_layout()
    fig.savefig(summary / "wind_factor_tracking.png", dpi=180); plt.close(fig)

    for metric, filename, ylabel in (
        (
            "rolling_average_reward_500", "rolling_average_reward_curves.png",
            "Trailing-500 average reward",
        ),
        (
            "stream_average_reward", "cumulative_average_reward_curves.png",
            "Cumulative stream average reward",
        ),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
        for axis, scenario in zip(axes.flat, phase6_scenarios()):
            for method in METHOD_ORDER:
                runs = [
                    row for row in results
                    if row["scenario"] == scenario.name and row["method"] == method
                ]
                steps = np.asarray([point["step"] for point in runs[0]["curves"]])
                values = np.asarray([
                    [point.get(metric, np.nan) for point in run["curves"]]
                    for run in runs
                ], dtype=np.float64)
                valid_n = np.sum(np.isfinite(values), axis=0)
                mean = np.divide(
                    np.nansum(values, axis=0), valid_n,
                    out=np.full(values.shape[1], np.nan), where=valid_n > 0,
                )
                centered = values - mean
                squared = np.nansum(centered * centered, axis=0)
                sample_std = np.sqrt(np.divide(
                    squared, valid_n - 1,
                    out=np.full(values.shape[1], np.nan), where=valid_n > 1,
                ))
                ci = 1.96 * sample_std / np.sqrt(valid_n)
                axis.plot(
                    steps, mean, label=METHOD_LABELS[method],
                    color=COLORS[method], linewidth=1.35,
                )
                axis.fill_between(
                    steps, mean - ci, mean + ci,
                    color=COLORS[method], alpha=0.07,
                )
            for change in scenario.change_steps:
                axis.axvline(change, color="black", linestyle=":", linewidth=0.8)
            if scenario.local_window is not None:
                axis.axvspan(
                    scenario.local_window[0], scenario.local_window[1],
                    color="black", alpha=0.05,
                )
            axis.set_title(scenario.name)
            axis.set_xlabel("Real environment steps")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(summary / filename, dpi=180)
        plt.close(fig)


def conclusions_markdown(aggregate, paired, ablations):
    lines = [
        "# Phase 6–9 统计结论", "",
        "所有 runs 都是 continuing average-reward streams；竖直 context 线不会终止环境或重置 learner。"
        "每 100 real steps 的 exact policy diagnostic 是只读计算。", "",
        "## 各场景最低 normalized dynamic regret", "",
    ]
    for scenario in phase6_scenarios():
        rows = [row for row in aggregate if row["scenario"] == scenario.name and row["method"] != "oracle_dyna"]
        best = min(rows, key=lambda row: row["mean_normalized_dynamic_regret_mean"])
        lines.append(
            "- **%s**：%s，`%.4f ± %.4f`。" % (
                scenario.name, METHOD_LABELS[best["method"]],
                best["mean_normalized_dynamic_regret_mean"],
                best["mean_normalized_dynamic_regret_ci95"],
            )
        )
    lines.extend(["", "## 相对 Q-learning 的机制结论", ""])
    for scenario in phase6_scenarios():
        for method in ("ema_dyna", "prioritized_ema", "cafd_uniform", "cafd_lite", "cafd_surprise"):
            matches = [
                row for row in paired
                if row["scenario"] == scenario.name and row["method"] == method
                and row["metric"] == "mean_normalized_dynamic_regret"
            ]
            if matches:
                row = matches[0]
                lines.append(
                    "- %s / %s：regret difference `%.4f ± %.4f`%s。" % (
                        scenario.name, METHOD_LABELS[method],
                        row["difference_vs_q_mean"], row["difference_vs_q_ci95"],
                        "（CI 排除 0）" if row["ci_excludes_zero"] else "",
                    )
                )
    lines.extend([
        "", "负 difference 表示比 Q-learning 更低的 dynamic regret。",
        "Oracle Dyna 只作为 perfect-current-model upper bound，不计作 learned method。", "",
    ])
    lines.extend(["## 配对机制消融", ""])
    for scenario in phase6_scenarios():
        for comparison in (
            "recency_minus_empirical", "priority_minus_uniform_ema",
            "priority_minus_uniform_factored", "surprise_minus_fixed_cafd",
        ):
            row = next(
                item for item in ablations
                if item["scenario"] == scenario.name
                and item["comparison"] == comparison
                and item["metric"] == "mean_normalized_dynamic_regret"
            )
            lines.append(
                "- %s / %s：paired difference `%.4f ± %.4f`%s。" % (
                    scenario.name, comparison, row["paired_difference_mean"],
                    row["paired_difference_ci95"],
                    "（CI 排除 0）" if row["ci_excludes_zero"] else "",
                )
            )
    lines.extend([
        "", "这里同样是 treatment − control；负值表示加入该机制后 regret 更低。",
        "`preferred_goal_fraction` 是辅助指标，定义为当前奖励更高目标的到达比例；"
        "它不参与 dynamic regret 或任何显著性主结论。", "",
    ])
    return "\n".join(lines)


def make_manifest(output: Path, seeds: int, selected):
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "continuing": True, "episodic_termination": False,
        "objective": "average_reward/dynamic_regret",
        "policy_diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "feature_config": asdict(FeatureConfig()),
        "planning_steps": PLANNING_STEPS,
        "methods": list(METHOD_ORDER),
        "scenarios": [asdict(scenario) for scenario in phase6_scenarios()],
        "selected_adaptation_parameters": selected,
        "final_seeds": list(range(seeds)),
    }
    path = output / "experiment_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise ValueError("incompatible Phase 6--9 manifest")
    else:
        _write_json(path, manifest)


def build_parser():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Execute continual MBRL Phases 6--9")
    parser.add_argument("--output", type=Path, default=root / "experiment_results" / "phase6_9")
    parser.add_argument("--summary", type=Path, default=root / "phase6_9_summary")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    if args.workers < 1 or args.seeds < 4:
        raise SystemExit("workers>=1 and seeds>=4 are required")
    output = args.output.resolve(); summary = args.summary.resolve()
    output.mkdir(parents=True, exist_ok=True); summary.mkdir(parents=True, exist_ok=True)
    print("Phase 6: validating policy-relevant scenarios", flush=True)
    validate_scenarios(output)
    print("Phases 7-8: compact recency/factor-rate pilot", flush=True)
    selected = compact_pilot(output, args.workers)
    make_manifest(output, args.seeds, selected)
    jobs = [
        _job(scenario, method, seed, selected)
        for scenario in phase6_scenarios()
        for method in METHOD_ORDER
        for seed in range(args.seeds)
    ]
    print("Phase 9: %d locked final runs" % len(jobs), flush=True)
    results = execute_jobs(jobs, output / "final_runs", args.workers)
    _write_csv(
        output / "final_run_summary.csv",
        [{key: value for key, value in row.items() if key not in ("curves", "events")} for row in results],
    )
    aggregate_results(results, summary)
    print("Complete: %s" % summary, flush=True)


if __name__ == "__main__":
    main()
