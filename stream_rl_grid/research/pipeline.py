"""Reproducible Phase 0--5 stationary research pipeline.

The command intentionally keeps the new diagnostic stack separate from the legacy
comparison code.  Every run is a single continuing stream and every reported return
quantity is a reward rate, never an episodic return.
"""

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..features import GridFeatureEncoder
from .agents import DifferentialLinearAgent, ResearchAgentConfig
from .environment import ContinuingGridMDP, StationaryGridSpec, stationary_ladder
from .models import EmpiricalModel, LatestTransitionModel, OracleModel, model_total_variation
from .oracle import evaluate_policy, policy_matrix_from_q, solve_average_reward
from .representations import FeatureConfig, MultiGroupTileCoder


PROTOCOL_VERSION = 4
DEFAULT_DIAGNOSTIC_INTERVAL = 100
METHODS = {
    "q_learning": {"method": "q_learning", "planning_steps": 0, "model": None},
    "sarsa_lambda": {"method": "sarsa_lambda", "planning_steps": 0, "model": None},
    "replay_q_p10": {"method": "replay_q", "planning_steps": 10, "model": None},
    "dyna_latest_p10": {"method": "dyna", "planning_steps": 10, "model": "latest"},
    "dyna_empirical_p5": {"method": "dyna", "planning_steps": 5, "model": "empirical"},
    "dyna_empirical_p10": {"method": "dyna", "planning_steps": 10, "model": "empirical"},
    "dyna_oracle_p10": {"method": "dyna", "planning_steps": 10, "model": "oracle"},
}
METHOD_LABELS = {
    "q_learning": "Q-learning",
    "sarsa_lambda": r"SARSA($\lambda$)",
    "replay_q_p10": "Replay-Q (10)",
    "dyna_latest_p10": "Dyna latest (10)",
    "dyna_empirical_p5": "Dyna empirical (5)",
    "dyna_empirical_p10": "Dyna empirical (10)",
    "dyna_oracle_p10": "Dyna oracle (10)",
}
COLORS = {
    "q_learning": "#4C78A8", "sarsa_lambda": "#F58518",
    "replay_q_p10": "#72B7B2", "dyna_latest_p10": "#E45756",
    "dyna_empirical_p5": "#54A24B", "dyna_empirical_p10": "#2E7D32",
    "dyna_oracle_p10": "#B279A2",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase0_audit(root: Path, output: Path) -> Dict[str, Any]:
    legacy = root / "experiment_results" / "eight_algorithm_comparison"
    summaries = sorted(legacy.glob("**/summary.json"))
    failures = []
    rows = []
    for path in summaries:
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics", {})
        steps = int(summary.get("completed_steps", 0))
        reward_auc = float(metrics.get("reward_auc", float("nan")))
        stream_average = float(metrics.get("stream_average_reward", float("nan")))
        expected_average = reward_auc / steps if steps else float("nan")
        if not np.isclose(stream_average, expected_average):
            failures.append("reward checksum: %s" % path)
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "steps": steps,
            "total_goals": int(metrics.get("total_goals", 0)),
            "goals_per_1000_over_stream": 1000.0 * float(metrics.get("total_goals", 0)) / steps,
            "reward_auc": reward_auc,
            "stream_average_reward": stream_average,
        })
    target_rows = [row for row in rows if "wind_only/dyna_q_plus/" in row["relative_path"]]
    report = {
        "legacy_summary_count": len(summaries),
        "expected_summary_count": 200,
        "all_runs_present": len(summaries) == 200,
        "reward_checksum_failures": failures,
        "goal_metric_interpretation": (
            "summary.metrics.goals_per_1000_steps is a trailing-window metric; "
            "total_goals / completed_steps * 1000 is the full-stream rate"
        ),
        "wind_only_dyna_q_plus_total_goals": [row["total_goals"] for row in target_rows],
        "wind_only_dyna_q_plus_stream_goal_rates": [row["goals_per_1000_over_stream"] for row in target_rows],
        "audit_scope_note": (
            "Legacy metrics.csv is sampled every 50 steps, so it cannot independently reconstruct every goal. "
            "The new Phase 5 format records interval goal counts and exact policy goal rate."
        ),
    }
    _write_json(output / "phase0_audit.json", report)
    _write_csv(output / "phase0_legacy_run_checks.csv", rows)
    return report


def _feature_matrix(coder, observations: Sequence[Sequence[int]], use_values: bool = False) -> np.ndarray:
    rows = []
    for observation in observations:
        for action in range(5):
            row = np.zeros(coder.size, dtype=np.float64)
            if hasattr(coder, "feature_values"):
                indices, values = coder.feature_values(observation, action, readonly=False)
            else:
                indices = coder.active(observation, action, readonly=False)
                values = np.ones(len(indices))
            row[indices] = values
            rows.append(row)
    matrix = np.asarray(rows)
    used = np.flatnonzero(np.any(matrix != 0.0, axis=0))
    return matrix[:, used]


def phase1_2_validation(output: Path) -> Dict[str, Any]:
    results = {"protocol_version": PROTOCOL_VERSION, "tasks": []}
    feature_rows = []
    for spec in stationary_ladder():
        mdp = ContinuingGridMDP(spec)
        oracle = solve_average_reward(mdp)
        results["tasks"].append({
            "name": spec.name, "states": len(mdp.states), "oracle_gain": oracle.gain,
            "oracle_iterations": oracle.iterations, "oracle_residual": oracle.residual,
        })
        if spec.name != "D_corridor9":
            continue
        observations = [mdp.observation(state) for state in range(len(mdp.states))]
        targets = oracle.q_values - oracle.q_values.max(axis=1, keepdims=True)
        targets = targets.reshape(-1)
        coders = {
            "D55": GridFeatureEncoder(spec.width, spec.height),
            "MultiGroup_no_joint": MultiGroupTileCoder(
                spec.width, spec.height, FeatureConfig(joint_tilings=0)
            ),
            "MultiGroup_full": MultiGroupTileCoder(spec.width, spec.height),
            "MultiGroup_no_local": MultiGroupTileCoder(
                spec.width, spec.height, FeatureConfig(include_local_geometry=False)
            ),
        }
        for name, coder in coders.items():
            if hasattr(coder, "preallocate"):
                coder.preallocate(observations)
            matrix = _feature_matrix(coder, observations)
            weights, _, _, _ = np.linalg.lstsq(matrix, targets, rcond=1e-10)
            fitted = (matrix @ weights).reshape(len(mdp.states), 5)
            greedy = np.argmax(fitted, axis=1)
            gain, _ = evaluate_policy(mdp, deterministic_policy=greedy)
            agreement = float(np.mean(greedy == oracle.greedy_policy))
            feature_rows.append({
                "representation": name,
                "used_features": matrix.shape[1],
                "advantage_rmse": float(np.sqrt(np.mean((fitted.reshape(-1) - targets) ** 2))),
                "greedy_action_agreement": agreement,
                "fitted_policy_gain": gain,
                "oracle_gain": oracle.gain,
                "gain_ratio": gain / oracle.gain if oracle.gain != 0 else float("nan"),
            })
    results["feature_realizability"] = feature_rows
    _write_json(output / "phase1_oracle_and_phase2_features.json", results)
    _write_csv(output / "feature_realizability.csv", feature_rows)
    return results


def _make_model(kind: Any, mdp: ContinuingGridMDP, seed: int):
    if kind is None:
        return None
    if kind == "latest":
        return LatestTransitionModel(seed)
    if kind == "empirical":
        return EmpiricalModel(seed)
    if kind == "oracle":
        return OracleModel(mdp, seed)
    raise ValueError("unknown model kind")


def _goal_probability_matrix(mdp: ContinuingGridMDP) -> np.ndarray:
    values = np.zeros((len(mdp.states), mdp.num_actions))
    for state in range(len(mdp.states)):
        for action in range(mdp.num_actions):
            values[state, action] = sum(
                probability for probability, _, _, goal, _ in mdp.transition_distribution(state, action) if goal
            )
    return values


def run_single(job: Mapping[str, Any]) -> Dict[str, Any]:
    spec = StationaryGridSpec(**job["spec"])
    seed = int(job["seed"])
    steps = int(job["steps"])
    interval = int(job["eval_interval"])
    method_name = str(job["method_name"])
    method = METHODS[method_name]
    mdp = ContinuingGridMDP(spec, seed=10_000 + seed)
    oracle = solve_average_reward(mdp)
    observations = [mdp.observation(state) for state in range(len(mdp.states))]
    coder = MultiGroupTileCoder(spec.width, spec.height, FeatureConfig())
    coder.preallocate(observations)
    config = ResearchAgentConfig(
        method=method["method"], epsilon=float(job["epsilon"]),
        effective_step_size=float(job["alpha"]),
        reward_rate_step=float(job["reward_rate_step"]), lambda_=float(job.get("lambda", 0.8)),
        planning_steps=int(method["planning_steps"]), planning_step_scale=float(job["planning_step_scale"]),
    )
    model = _make_model(method["model"], mdp, seed + 30_000)
    agent = DifferentialLinearAgent(
        coder, observations, config, seed=seed + 20_000, model=model,
    )
    state, _ = mdp.reset(seed=seed + 10_000)
    action = agent.select_action(state)
    goal_probability = _goal_probability_matrix(mdp)
    interval_reward = 0.0
    interval_goals = 0
    interval_collisions = 0
    total_reward = 0.0
    total_goals = 0
    total_collisions = 0
    curves = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        next_state, reward, terminated, truncated, info = mdp.step(action)
        if terminated or truncated:
            raise RuntimeError("research environment terminated")
        next_action = agent.select_action(next_state)
        agent.update_real(
            state, action, reward, next_state,
            next_action if config.method == "sarsa_lambda" else None,
        )
        state, action = next_state, next_action
        interval_reward += reward
        total_reward += reward
        interval_goals += int(info["goal_reached"])
        total_goals += int(info["goal_reached"])
        interval_collisions += int(info["collision"])
        total_collisions += int(info["collision"])
        if step % interval == 0:
            q_values = np.vstack([agent.values(s) for s in range(len(mdp.states))])
            policy = policy_matrix_from_q(q_values, config.epsilon)
            exact_gain, distribution = evaluate_policy(mdp, policy_matrix=policy)
            exact_goal_rate = float(np.einsum("s,sa,sa->", distribution, policy, goal_probability))
            curves.append({
                "step": step,
                "interval_average_reward": interval_reward / interval,
                "stream_average_reward": total_reward / step,
                "interval_goals_per_1000": 1000.0 * interval_goals / interval,
                "interval_collision_rate": interval_collisions / interval,
                "exact_policy_gain": exact_gain,
                "exact_policy_goal_rate_per_1000": 1000.0 * exact_goal_rate,
                "reward_rate_estimate": agent.reward_rate,
                "model_tv_error": model_total_variation(model, mdp) if model is not None and method["model"] != "oracle" else (0.0 if method["model"] == "oracle" else float("nan")),
            })
            interval_reward = 0.0
            interval_goals = 0
            interval_collisions = 0
    tail = curves[max(0, int(0.8 * len(curves))):]
    threshold = 0.80 * oracle.gain
    threshold_hits = [row["step"] for row in curves if row["exact_policy_gain"] >= threshold]
    summary = {
        "method_name": method_name, "seed": seed, "task": spec.name,
        "steps": steps, "alpha": config.effective_step_size,
        "epsilon": config.epsilon, "reward_rate_step": config.reward_rate_step,
        "planning_steps": config.planning_steps,
        "planning_step_scale": config.planning_step_scale,
        "oracle_gain": oracle.gain,
        "final_exact_policy_gain": curves[-1]["exact_policy_gain"],
        "tail_exact_policy_gain": float(np.mean([row["exact_policy_gain"] for row in tail])),
        "learning_curve_gain_auc": float(np.mean([row["exact_policy_gain"] for row in curves])),
        "early_exact_policy_gain": float(np.mean([
            row["exact_policy_gain"] for row in curves if row["step"] <= min(5_000, steps)
        ])),
        "steps_to_80pct_oracle": int(threshold_hits[0] if threshold_hits else steps + interval),
        "reached_80pct_oracle": bool(threshold_hits),
        "tail_model_tv_error": float(np.mean([
            row["model_tv_error"] for row in tail if np.isfinite(row["model_tv_error"])
        ])) if any(np.isfinite(row["model_tv_error"]) for row in tail) else float("nan"),
        "tail_interval_average_reward": float(np.mean([row["interval_average_reward"] for row in tail])),
        "stream_average_reward": total_reward / steps,
        "total_goals": total_goals,
        "goals_per_1000_over_stream": 1000.0 * total_goals / steps,
        "collision_rate_over_stream": total_collisions / steps,
        "real_updates": agent.real_updates,
        "planning_updates": agent.planning_updates,
        "iht_used": len(coder.iht.dictionary), "iht_collisions": coder.iht.overfull_count,
        "elapsed_seconds": time.perf_counter() - started,
        "curves": curves,
    }
    return summary


def _execute_jobs(jobs: Sequence[Mapping[str, Any]], folder: Path, workers: int) -> List[Dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    pending = []
    for job in jobs:
        path = folder / ("%s__%s__seed_%03d.json" % (job["spec"]["name"], job["method_name"], job["seed"]))
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        else:
            pending.append((job, path))
    print("%s: %d complete, %d pending" % (folder.name, len(results), len(pending)), flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_single, job): (job, path) for job, path in pending}
            for index, future in enumerate(as_completed(futures), 1):
                job, path = futures[future]
                result = future.result()
                _write_json(path, result)
                results.append(result)
                if index % max(1, min(10, len(pending))) == 0 or index == len(pending):
                    print("%s: %d/%d new runs" % (folder.name, index, len(pending)), flush=True)
    return results


def _job(
    spec, method_name, seed, steps, alpha,
    eval_interval=DEFAULT_DIAGNOSTIC_INTERVAL,
) -> Dict[str, Any]:
    return {
        "spec": asdict(spec), "method_name": method_name, "seed": int(seed),
        "steps": int(steps), "eval_interval": int(eval_interval),
        "alpha": float(alpha), "epsilon": 0.05, "reward_rate_step": 0.005,
        "lambda": 0.8, "planning_step_scale": 0.25,
    }


def phase3_calibration(output: Path, workers: int) -> Dict[str, float]:
    # C is the competence-gate task: it retains obstacles and stochastic wind.
    # D remains a deliberately harder representation/control stress test.
    main = stationary_ladder()[2]
    pilot_methods = ("q_learning", "sarsa_lambda", "dyna_empirical_p10")
    jobs = [
        _job(
            main, method, seed, 8_000, alpha,
            eval_interval=DEFAULT_DIAGNOSTIC_INTERVAL,
        )
        for method in pilot_methods for alpha in (0.05, 0.1, 0.2, 0.4, 0.8) for seed in range(3)
    ]
    # Encode alpha into the filename-facing method key so resumability does not collapse jobs.
    for job in jobs:
        job["method_name_base"] = job["method_name"]
        job["method_name"] = "%s_a%s" % (job["method_name"], str(job["alpha"]).replace(".", "p"))
    original_methods = dict(METHODS)
    try:
        for job in jobs:
            METHODS[job["method_name"]] = original_methods[job["method_name_base"]]
        results = _execute_jobs(jobs, output / "pilot_runs", workers)
    finally:
        for name in tuple(METHODS):
            if "_a0p" in name:
                METHODS.pop(name)
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for result in results:
        base, alpha_text = result["method_name"].rsplit("_a", 1)
        grouped.setdefault((base, float(alpha_text.replace("p", "."))), []).append(result["tail_exact_policy_gain"])
    selected = {}
    rows = []
    for (method, alpha), values in sorted(grouped.items()):
        rows.append({"method": method, "alpha": alpha, "n": len(values), "mean_tail_policy_gain": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})
    for method in pilot_methods:
        candidates = [row for row in rows if row["method"] == method and np.isfinite(row["mean_tail_policy_gain"])]
        selected[method] = max(candidates, key=lambda row: row["mean_tail_policy_gain"])["alpha"]
    _write_csv(output / "pilot_summary.csv", rows)
    _write_json(output / "selected_hyperparameters.json", selected)
    return selected


def phase3_ladder(output: Path, workers: int, selected: Mapping[str, float]) -> List[Dict[str, Any]]:
    methods = ("q_learning", "sarsa_lambda", "dyna_empirical_p5")
    jobs = []
    for spec in stationary_ladder():
        for method in methods:
            alpha_key = "dyna_empirical_p10" if method.startswith("dyna") else method
            for seed in range(3):
                jobs.append(_job(spec, method, seed, 10_000, selected[alpha_key]))
    results = _execute_jobs(jobs, output / "ladder_runs", workers)
    rows = [{k: value for k, value in result.items() if k != "curves"} for result in results]
    _write_csv(output / "stationary_ladder_summary.csv", rows)
    return results


def phase5_final(output: Path, workers: int, selected: Mapping[str, float], seeds: int, steps: int) -> List[Dict[str, Any]]:
    main = stationary_ladder()[2]
    jobs = []
    for method in METHODS:
        if method == "sarsa_lambda":
            alpha = selected["sarsa_lambda"]
        elif method.startswith("dyna"):
            alpha = selected["dyna_empirical_p10"]
        else:
            alpha = selected["q_learning"]
        for seed in range(seeds):
            jobs.append(_job(main, method, seed, steps, alpha))
    results = _execute_jobs(jobs, output / "final_runs", workers)
    _write_csv(
        output / "final_run_summary.csv",
        [{k: value for k, value in result.items() if k != "curves"} for result in results],
    )
    return results


def _mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se, 1.96 * se


def analyse_and_plot(results: Sequence[Mapping[str, Any]], summary: Path) -> Dict[str, Any]:
    summary.mkdir(parents=True, exist_ok=True)
    # Backfill this derived statistic when analysing resumable runs produced by
    # an earlier code revision; raw checkpoint curves remain the source of truth.
    for run in results:
        if "tail_model_tv_error" not in run:
            tail = run["curves"][max(0, int(0.8 * len(run["curves"]))):]
            values = [point["model_tv_error"] for point in tail if np.isfinite(point["model_tv_error"])]
            run["tail_model_tv_error"] = float(np.mean(values)) if values else float("nan")
    methods = [method for method in METHODS if any(row["method_name"] == method for row in results)]
    aggregate_rows = []
    for method in methods:
        runs = [row for row in results if row["method_name"] == method]
        row = {"method": method, "label": METHOD_LABELS[method], "n": len(runs)}
        for metric in (
            "tail_interval_average_reward", "tail_exact_policy_gain",
            "learning_curve_gain_auc", "early_exact_policy_gain",
            "steps_to_80pct_oracle", "goals_per_1000_over_stream",
            "collision_rate_over_stream", "elapsed_seconds",
        ):
            mean, se, ci = _mean_ci([run[metric] for run in runs])
            row[metric + "_mean"] = mean
            row[metric + "_se"] = se
            row[metric + "_ci95"] = ci
        row["oracle_gain"] = runs[0]["oracle_gain"]
        row["policy_gain_ratio"] = row["tail_exact_policy_gain_mean"] / row["oracle_gain"]
        aggregate_rows.append(row)
    _write_csv(summary / "aggregate_summary.csv", aggregate_rows)

    baseline = {row["seed"]: row for row in results if row["method_name"] == "q_learning"}
    paired_rows = []
    for method in methods:
        if method == "q_learning":
            continue
        runs = {row["seed"]: row for row in results if row["method_name"] == method}
        seeds = sorted(set(baseline) & set(runs))
        for metric in (
            "tail_interval_average_reward", "tail_exact_policy_gain",
            "learning_curve_gain_auc", "early_exact_policy_gain",
            "steps_to_80pct_oracle",
        ):
            differences = [runs[seed][metric] - baseline[seed][metric] for seed in seeds]
            mean, se, ci = _mean_ci(differences)
            paired_rows.append({
                "method": method, "metric": metric, "n": len(seeds),
                "paired_difference_mean": mean, "paired_difference_se": se,
                "paired_difference_ci95": ci, "ci_excludes_zero": abs(mean) > ci,
            })
    _write_csv(summary / "paired_vs_q_learning.csv", paired_rows)

    stepwise_rows = []
    curve_metrics = (
        "interval_average_reward", "stream_average_reward",
        "interval_goals_per_1000", "interval_collision_rate",
        "exact_policy_gain", "exact_policy_goal_rate_per_1000",
        "reward_rate_estimate", "model_tv_error",
    )
    for method in methods:
        runs = [row for row in results if row["method_name"] == method]
        for point_index, point in enumerate(runs[0]["curves"]):
            row = {"method": method, "step": int(point["step"]), "n": len(runs)}
            for metric in curve_metrics:
                values = np.asarray(
                    [run["curves"][point_index][metric] for run in runs],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                if finite.size:
                    mean, se, ci = _mean_ci(finite)
                else:
                    mean = se = ci = float("nan")
                row[metric + "_mean"] = mean
                row[metric + "_se"] = se
                row[metric + "_ci95"] = ci
            stepwise_rows.append(row)
    _write_csv(summary / "learning_curve_stepwise_summary.csv", stepwise_rows)

    model_error_rows = []
    for method in ("dyna_latest_p10", "dyna_empirical_p5", "dyna_empirical_p10"):
        values = [row["tail_model_tv_error"] for row in results if row["method_name"] == method]
        mean, se, ci = _mean_ci(values)
        model_error_rows.append({
            "method": method, "n": len(values), "tail_model_tv_mean": mean,
            "tail_model_tv_se": se, "tail_model_tv_ci95": ci,
        })
    _write_csv(summary / "model_error_summary.csv", model_error_rows)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for method in methods:
        runs = [row for row in results if row["method_name"] == method]
        steps = np.asarray([point["step"] for point in runs[0]["curves"]])
        for axis, metric in zip(axes, ("interval_average_reward", "exact_policy_gain")):
            values = np.asarray([[point[metric] for point in run["curves"]] for run in runs])
            mean = values.mean(axis=0)
            ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
            axis.plot(steps, mean, label=METHOD_LABELS[method], color=COLORS[method], linewidth=1.8)
            axis.fill_between(steps, mean - ci, mean + ci, color=COLORS[method], alpha=0.14)
    axes[0].set_ylabel("Interval average reward")
    axes[1].set_ylabel("Exact fixed-policy diagnostic gain")
    axes[1].set_xlabel("Real environment steps")
    axes[1].axhline(results[0]["oracle_gain"], color="black", linestyle="--", linewidth=1, label="Oracle optimum")
    axes[0].grid(alpha=0.25); axes[1].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(summary / "stationary_learning_curves.png", dpi=180)
    plt.close(fig)

    # The clean planning advantage is concentrated in the first few thousand
    # real transitions.  Keep a dedicated high-resolution view so a 30k x-axis
    # does not visually compress the effect.
    fig, axis = plt.subplots(figsize=(9, 5))
    for method in methods:
        runs = [row for row in results if row["method_name"] == method]
        steps = np.asarray([point["step"] for point in runs[0]["curves"]])
        mask = steps <= min(5_000, int(steps[-1]))
        values = np.asarray([
            [point["exact_policy_gain"] for point in run["curves"]]
            for run in runs
        ])[:, mask]
        mean = values.mean(axis=0)
        ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
        axis.plot(
            steps[mask], mean, label=METHOD_LABELS[method],
            color=COLORS[method], linewidth=1.7, marker="o",
            markersize=2.2, markevery=5,
        )
        axis.fill_between(
            steps[mask], mean - ci, mean + ci,
            color=COLORS[method], alpha=0.12,
        )
    axis.axhline(
        results[0]["oracle_gain"], color="black", linestyle="--",
        linewidth=1, label="Oracle optimum",
    )
    axis.set_xlabel("Real environment steps")
    axis.set_ylabel("Exact fixed-policy diagnostic gain")
    axis.set_title("Early learning at 100-step diagnostic resolution")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(summary / "stationary_learning_curves_early.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(aggregate_rows))
    means = [row["tail_exact_policy_gain_mean"] for row in aggregate_rows]
    errors = [row["tail_exact_policy_gain_ci95"] for row in aggregate_rows]
    axis.bar(x, means, yerr=errors, color=[COLORS[row["method"]] for row in aggregate_rows], capsize=3)
    axis.axhline(aggregate_rows[0]["oracle_gain"], color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, [row["label"] for row in aggregate_rows], rotation=25, ha="right")
    axis.set_ylabel("Tail exact policy gain (95% CI)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(summary / "stationary_final_policy_gain.png", dpi=180)
    plt.close(fig)

    model_methods = [method for method in methods if "latest" in method or "empirical" in method]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for method in model_methods:
        runs = [row for row in results if row["method_name"] == method]
        steps = np.asarray([point["step"] for point in runs[0]["curves"]])
        values = np.asarray([[point["model_tv_error"] for point in run["curves"]] for run in runs])
        axis.plot(steps, np.nanmean(values, axis=0), label=METHOD_LABELS[method], color=COLORS[method])
    axis.set_xlabel("Real environment steps"); axis.set_ylabel("Mean model TV error")
    axis.grid(alpha=0.25); axis.legend()
    fig.tight_layout(); fig.savefig(summary / "stationary_model_error.png", dpi=180); plt.close(fig)

    conclusion = build_conclusion(aggregate_rows, paired_rows, model_error_rows)
    (summary / "STATISTICAL_CONCLUSIONS.md").write_text(conclusion, encoding="utf-8")
    report = {"aggregate": aggregate_rows, "paired": paired_rows}
    _write_json(summary / "analysis.json", report)
    return report


def build_conclusion(aggregate_rows, paired_rows, model_error_rows) -> str:
    by_method = {row["method"]: row for row in aggregate_rows}
    best_learned = max(
        (row for row in aggregate_rows if row["method"] != "dyna_oracle_p10"),
        key=lambda row: row["tail_exact_policy_gain_mean"],
    )
    paired_lookup = {(row["method"], row["metric"]): row for row in paired_rows}
    empirical = paired_lookup.get(("dyna_empirical_p10", "tail_exact_policy_gain"))
    empirical_auc = paired_lookup.get(("dyna_empirical_p10", "learning_curve_gain_auc"))
    empirical_time = paired_lookup.get(("dyna_empirical_p10", "steps_to_80pct_oracle"))
    replay = paired_lookup.get(("replay_q_p10", "tail_exact_policy_gain"))
    model_errors = {row["method"]: row for row in model_error_rows}
    lines = [
        "# Phase 0–5 Stationary 实验统计结论", "",
        "## 问题定义核对", "",
        "所有实验都是单一 continuing stream；目标命中后立即随机重启到合法非目标状态，"
        "但 `terminated=False`、`truncated=False`，learner、平均奖励估计与模型均不重置。"
        "主指标是 average reward/gain，不存在 episodic return。", "",
        "## 主要结果", "",
        "- 最佳 learned-model 方法是 **%s**，tail exact gain 为 `%.4f ± %.4f`（95%% CI），为 oracle 的 `%.1f%%`。" % (
            best_learned["label"], best_learned["tail_exact_policy_gain_mean"],
            best_learned["tail_exact_policy_gain_ci95"], 100 * best_learned["policy_gain_ratio"],
        ),
    ]
    if empirical:
        lines.append(
            "- Dyna empirical(10) 相对 Q-learning 的 paired exact-gain 差为 `%.4f ± %.4f`；95%% CI%s跨过 0。" % (
                empirical["paired_difference_mean"], empirical["paired_difference_ci95"],
                "不" if empirical["ci_excludes_zero"] else "",
            )
        )
    if empirical_auc:
        lines.append(
            "- Dyna empirical(10) 相对 Q-learning 的全程 exact-gain AUC paired 差为 `%.4f ± %.4f`。" % (
                empirical_auc["paired_difference_mean"], empirical_auc["paired_difference_ci95"],
            )
        )
    if empirical_time:
        lines.append(
            "- 达到 80%% oracle gain 的步数差（Dyna−Q）为 `%.0f ± %.0f`；负值代表 Dyna 更快。" % (
                empirical_time["paired_difference_mean"], empirical_time["paired_difference_ci95"],
            )
        )
    if replay:
        lines.append(
            "- 等 update-budget 的 Replay-Q 相对 Q-learning 差为 `%.4f ± %.4f`，用于区分 model structure 与单纯增加更新次数。" % (
                replay["paired_difference_mean"], replay["paired_difference_ci95"],
            )
        )
    latest_error = model_errors.get("dyna_latest_p10")
    empirical_error = model_errors.get("dyna_empirical_p10")
    if latest_error and empirical_error:
        lines.append(
            "- latest model 的 tail TV error 为 `%.3f ± %.3f`，empirical model 为 `%.3f ± %.3f`；随机 dynamics 下 latest-transition table 存在持续偏差。" % (
                latest_error["tail_model_tv_mean"], latest_error["tail_model_tv_ci95"],
                empirical_error["tail_model_tv_mean"], empirical_error["tail_model_tv_ci95"],
            )
        )
    lines.extend([
        "- Oracle-model Dyna 是 representation/planning upper bound，不计作可部署算法。",
        "- latest-transition 与 empirical stochastic model 的差异用于检验：随机 dynamics 下，只记最后一次结果是否会产生系统性 model bias。",
        "", "## 可支持的结论边界", "",
        "Phase 0–5 只能回答 stationary competence、表示上限、模型正确性和 clean planning advantage。"
        "它不能单独支持‘MBRL 更适应环境漂移’；该命题必须在 Phase 6 以后通过 post-change regret、"
        "model tracking error 与 recurrence/composition 实验检验。", "",
        "数值来源：`aggregate_summary.csv` 与 `paired_vs_q_learning.csv`。误差为跨独立随机种子的 95% 正态近似置信区间；"
        "方法差异使用相同 seed 的 paired difference。", "",
    ])
    return "\n".join(lines)


def make_manifest(output: Path, seeds: int, steps: int) -> None:
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "continuing": True, "episodic_termination": False,
        "objective": "average_reward",
        "goal_transition": "reward then uniform restart; no learner reset",
        "feature_config": asdict(FeatureConfig()),
        "methods": METHODS, "method_labels": METHOD_LABELS,
        "ladder": [asdict(spec) for spec in stationary_ladder()],
        "final_seeds": list(range(seeds)), "final_steps": steps,
        "evaluation_interval": DEFAULT_DIAGNOSTIC_INTERVAL,
        "rng_streams": {"environment": "seed+10000", "behavior": "seed+20000", "model": "seed+30000"},
    }
    path = output / "experiment_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise ValueError(
                "Existing experiment manifest is incompatible; use a new output directory."
            )
        return
    _write_json(path, manifest)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Execute stationary research Phases 0--5")
    parser.add_argument(
        "--output", type=Path,
        default=root / "experiment_results" / "stationary_phase0_5_highres",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=root / "stationary_phase0_5_summary_highres",
    )
    parser.add_argument(
        "--selected-hyperparameters", type=Path,
        help="Lock an existing selected_hyperparameters.json and skip retuning.",
    )
    parser.add_argument("--workers", type=int, default=min(24, max(1, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=30_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.seeds < 2 or args.steps < 5_000:
        raise SystemExit("workers>=1, seeds>=2 and steps>=5000 are required")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve(); summary = args.summary.resolve()
    output.mkdir(parents=True, exist_ok=True); summary.mkdir(parents=True, exist_ok=True)
    make_manifest(output, args.seeds, args.steps)
    print("Phase 0: legacy audit", flush=True); phase0_audit(root, output)
    print("Phases 1-2: oracle and representation validation", flush=True); phase1_2_validation(output)
    if args.selected_hyperparameters is not None:
        with args.selected_hyperparameters.resolve().open("r", encoding="utf-8") as handle:
            selected = json.load(handle)
        required = {"q_learning", "sarsa_lambda", "dyna_empirical_p10"}
        if set(selected) != required:
            raise ValueError("locked hyperparameter file has incompatible keys")
        _write_json(output / "selected_hyperparameters.json", selected)
        print("Phase 3: using locked hyperparameters", flush=True)
    else:
        print("Phase 3: compact hyperparameter calibration", flush=True)
        selected = phase3_calibration(output, args.workers)
    print("Phase 3: stationary ladder", flush=True); phase3_ladder(output, args.workers, selected)
    print("Phases 4-5: final clean Dyna diagnostic", flush=True)
    results = phase5_final(output, args.workers, selected, args.seeds, args.steps)
    print("Analysis and plots", flush=True); analyse_and_plot(results, summary)
    print("Complete: %s" % summary, flush=True)


if __name__ == "__main__":
    main()
