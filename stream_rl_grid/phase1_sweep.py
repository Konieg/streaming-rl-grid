"""Resumable 945-run D=55 phase-one algorithm sweep."""

import argparse
import csv
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .config import AgentConfig, AppConfig, EnvironmentConfig, TrainingConfig
from .environment import ContinualWindyGridWorld
from .trainer import Trainer


PROTOCOL_VERSION = 1
DEFAULT_STEPS = 60_000
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.01, 0.05, 0.10)
LAMBDAS = (0.50, 0.80, 0.95)
PLANNING_STEPS = (1, 5, 20)

SETTINGS = {
    "transition_shift": {
        "wind_changes": True,
        "goal_moves": True,
        "obstacle_switches": True,
        "reward_changes": False,
    },
    "reward_shift": {
        "wind_changes": False,
        "goal_moves": False,
        "obstacle_switches": False,
        "reward_changes": True,
    },
    "combined": {
        "wind_changes": True,
        "goal_moves": True,
        "obstacle_switches": True,
        "reward_changes": True,
    },
}

SCHEDULE = {
    "period": 6_000,
    "wind_start_step": 5_500,
    "target_move_start_step": 7_000,
    "context_switch_start_step": 8_500,
    "reward_start_step": 10_000,
}

METHOD_LABELS = {
    "sarsa": "SARSA",
    "sarsa_lambda": "SARSA(lambda)",
    "tidbd": "SARSA(lambda)+TIDBD",
    "q_learning": "Q-learning",
    "q_lambda": "Q(lambda)",
    "dyna_q": "Dyna-Q",
    "dyna_q_lambda": "Dyna-Q(lambda)",
}


def _number_tag(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def parameter_configurations() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    def add(method: str, algorithm: str, alpha=None, lambda_=None, planning=None):
        parts = []
        if alpha is not None:
            parts.append("a" + _number_tag(alpha))
        if lambda_ is not None:
            parts.append("l" + _number_tag(lambda_))
        if planning is not None:
            parts.append("p%d" % planning)
        configs.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "algorithm": algorithm,
            "effective_initial_step": 0.10 if alpha is None else float(alpha),
            "lambda": 0.0 if lambda_ is None else float(lambda_),
            "planning_steps": 5 if planning is None else int(planning),
            "config_id": "_".join(parts) if parts else "default",
        })

    for alpha in ALPHAS:
        add("sarsa", "sarsa", alpha=alpha, lambda_=0.0)
    for alpha in ALPHAS:
        for lambda_ in LAMBDAS:
            add("sarsa_lambda", "sarsa", alpha=alpha, lambda_=lambda_)
    for lambda_ in LAMBDAS:
        add("tidbd", "tidbd", lambda_=lambda_)
    for alpha in ALPHAS:
        add("q_learning", "q_learning", alpha=alpha)
    for alpha in ALPHAS:
        for lambda_ in LAMBDAS:
            add("q_lambda", "q_lambda", alpha=alpha, lambda_=lambda_)
    for alpha in ALPHAS:
        for planning in PLANNING_STEPS:
            add("dyna_q", "dyna_q", alpha=alpha, planning=planning)
    for alpha in ALPHAS:
        for lambda_ in LAMBDAS:
            for planning in PLANNING_STEPS:
                add(
                    "dyna_q_lambda", "dyna_q_lambda", alpha=alpha,
                    lambda_=lambda_, planning=planning,
                )
    if len(configs) != 63:
        raise AssertionError("Expected 63 parameter configurations, got %d" % len(configs))
    return configs


def _seed_environment_manifest(seed: int) -> Dict[str, Any]:
    config = EnvironmentConfig(
        width=10,
        height=7,
        obstacle_count=8,
        num_contexts=3,
        wind_changes=True,
        goal_moves=True,
        obstacle_switches=True,
        reward_changes=True,
        seed=int(seed),
        obstacle_coordinates=None,
        context_maps=None,
        start_position=None,
        goal_position=None,
        goal_reached_behavior="random_agent_restart",
    )
    environment = ContinualWindyGridWorld(config)
    return {
        "seed": int(seed),
        "context_maps": [
            [list(point) for point in sorted(layout)]
            for layout in environment.context_maps
        ],
        "goal_path": [list(point) for point in environment.goal_path],
        "start_position": list(environment.start_position),
        "goal_position": list(environment.goal),
    }


def make_manifest(steps: int, seeds: Sequence[int]) -> Dict[str, Any]:
    parameter_configs = parameter_configurations()
    seeds = [int(seed) for seed in seeds]
    expected_runs = len(SETTINGS) * len(parameter_configs) * len(seeds)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "name": "phase1_d55_algorithm_sweep",
        "steps": int(steps),
        "seeds": seeds,
        "expected_runs": expected_runs,
        "feature_representation": "handcrafted_lfa",
        "environment": {
            "width": 10,
            "height": 7,
            "obstacle_count": 8,
            "num_contexts": 3,
            "wind_strength": 0.3,
            "goal_reached_behavior": "random_agent_restart",
        },
        "agent_common": {
            "epsilon": 0.1,
            "reward_rate_step": 0.01,
            "tidbd_theta": 0.01,
            "tidbd_beta_min": -20.0,
            "tidbd_beta_max": 0.0,
        },
        "metrics": {
            "sample_interval": 50,
            "trailing_reward_window": 1_000,
            "post_change_window": 500,
            "recovery_smoothing": 250,
            "recovery_tolerance": 0.10,
            "recovery_horizon": 5_000,
            "recovery_censored_at_next_external_change": True,
        },
        "schedule": dict(SCHEDULE),
        "settings": SETTINGS,
        "parameter_configurations": parameter_configs,
        "seed_manifests": {
            str(seed): _seed_environment_manifest(seed) for seed in seeds
        },
    }


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else [
        "step", "event_type", "event", "prechange_window",
        "prechange_mean_reward", "postchange_window", "postchange_reward_auc",
        "postchange_mean_reward", "recovery_steps",
        "recovered_before_next_change", "next_change_step",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _load_or_create_manifest(output: Path, steps: int, seeds: Sequence[int]) -> Dict[str, Any]:
    path = output / "experiment_manifest.json"
    requested = make_manifest(steps, seeds)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        keys = ("protocol_version", "steps", "seeds", "feature_representation")
        if any(existing.get(key) != requested.get(key) for key in keys):
            raise ValueError(
                "Existing experiment_manifest.json uses a different protocol; "
                "choose a different --output directory."
            )
        return existing
    _atomic_json(path, requested)
    return requested


def build_jobs(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs = []
    for setting in manifest["settings"]:
        for parameters in manifest["parameter_configurations"]:
            applies_to = parameters.get("applies_to_settings")
            if applies_to is not None and setting not in applies_to:
                continue
            for seed in manifest["seeds"]:
                jobs.append({
                    "setting": setting,
                    "parameters": parameters,
                    "seed": int(seed),
                    "relative_dir": "%s/%s/%s/seed_%03d" % (
                        setting, parameters["method"], parameters["config_id"], seed
                    ),
                })
    if len(jobs) != int(manifest["expected_runs"]):
        raise AssertionError("Manifest job count is inconsistent.")
    return jobs


def _app_config(manifest: Dict[str, Any], job: Dict[str, Any]) -> AppConfig:
    seed = int(job["seed"])
    setting = manifest["settings"][job["setting"]]
    seed_data = manifest["seed_manifests"][str(seed)]
    schedule = manifest["schedule"]
    maps = seed_data["context_maps"]
    environment = EnvironmentConfig(
        width=int(manifest["environment"]["width"]),
        height=int(manifest["environment"]["height"]),
        obstacle_count=int(manifest["environment"]["obstacle_count"]),
        num_contexts=int(manifest["environment"]["num_contexts"]),
        wind_changes=bool(setting["wind_changes"]),
        goal_moves=bool(setting["goal_moves"]),
        obstacle_switches=bool(setting["obstacle_switches"]),
        reward_changes=bool(setting["reward_changes"]),
        seed=seed,
        w_strength=float(manifest["environment"]["wind_strength"]),
        wind_period=int(schedule["period"]),
        reward_period=int(schedule["period"]),
        target_move_interval=int(schedule["period"]),
        context_switch_interval=int(schedule["period"]),
        wind_start_step=int(schedule["wind_start_step"]),
        reward_start_step=int(schedule["reward_start_step"]),
        target_move_start_step=int(schedule["target_move_start_step"]),
        context_switch_start_step=int(schedule["context_switch_start_step"]),
        obstacle_coordinates=None,
        context_maps=maps if setting["obstacle_switches"] else [maps[0]],
        goal_path=seed_data["goal_path"],
        start_position=seed_data["start_position"],
        goal_position=seed_data["goal_position"],
        goal_reached_behavior="random_agent_restart",
    )
    parameters = job["parameters"]
    common = manifest["agent_common"]
    agent = AgentConfig(
        algorithm=parameters["algorithm"],
        feature_representation="handcrafted_lfa",
        lambda_=float(parameters["lambda"]),
        epsilon=float(common["epsilon"]),
        theta=float(common["tidbd_theta"]),
        effective_initial_step=float(parameters["effective_initial_step"]),
        reward_rate_step=float(common["reward_rate_step"]),
        beta_min=float(common["tidbd_beta_min"]),
        beta_max=float(common["tidbd_beta_max"]),
        planning_steps=int(parameters["planning_steps"]),
        dyna_plus_kappa=float(parameters.get("dyna_plus_kappa", 0.001)),
    )
    metric = manifest["metrics"]
    training = TrainingConfig(
        metric_window=int(metric["trailing_reward_window"]),
        chart_points=2_000,
        ui_update_steps=int(metric["sample_interval"]),
        auto_checkpoint_steps=int(manifest["steps"]) + 1,
        checkpoint_dir="checkpoints",
        log_dir="",
        record_step_metrics=True,
        post_change_window=int(metric["post_change_window"]),
        recovery_smoothing=int(metric["recovery_smoothing"]),
        recovery_tolerance=float(metric["recovery_tolerance"]),
        recovery_horizon=int(metric["recovery_horizon"]),
    )
    return AppConfig(environment, agent, training)


def _truncate_metrics_csv(path: Path, step: int) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if int(row["step"]) <= int(step)]
    if not fieldnames:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _mean_or_none(values: Iterable[Any]):
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def run_job(
    output_text: str,
    manifest: Dict[str, Any],
    job: Dict[str, Any],
    checkpoint_interval: int,
    keep_checkpoint: bool,
) -> Dict[str, Any]:
    output = Path(output_text).resolve()
    job_dir = output / job["relative_dir"]
    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        return {"status": "skipped", "job": job["relative_dir"]}
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.pkl"
    failure_path = job_dir / "failure.json"
    config = _app_config(manifest, job)
    _atomic_json(job_dir / "config.json", {
        "job": job,
        "config": config.to_dict(),
        "protocol_version": manifest["protocol_version"],
    })
    started = time.time()
    try:
        if progress_path.exists():
            trainer = Trainer.from_checkpoint(progress_path, base_dir=output)
            _truncate_metrics_csv(job_dir / "metrics.csv", trainer.step_count)
        else:
            trainer = Trainer(
                config, base_dir=output, run_id=job["relative_dir"]
            )
        target = int(manifest["steps"])
        while trainer.step_count < target:
            count = min(int(checkpoint_interval), target - trainer.step_count)
            trainer.run_steps(count, with_snapshot=False)
            trainer.save(progress_path)

        metric_summary = trainer.metrics.summary(trainer.step_count)
        diagnostics = trainer.agent.diagnostics()
        event_rows = trainer.metrics.change_metric_rows()
        _atomic_csv(job_dir / "events.csv", event_rows)
        recovered = [row["recovery_steps"] for row in event_rows]
        result = {
            "status": "complete",
            "job": job,
            "completed_steps": trainer.step_count,
            "elapsed_seconds": time.time() - started,
            "metrics": metric_summary,
            "diagnostics": diagnostics,
            "event_metrics": {
                "event_count": len(event_rows),
                "mean_postchange_reward": _mean_or_none(
                    row["postchange_mean_reward"] for row in event_rows
                ),
                "mean_recovery_steps": _mean_or_none(recovered),
                "recovery_fraction": (
                    float(sum(value is not None for value in recovered) / len(recovered))
                    if recovered else None
                ),
            },
        }
        _atomic_json(summary_path, result)
        if failure_path.exists():
            failure_path.unlink()
        if progress_path.exists() and not keep_checkpoint:
            progress_path.unlink()
        return {"status": "complete", "job": job["relative_dir"]}
    except Exception as exc:
        failure = {
            "status": "failed",
            "job": job,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(failure_path, failure)
        return {"status": "failed", "job": job["relative_dir"], "error": repr(exc)}


def default_output() -> Path:
    return Path(__file__).resolve().parents[1] / "experiment_results" / "phase1_d55"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the resumable 945-job D=55 phase-one sweep."
    )
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0 or args.workers <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("steps, workers, and checkpoint-every must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(output, args.steps, args.seeds)
    jobs = build_jobs(manifest)
    pending = [
        job for job in jobs
        if not (output / job["relative_dir"] / "summary.json").exists()
    ]
    completed_before = len(jobs) - len(pending)
    print(
        "Phase-one sweep: %d total, %d already complete, %d pending, %d workers"
        % (len(jobs), completed_before, len(pending), args.workers),
        flush=True,
    )
    if not pending:
        print("All runs are already complete.", flush=True)
        return

    completed = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job, str(output), manifest, job, args.checkpoint_every,
                args.keep_checkpoints,
            ): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "job": job["relative_dir"],
                    "error": "worker process error: %r" % exc,
                }
            if result["status"] == "failed":
                failed += 1
                print("FAILED %s: %s" % (result["job"], result["error"]), flush=True)
            else:
                completed += 1
            finished = completed_before + completed + failed
            print(
                "[%d/%d] complete_now=%d failed=%d latest=%s"
                % (finished, len(jobs), completed, failed, result["job"]),
                flush=True,
            )
    print("Results: %s" % output, flush=True)
    if failed:
        raise SystemExit(
            "%d run(s) failed; inspect failure.json files and execute the same command "
            "again to resume/retry." % failed
        )


if __name__ == "__main__":
    main()
