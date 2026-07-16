"""Versioned result schema shared by all continual-learning phases.

The schema separates raw traces (``.npz``) from small, portable ``summary.json``
bundles.  Phase-specific fields are allowed only under ``analysis`` or
``diagnostics``; the core fields remain comparable across phases.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import json
import numpy as np


SCHEMA_VERSION = "1.0"

METRIC_DEFINITIONS = {
    "mean_reward": "Mean environment reward per transition.",
    "online_squared_td_error": (
        "Mean squared TD error (delta^2). This is an online learning-signal proxy, "
        "not a reference MSVE unless a phase explicitly supplies reference_msve."
    ),
    "reference_msve": "Mean squared value error against a separately specified reference value.",
    "goal_rate_per_1000": "Goals reached per 1,000 environment transitions.",
    "collision_rate": "Fraction of environment transitions ending in collision.",
    "mean_policy_entropy": "Mean entropy of the behavior policy; prediction uses its fixed policy.",
}


def make_run_record(
    summary: Mapping[str, object], trace_file: str, model_file: Optional[str] = None
) -> Dict[str, object]:
    """Convert a phase runner's internal summary into the common portable record."""
    if "diagnostics" in summary:
        diagnostics = dict(summary["diagnostics"])
    else:
        diagnostics = {
            "unique_state_actions_visited": int(summary["unique_state_actions_visited"]),
            "average_reward_estimate": float(summary["average_reward_estimate"]),
            "num_table_entries": int(summary["num_table_entries"]),
            "q_l2_norm": float(summary["q_l2_norm"]),
            "q_max_abs": float(summary["q_max_abs"]),
            "alpha_p10": float(summary["alpha_p10"]),
            "alpha_median": float(summary["alpha_median"]),
            "alpha_p90": float(summary["alpha_p90"]),
            "alpha_min": float(summary["alpha_min"]),
            "alpha_max": float(summary["alpha_max"]),
        }
    record = {
        "run_id": str(summary["run_id"]),
        "condition": str(summary["condition"]),
        "method": str(summary["mechanism"]),
        "seed": int(summary["seed"]),
        "trace_file": trace_file,
        "status": {
            "requested_steps": int(summary["requested_steps"]),
            "completed_steps": int(summary["completed_steps"]),
            "numerically_stable": bool(summary["numerically_stable"]),
            "failure": summary["failure"],
        },
        "metrics": {
            "mean_reward": float(summary["mean_reward"]),
            "online_squared_td_error": float(summary["mean_squared_td_error"]),
            "goal_rate_per_1000": float(summary["goal_rate_per_1000"]),
            "collision_rate": float(summary["collision_rate"]),
            "mean_policy_entropy": float(summary["mean_policy_entropy"]),
        },
        "analysis": {
            "change_steps": list(summary["change_steps"]),
            "changes": list(summary["changes"]),
            "recurrences": list(summary["recurrences"]),
            "a_probes": list(summary["a_probes"]),
        },
        "diagnostics": diagnostics,
    }
    if model_file is not None:
        record["model_file"] = model_file
    return record


def aggregate_run_records(runs: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compute common scalar mean, standard error, and 95% CI by condition/method."""
    grouped: Dict[tuple, List[Mapping[str, object]]] = {}
    for run in runs:
        key = (run["condition"], run["method"])
        grouped.setdefault(key, []).append(run)
    aggregates = []
    for (condition, method), records in sorted(grouped.items()):
        metric_names = sorted({name for record in records for name in record["metrics"]})
        metrics = {}
        for name in metric_names:
            values = np.asarray(
                [float(record["metrics"][name]) for record in records if name in record["metrics"]],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            error = 0.0 if values.size == 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))
            mean = float(np.mean(values))
            metrics[name] = {
                "mean": mean,
                "standard_error": error,
                "ci95": [mean - 1.96 * error, mean + 1.96 * error],
            }
        aggregates.append(
            {
                "condition": condition,
                "mechanism": method,
                "num_runs": len(records),
                "num_stable_runs": sum(
                    bool(record["status"]["numerically_stable"]) for record in records
                ),
                "metrics": metrics,
            }
        )
    return aggregates


def make_result_bundle(
    phase: str,
    subexperiment: str,
    task: str,
    protocol: Mapping[str, object],
    runs: Iterable[Mapping[str, object]],
    aggregates: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "subexperiment": subexperiment,
        "task": task,
        "metric_definitions": METRIC_DEFINITIONS,
        "protocol": dict(protocol),
        "runs": [dict(run) for run in runs],
        "aggregates": [dict(aggregate) for aggregate in aggregates],
    }
    validate_result_bundle(bundle)
    return bundle


def validate_result_bundle(bundle: Mapping[str, object]) -> None:
    required = ("schema_version", "phase", "subexperiment", "task", "protocol", "runs", "aggregates")
    missing = [name for name in required if name not in bundle]
    if missing:
        raise ValueError("Result bundle missing: %s" % ", ".join(missing))
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema: %s" % bundle["schema_version"])
    if not isinstance(bundle["runs"], list):
        raise ValueError("runs must be a list")
    for run in bundle["runs"]:
        run_required = ("run_id", "condition", "method", "seed", "trace_file", "status", "metrics")
        absent = [name for name in run_required if name not in run]
        if absent:
            raise ValueError("Run record missing: %s" % ", ".join(absent))


def write_result_bundle(path: Path, bundle: Mapping[str, object]) -> None:
    validate_result_bundle(bundle)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2)


def load_result_bundle(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    validate_result_bundle(bundle)
    return bundle
