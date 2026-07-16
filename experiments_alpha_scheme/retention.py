"""Offline retention analysis for existing schema-v1 formal result bundles.

This module never trains a learner. It derives a fixed A->B->A retention summary
from saved traces and frozen A-probe checkpoints.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from experiments.result_schema import load_result_bundle


RETENTION_SCHEMA_VERSION = "1.0"


def _segments(labels: Sequence[str]) -> List[Dict[str, object]]:
    if len(labels) == 0:
        return []
    segments: List[Dict[str, object]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append(
                {"label": str(labels[start]), "start": int(start), "end": int(index)}
            )
            start = index
    return segments


def _mean_se_ci(values: Iterable[Optional[float]]) -> Optional[Dict[str, object]]:
    finite = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=float,
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    mean = float(np.mean(finite))
    standard_error = (
        0.0
        if finite.size == 1
        else float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "n": int(finite.size),
    }


def _recovery_steps(
    values: np.ndarray,
    baseline: float,
    higher_is_better: bool,
    smoothing: int,
    tolerance_fraction: float,
) -> Optional[int]:
    finite = np.asarray(values, dtype=float)
    if finite.size < smoothing or not np.all(np.isfinite(finite)):
        return None
    smooth = np.convolve(finite, np.full(smoothing, 1.0 / smoothing), mode="valid")
    tolerance = tolerance_fraction * max(1.0, abs(baseline))
    recovered = np.flatnonzero(
        smooth >= baseline - tolerance
        if higher_is_better
        else smooth <= baseline + tolerance
    )
    return None if recovered.size == 0 else int(recovered[0] + smoothing)


def _probe_at(
    probes: Sequence[Mapping[str, object]], step: int
) -> Optional[Mapping[str, object]]:
    return next((probe for probe in probes if int(probe["step"]) == step), None)


def analyze_run(
    run: Mapping[str, object],
    trace_path: Path,
    task: str,
    protocol: Mapping[str, object],
) -> Dict[str, object]:
    with np.load(trace_path) as trace:
        labels = np.asarray(trace["mode"]).astype(str)
        signal = np.asarray(
            trace["squared_td_error"] if task == "prediction" else trace["reward"],
            dtype=float,
        )

    condition = str(run["condition"])
    steps = int(run["status"]["completed_steps"])
    interval = int(protocol["switch_interval"])
    expected_change_steps = (
        [] if condition == "stationary" else list(range(interval, steps + 1, interval))
    )
    actual_change_steps = [int(step) for step in run["analysis"]["change_steps"]]
    schedule_compliant = actual_change_steps == expected_change_steps

    result: Dict[str, object] = {
        "run_id": str(run["run_id"]),
        "condition": condition,
        "method": str(run["method"]),
        "seed": int(run["seed"]),
        "schedule_compliant": schedule_compliant,
        "expected_change_steps": expected_change_steps,
        "actual_change_steps": actual_change_steps,
        "initial_a_acquisition_steps": None,
        "recurrence_recovery_steps": None,
        "recurrence_minus_initial_acquisition": None,
        "a_probe": None,
    }
    if condition == "stationary" or not schedule_compliant:
        return result

    segments = _segments(labels)
    if not segments:
        return result
    a_label = str(segments[0]["label"])
    a_segments = [segment for segment in segments if segment["label"] == a_label]
    if len(a_segments) < 2:
        return result

    first, recurrent = a_segments[0], a_segments[1]
    metric_window = int(protocol["metric_window"])
    smoothing = int(protocol["recovery_smoothing"])
    tolerance = float(protocol["recovery_tolerance"])
    first_values = signal[int(first["start"]):int(first["end"])]
    recurrent_values = signal[int(recurrent["start"]):int(recurrent["end"])]
    stable_tail = first_values[max(0, len(first_values) - metric_window):]
    stable_tail = stable_tail[np.isfinite(stable_tail)]
    if stable_tail.size == 0:
        return result
    baseline = float(np.mean(stable_tail))
    higher_is_better = task == "control"
    initial_steps = _recovery_steps(
        first_values, baseline, higher_is_better, smoothing, tolerance
    )
    recurrence_steps = _recovery_steps(
        recurrent_values, baseline, higher_is_better, smoothing, tolerance
    )
    result["initial_a_acquisition_steps"] = initial_steps
    result["recurrence_recovery_steps"] = recurrence_steps
    if initial_steps is not None and recurrence_steps is not None:
        result["recurrence_minus_initial_acquisition"] = (
            recurrence_steps - initial_steps
        )

    probes = list(run["analysis"]["a_probes"])
    pre_a_step = max(
        (
            int(probe["step"])
            for probe in probes
            if int(probe["step"]) < int(first["end"])
        ),
        default=-1,
    )
    end_b_step = max(
        (
            int(probe["step"])
            for probe in probes
            if int(first["end"]) <= int(probe["step"]) < int(recurrent["start"])
        ),
        default=-1,
    )
    if pre_a_step < 0 or end_b_step < 0:
        return result
    before = _probe_at(probes, pre_a_step)
    after = _probe_at(probes, end_b_step)
    if before is None or after is None:
        return result

    if task == "prediction":
        before_value = float(before["squared_td_error"])
        after_value = float(after["squared_td_error"])
        result["a_probe"] = {
            "metric": "squared_td_error",
            "pre_b_step": pre_a_step,
            "end_b_step": end_b_step,
            "pre_b_value": before_value,
            "end_b_value": after_value,
            "retention_loss": after_value - before_value,
        }
    else:
        before_reward = float(before["mean_reward"])
        after_reward = float(after["mean_reward"])
        before_goal = float(before["goal_rate_per_1000"])
        after_goal = float(after["goal_rate_per_1000"])
        result["a_probe"] = {
            "metric": "control",
            "pre_b_step": pre_a_step,
            "end_b_step": end_b_step,
            "pre_b_mean_reward": before_reward,
            "end_b_mean_reward": after_reward,
            "reward_retention_loss": before_reward - after_reward,
            "pre_b_goal_rate_per_1000": before_goal,
            "end_b_goal_rate_per_1000": after_goal,
            "goal_retention_loss": before_goal - after_goal,
        }
    return result


def aggregate_retention(
    records: Sequence[Mapping[str, object]], task: str
) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], record["method"])].append(record)

    aggregates: List[Dict[str, object]] = []
    for (condition, method), group in sorted(grouped.items()):
        valid = [record for record in group if record["schedule_compliant"]]
        row: Dict[str, object] = {
            "condition": condition,
            "method": method,
            "num_runs": len(group),
            "num_schedule_compliant": len(valid),
            "num_excluded": len(group) - len(valid),
            "initial_a_acquisition_steps": _mean_se_ci(
                record["initial_a_acquisition_steps"] for record in valid
            ),
            "recurrence_recovery_steps": _mean_se_ci(
                record["recurrence_recovery_steps"] for record in valid
            ),
            "recurrence_minus_initial_acquisition": _mean_se_ci(
                record["recurrence_minus_initial_acquisition"] for record in valid
            ),
        }
        probes = [
            record["a_probe"]
            for record in valid
            if record["a_probe"] is not None
        ]
        if task == "prediction":
            row["a_probe_retention_loss"] = _mean_se_ci(
                probe["retention_loss"] for probe in probes
            )
        else:
            row["a_probe_reward_retention_loss"] = _mean_se_ci(
                probe["reward_retention_loss"] for probe in probes
            )
            row["a_probe_goal_retention_loss"] = _mean_se_ci(
                probe["goal_retention_loss"] for probe in probes
            )
        aggregates.append(row)
    return aggregates


def analyze_bundle(summary_path: Path) -> Dict[str, object]:
    bundle = load_result_bundle(summary_path)
    records = [
        analyze_run(
            run,
            summary_path.parent / str(run["trace_file"]),
            str(bundle["task"]),
            bundle["protocol"],
        )
        for run in bundle["runs"]
    ]
    return {
        "retention_schema_version": RETENTION_SCHEMA_VERSION,
        "source_summary": str(summary_path),
        "phase": bundle["phase"],
        "subexperiment": bundle["subexperiment"],
        "task": bundle["task"],
        "definitions": {
            "schedule": (
                "One fixed switch interval; runs with actual change_steps different "
                "from the expected schedule are excluded."
            ),
            "initial_a_acquisition_steps": (
                "First rolling window in initial A reaching its own final metric-window "
                "baseline within the frozen recovery tolerance."
            ),
            "recurrence_recovery_steps": (
                "First rolling window after the first return to A reaching the same "
                "initial-A baseline and tolerance."
            ),
            "recurrence_minus_initial_acquisition": (
                "Negative means recurrent A reached the criterion faster than initial A."
            ),
            "a_probe_retention_loss": (
                "Frozen-A probe degradation from the last checkpoint before leaving A "
                "to the last checkpoint before A first returns; positive means worse."
            ),
        },
        "runs": records,
        "aggregates": aggregate_retention(records, str(bundle["task"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+")
    args = parser.parse_args()
    for summary_path in args.summary:
        payload = analyze_bundle(summary_path)
        output = summary_path.parent / "retention.json"
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("Retention analysis written to %s" % output)


if __name__ == "__main__":
    main()

