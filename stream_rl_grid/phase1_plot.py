"""Aggregate and plot completed D=55 phase-one sweep results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .phase1_sweep import METHOD_LABELS, build_jobs, default_output


METHOD_ORDER = tuple(METHOD_LABELS)
EVENT_COLORS = {
    "wind": "#1f77b4",
    "goal": "#2ca02c",
    "obstacles": "#9467bd",
    "reward": "#d62728",
}
METHOD_COLORS = {
    method: color for method, color in zip(
        METHOD_ORDER,
        ("#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2", "#ff9da6"),
    )
}
METHOD_COLORS["dyna_q_plus"] = "#9c755f"
POST_CHANGE_WINDOWS = (250, 500, 1_000)


def default_summary_output() -> Path:
    return Path(__file__).resolve().parents[1] / "phase1_summary"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_se_ci(values: List[float]) -> Tuple[float, float, float, int]:
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, se, 1.96 * se, int(array.size)


def _selection_eligible(
    successful_seeds: int, expected_seeds: int, failed_seeds: int, missing_seeds: int
) -> bool:
    """Only fully successful parameter configurations may win a sweep."""
    return (
        int(successful_seeds) == int(expected_seeds)
        and int(failed_seeds) == 0
        and int(missing_seeds) == 0
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _postchange_mean_from_sampled_csv(job_dir: Path, window: int) -> float:
    """Exactly reconstruct an event-window mean from cumulative sampled reward.

    Phase-one samples, event boundaries, and supported windows are all aligned to
    50 environment steps. The cumulative reward AUC therefore retains the exact
    reward sum at both ends of every requested interval.
    """
    with (job_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    cumulative = {0: 0.0}
    for row in metric_rows:
        cumulative[int(row["step"])] = float(row["reward_auc"])
    final_step = max(cumulative)
    with (job_dir / "events.csv").open("r", newline="", encoding="utf-8") as handle:
        event_rows = list(csv.DictReader(handle))
    means = []
    for event in event_rows:
        start = int(event["step"])
        end = min(final_step, int(event["next_change_step"]), start + int(window))
        if end <= start:
            continue
        if start not in cumulative or end not in cumulative:
            raise ValueError(
                "Window boundaries are absent from sampled cumulative reward: %s [%d, %d]"
                % (job_dir, start, end)
            )
        means.append((cumulative[end] - cumulative[start]) / (end - start))
    return float(np.mean(means)) if means else float("nan")


def _event_steps(manifest: Dict[str, Any], event_type: str) -> List[int]:
    schedule = manifest["schedule"]
    start_key = {
        "wind": "wind_start_step",
        "goal": "target_move_start_step",
        "obstacles": "context_switch_start_step",
        "reward": "reward_start_step",
    }[event_type]
    return list(range(
        int(schedule[start_key]), int(manifest["steps"]), int(schedule["period"])
    ))


def _add_event_lines(axis, manifest: Dict[str, Any], setting: str) -> None:
    flags = manifest["settings"][setting]
    enabled = {
        "wind": flags["wind_changes"],
        "goal": flags["goal_moves"],
        "obstacles": flags["obstacle_switches"],
        "reward": flags["reward_changes"],
    }
    for event_type, active in enabled.items():
        if not active:
            continue
        for index, step in enumerate(_event_steps(manifest, event_type)):
            axis.axvline(
                step,
                color=EVENT_COLORS[event_type],
                linestyle="--",
                linewidth=0.9,
                alpha=0.30,
                label=(event_type + " change") if index == 0 else None,
            )


def aggregate(
    input_dir: Path,
    summary_dir: Path = None,
    allow_incomplete: bool = False,
) -> None:
    summary_dir = default_summary_output() if summary_dir is None else summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(input_dir / "experiment_manifest.json")
    method_labels = dict(METHOD_LABELS)
    method_labels.update(manifest.get("method_labels", {}))
    method_order = tuple(manifest.get("method_order", METHOD_ORDER))
    jobs = build_jobs(manifest)
    completed = []
    missing = []
    failed = []
    status_counts = defaultdict(lambda: {"expected": 0, "failed": 0, "missing": 0})
    for job in jobs:
        job_dir = input_dir / job["relative_dir"]
        key = (
            job["setting"], job["parameters"]["method"],
            job["parameters"]["config_id"],
        )
        status_counts[key]["expected"] += 1
        summary_path = job_dir / "summary.json"
        if not summary_path.exists():
            failure_path = job_dir / "failure.json"
            if failure_path.exists():
                failed.append(job["relative_dir"])
                status_counts[key]["failed"] += 1
            else:
                missing.append(job["relative_dir"])
                status_counts[key]["missing"] += 1
            continue
        completed.append((job, job_dir, _read_json(summary_path)))
    if missing and not allow_incomplete:
        raise RuntimeError(
            "%d/%d runs are still pending without a failure record. Finish the sweep "
            "or pass --allow-incomplete."
            % (len(missing), len(jobs))
        )
    if not completed:
        raise RuntimeError("No completed runs were found.")

    chart_dir = summary_dir
    grouped = defaultdict(list)
    for job, job_dir, summary in completed:
        key = (
            job["setting"], job["parameters"]["method"],
            job["parameters"]["config_id"],
        )
        grouped[key].append((job, job_dir, summary))

    aggregate_rows = []
    for (setting, method, config_id), entries in sorted(grouped.items()):
        rewards = [entry[2]["metrics"]["stream_average_reward"] for entry in entries]
        postchange = [
            _postchange_mean_from_sampled_csv(entry[1], 500)
            for entry in entries
        ]
        recovery = [entry[2]["event_metrics"]["mean_recovery_steps"] for entry in entries]
        reward_mean, reward_se, reward_ci, n = _mean_se_ci(rewards)
        post_mean, post_se, post_ci, _ = _mean_se_ci(postchange)
        recovery_mean, recovery_se, recovery_ci, recovery_n = _mean_se_ci(recovery)
        parameters = entries[0][0]["parameters"]
        counts = status_counts[(setting, method, config_id)]
        eligible = _selection_eligible(
            n, counts["expected"], counts["failed"], counts["missing"]
        )
        aggregate_rows.append({
            "setting": setting,
            "method": method,
            "method_label": method_labels[method],
            "config_id": config_id,
            "effective_initial_step": parameters["effective_initial_step"],
            "lambda": parameters["lambda"],
            "planning_steps": parameters["planning_steps"],
            "dyna_plus_kappa": parameters.get("dyna_plus_kappa", ""),
            "n": n,
            "expected_seed_count": counts["expected"],
            "failed_seed_count": counts["failed"],
            "missing_seed_count": counts["missing"],
            "selection_eligible": eligible,
            "stream_average_reward_mean": reward_mean,
            "stream_average_reward_se": reward_se,
            "stream_average_reward_ci95_halfwidth": reward_ci,
            "postchange_window": 500,
            "postchange_reward_mean": post_mean,
            "postchange_reward_se": post_se,
            "postchange_reward_ci95_halfwidth": post_ci,
            "recovery_steps_mean": recovery_mean,
            "recovery_steps_se": recovery_se,
            "recovery_steps_ci95_halfwidth": recovery_ci,
            "recovery_n": recovery_n,
        })
    _write_csv(summary_dir / "aggregate_summary.csv", aggregate_rows)

    selected = {}
    selected_rows = []
    for setting in manifest["settings"]:
        for method in method_order:
            candidates = [
                row for row in aggregate_rows
                if row["setting"] == setting and row["method"] == method
                and row["selection_eligible"]
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: row["stream_average_reward_mean"])
            selected[(setting, method)] = best
            selected_rows.append(dict(best))
    _write_csv(summary_dir / "selected_configs.csv", selected_rows)

    postchange_window_rows = []
    for setting in manifest["settings"]:
        figure, axis = plt.subplots(figsize=(13, 5.5))
        for method in method_order:
            best = selected.get((setting, method))
            if best is None:
                continue
            entries = grouped[(setting, method, best["config_id"])]
            traces = []
            steps = None
            for _, job_dir, _ in entries:
                with (job_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                current_steps = np.asarray([int(row["step"]) for row in rows], dtype=int)
                values = np.asarray([float(row["average_reward"]) for row in rows], dtype=float)
                if steps is None:
                    steps = current_steps
                minimum = min(len(steps), len(current_steps), len(values))
                steps = steps[:minimum]
                traces = [trace[:minimum] for trace in traces]
                traces.append(values[:minimum])
            matrix = np.asarray(traces, dtype=float)
            mean = matrix.mean(axis=0)
            se = (
                matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
                if matrix.shape[0] > 1 else np.zeros_like(mean)
            )
            color = METHOD_COLORS[method]
            axis.plot(steps, mean, color=color, label=method_labels[method], linewidth=1.8)
            axis.fill_between(steps, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.16)
        _add_event_lines(axis, manifest, setting)
        axis.set_title("%s - selected D=55 configurations" % setting)
        axis.set_xlabel("environment step")
        axis.set_ylabel("trailing-1000 mean reward")
        axis.grid(alpha=0.20)
        axis.legend(ncol=2, fontsize=9)
        figure.tight_layout()
        figure.savefig(chart_dir / ("learning_curves_%s.png" % setting), dpi=180)
        plt.close(figure)

        methods = [method for method in method_order if (setting, method) in selected]
        labels = [method_labels[method] for method in methods]
        post = [selected[(setting, method)]["postchange_reward_mean"] for method in methods]
        post_ci = [selected[(setting, method)]["postchange_reward_ci95_halfwidth"] for method in methods]
        recovery = [selected[(setting, method)]["recovery_steps_mean"] for method in methods]
        recovery_ci = [selected[(setting, method)]["recovery_steps_ci95_halfwidth"] for method in methods]
        x = np.arange(len(methods))
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        colors = [METHOD_COLORS[method] for method in methods]
        axes[0].bar(x, post, yerr=post_ci, color=colors, alpha=0.85, capsize=3)
        axes[0].set_ylabel("post-change mean reward")
        axes[0].grid(axis="y", alpha=0.2)
        axes[1].bar(x, recovery, yerr=recovery_ci, color=colors, alpha=0.85, capsize=3)
        axes[1].set_ylabel("recovery steps")
        axes[1].set_xticks(x, labels, rotation=20, ha="right")
        axes[1].grid(axis="y", alpha=0.2)
        figure.suptitle("%s - adaptation metrics" % setting)
        figure.tight_layout()
        figure.savefig(chart_dir / ("adaptation_metrics_%s.png" % setting), dpi=180)
        plt.close(figure)

        # These values can be reconstructed exactly from old phase-one metrics.csv
        # files, so changing the comparison windows does not require retraining.
        figure, axis = plt.subplots(figsize=(12, 5.5))
        width = 0.24
        for window_index, window in enumerate(POST_CHANGE_WINDOWS):
            means = []
            cis = []
            for method in methods:
                best = selected[(setting, method)]
                entries = grouped[(setting, method, best["config_id"])]
                seed_values = [
                    _postchange_mean_from_sampled_csv(job_dir, window)
                    for _, job_dir, _ in entries
                ]
                mean, se, ci, n = _mean_se_ci(seed_values)
                means.append(mean)
                cis.append(ci)
                postchange_window_rows.append({
                    "setting": setting,
                    "method": method,
                    "method_label": method_labels[method],
                    "config_id": best["config_id"],
                    "window": window,
                    "mean": mean,
                    "se": se,
                    "ci95_halfwidth": ci,
                    "n": n,
                })
            offset = (window_index - (len(POST_CHANGE_WINDOWS) - 1) / 2.0) * width
            axis.bar(
                x + offset, means, width=width, yerr=cis, capsize=3,
                alpha=0.82, label="%d steps" % window,
            )
        axis.set_ylabel("post-change mean reward")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title("%s - post-change window sensitivity" % setting)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            chart_dir / ("postchange_reward_windows_%s.png" % setting), dpi=180
        )
        plt.close(figure)

    _write_csv(summary_dir / "postchange_window_summary.csv", postchange_window_rows)

    print(
        "Aggregated %d completed runs from %s into %s"
        % (len(completed), input_dir, summary_dir)
    )
    if failed:
        print(
            "Recorded %d numerically failed run(s); their parameter configurations "
            "were excluded from selection." % len(failed)
        )
    if missing:
        print("Warning: %d runs were still pending." % len(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and plot phase-one sweep results")
    parser.add_argument("--input", type=Path, default=default_output())
    parser.add_argument("--output", type=Path, default=default_summary_output())
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate(
        args.input.resolve(),
        summary_dir=args.output.resolve(),
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
