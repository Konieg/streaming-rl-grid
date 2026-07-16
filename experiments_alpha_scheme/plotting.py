"""Common offline plotting for every Phase result bundle.

Usage: ``python -m experiments.plotting path/to/summary.json``.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .result_schema import aggregate_run_records, load_result_bundle


def _rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.full(window, 1.0 / window), mode="valid")


def _groups(runs: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, str], List[Mapping[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["condition"]), str(run["method"]))].append(run)
    return grouped


def _save(figure: plt.Figure, destination: Path) -> None:
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_learning_curves(bundle: Mapping[str, object], result_dir: Path, output: Path) -> None:
    grouped = _groups(bundle["runs"])
    conditions = sorted({condition for condition, _ in grouped})
    if not conditions:
        return
    figure, axes = plt.subplots(len(conditions), 2, figsize=(12, 3.2 * len(conditions)), squeeze=False)
    for row, condition in enumerate(conditions):
        for col, (trace_key, label) in enumerate(
            (("reward", "reward (rolling mean)"), ("squared_td_error", "online $\\delta^2$ (rolling mean)"))
        ):
            axis = axes[row, col]
            for (candidate, method), runs in grouped.items():
                if candidate != condition:
                    continue
                curves = []
                for run in runs:
                    trace_path = Path(run.get("_trace_path", result_dir / str(run["trace_file"])))
                    if not trace_path.exists():
                        continue
                    with np.load(trace_path) as trace:
                        values = trace[trace_key]
                    values = values[np.isfinite(values)]
                    curves.append(_rolling_mean(values))
                if not curves:
                    continue
                length = min(len(curve) for curve in curves)
                stacked = np.stack([curve[:length] for curve in curves])
                steps = np.arange(length) + (100 if length >= 100 else 1)
                axis.plot(steps, np.mean(stacked, axis=0), label=method)
            axis.set_title("%s — %s" % (condition, label))
            axis.set_xlabel("stream step")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    _save(figure, output / "learning_curves.png")


def plot_final_metrics(bundle: Mapping[str, object], output: Path) -> None:
    aggregates = list(bundle["aggregates"])
    if not aggregates:
        return
    metric_names = (
        "mean_reward",
        "online_squared_td_error",
        "goal_rate_per_1000",
        "collision_rate",
    )
    conditions = sorted({str(row["condition"]) for row in aggregates})
    methods = sorted({str(row["mechanism"]) for row in aggregates})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, metric in zip(axes.flat, metric_names):
        for method_index, method in enumerate(methods):
            means, errors = [], []
            for condition in conditions:
                row = next((item for item in aggregates if item["condition"] == condition and item["mechanism"] == method), None)
                details = None if row is None else row.get("metrics", {}).get(metric)
                means.append(np.nan if details is None else float(details["mean"]))
                errors.append(0.0 if details is None else float(details["standard_error"]))
            positions = np.arange(len(conditions)) + (method_index - (len(methods) - 1) / 2) * 0.18
            axis.errorbar(positions, means, yerr=errors, fmt="o", capsize=3, label=method)
        axis.set_title(metric)
        axis.set_xticks(np.arange(len(conditions)), conditions, rotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    _save(figure, output / "final_metrics.png")


def plot_adaptation_and_retention(bundle: Mapping[str, object], output: Path) -> None:
    grouped = _groups(bundle["runs"])
    labels, auec_values, recovery_values, recurrence_values = [], [], [], []
    for (condition, method), runs in sorted(grouped.items()):
        changes = [change for run in runs for change in run["analysis"]["changes"]]
        if not changes:
            continue
        labels.append("%s\n%s" % (condition, method))
        auec_values.append(float(np.mean([change["postchange_auec"] for change in changes])))
        recovered = [change["recovery_steps"] for change in changes if change["recovery_steps"] is not None]
        recovery_values.append(np.nan if not recovered else float(np.mean(recovered)))
        recurrences = [recurrence for run in runs for recurrence in run["analysis"]["recurrences"]]
        if bundle["task"] == "prediction":
            deltas = [
                recurrence["recurrence_stable_squared_td_error"]
                - recurrence["first_stable_squared_td_error"]
                for recurrence in recurrences
            ]
        else:
            deltas = [
                recurrence["recurrence_stable_reward"] - recurrence["first_stable_reward"]
                for recurrence in recurrences
            ]
        recurrence_values.append(np.nan if not deltas else float(np.mean(deltas)))
    if not labels:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(labels, auec_values)
    axes[0].set_title("mean post-change AUEC")
    axes[1].bar(labels, recovery_values)
    axes[1].set_title("mean recovery steps")
    axes[2].bar(labels, recurrence_values)
    axes[2].set_title(
        "recurrence minus first stable %s"
        % ("online $\\delta^2$" if bundle["task"] == "prediction" else "reward")
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    _save(figure, output / "adaptation_retention.png")


def plot_a_probes(bundle: Mapping[str, object], output: Path) -> None:
    grouped = _groups(bundle["runs"])
    conditions = sorted({condition for condition, _ in grouped})
    figure, axes = plt.subplots(len(conditions), 1, figsize=(10, 2.8 * len(conditions)), squeeze=False)
    drew_any = False
    for axis, condition in zip(axes.flat, conditions):
        for (candidate, method), runs in grouped.items():
            if candidate != condition:
                continue
            probes = [probe for run in runs for probe in run["analysis"]["a_probes"]]
            if not probes:
                continue
            metric = "squared_td_error" if bundle["task"] == "prediction" else "mean_reward"
            by_step: Dict[int, List[float]] = defaultdict(list)
            for probe in probes:
                if metric in probe:
                    by_step[int(probe["step"])].append(float(probe[metric]))
            if by_step:
                steps = sorted(by_step)
                axis.plot(steps, [np.mean(by_step[step]) for step in steps], label=method)
                drew_any = True
        axis.set_title("fixed A probe — %s" % condition)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    if drew_any:
        _save(figure, output / "a_probes.png")
    else:
        plt.close(figure)


def plot_alpha_dynamics(bundle: Mapping[str, object], result_dir: Path, output: Path) -> None:
    grouped = _groups(bundle["runs"])
    conditions = sorted({condition for condition, _ in grouped})
    if not conditions:
        return
    figure, axes = plt.subplots(len(conditions), 1, figsize=(10, 2.8 * len(conditions)), squeeze=False)
    drew_any = False
    for axis, condition in zip(axes.flat, conditions):
        for (candidate, method), runs in grouped.items():
            if candidate != condition:
                continue
            medians, lows, highs = [], [], []
            for run in runs:
                trace_path = Path(run.get("_trace_path", result_dir / str(run["trace_file"])))
                if not trace_path.exists():
                    continue
                with np.load(trace_path) as trace:
                    if "alpha_median" not in trace:
                        continue
                    medians.append(trace["alpha_median"])
                    lows.append(trace["alpha_p10"])
                    highs.append(trace["alpha_p90"])
            if not medians:
                continue
            length = min(len(values) for values in medians)
            median = np.nanmean(np.stack([values[:length] for values in medians]), axis=0)
            low = np.nanmean(np.stack([values[:length] for values in lows]), axis=0)
            high = np.nanmean(np.stack([values[:length] for values in highs]), axis=0)
            steps = np.arange(length)
            line = axis.plot(steps, median, label=method)[0]
            axis.fill_between(steps, low, high, color=line.get_color(), alpha=0.15)
            drew_any = True
        axis.set_title("step-size dynamics — %s" % condition)
        axis.set_xlabel("stream step")
        axis.set_ylabel("alpha")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    if drew_any:
        _save(figure, output / "alpha_dynamics.png")
    else:
        plt.close(figure)


def plot_ablation_results(payload: Mapping[str, object], output: Path) -> Path:
    """Plot a Phase 3 frozen-ablation JSON payload."""
    output.mkdir(parents=True, exist_ok=True)
    task = str(payload["task"])
    metric_names = (
        ("squared_td_error", "online $\\delta^2$"),
        ("mean_reward", "mean reward"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (metric, title) in zip(axes, metric_names):
        grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for record in payload["records"]:
            for evaluation in record["evaluations"]:
                grouped[(evaluation["mode"], evaluation["ablation"])].append(float(evaluation[metric]))
        labels = sorted(grouped)
        axis.bar(
            ["%s\n%s" % label for label in labels],
            [np.mean(grouped[label]) for label in labels],
        )
        axis.set_title("%s frozen ablation — %s" % (task, title))
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    destination = output / "ablation.png"
    _save(figure, destination)
    return destination


def plot_result_bundle(bundle_path: Path, output: Path = None) -> Path:
    return plot_result_bundles((bundle_path,), output)


def plot_result_bundles(bundle_paths: Sequence[Path], output: Path = None) -> Path:
    if not bundle_paths:
        raise ValueError("At least one summary is required")
    loaded = [load_result_bundle(path) for path in bundle_paths]
    tasks = {bundle["task"] for bundle in loaded}
    if len(tasks) != 1:
        raise ValueError("Only bundles with the same task can be plotted together")
    combined_runs = []
    for path, bundle in zip(bundle_paths, loaded):
        for original in bundle["runs"]:
            run = dict(original)
            dimension = run.get("diagnostics", {}).get("feature_dimension")
            representation = "tabular" if dimension is None else "D%d" % int(dimension)
            run["method"] = "%s/%s" % (run["method"], representation)
            run["_trace_path"] = str(path.parent / str(run["trace_file"]))
            combined_runs.append(run)
    bundle = dict(loaded[0])
    bundle["runs"] = combined_runs
    bundle["aggregates"] = aggregate_run_records(combined_runs)
    destination = output or bundle_paths[0].parent / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(bundle, bundle_paths[0].parent, destination)
    plot_final_metrics(bundle, destination)
    plot_adaptation_and_retention(bundle, destination)
    plot_a_probes(bundle, destination)
    plot_alpha_dynamics(bundle, bundle_paths[0].parent, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+", help="One or more schema-v1 summary.json files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print("Figures written to %s" % plot_result_bundles(args.summary, args.output))


if __name__ == "__main__":
    main()
