"""Generate compact, mechanism-focused figures for the project poster."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiment_results" / "phase6_9" / "final_runs"
SUMMARY = ROOT / "phase6_9_summary"
OUTPUT = ROOT / "poster_assets"

SCENARIOS = (
    "loca_reward", "obstacle_abrupt", "wind_drift", "recurring_composition",
)
TITLES = {
    "loca_reward": "Local reward revaluation",
    "obstacle_abrupt": "Abrupt corridor blockage",
    "wind_drift": "Smooth wind drift",
    "recurring_composition": "Recurring + novel composition",
}
METHODS = ("q_learning", "ema_dyna", "cafd_uniform", "cafd_lite", "oracle_dyna")
LABELS = {
    "q_learning": "Q-learning",
    "ema_dyna": "EMA Dyna",
    "cafd_uniform": "Factored Dyna",
    "cafd_lite": "CAFD-Lite",
    "cafd_surprise": "CAFD-Surprise",
    "oracle_dyna": "Oracle Dyna",
}
COLORS = {
    "q_learning": "#4C78A8",
    "ema_dyna": "#72B7B2",
    "cafd_uniform": "#D5A500",
    "cafd_lite": "#2E7D32",
    "cafd_surprise": "#7B2CBF",
    "oracle_dyna": "#8E6C8A",
}
CHANGES = {
    "loca_reward": (10_000, 10_500),
    "obstacle_abrupt": (10_000,),
    "wind_drift": (4_000, 8_000, 12_000, 16_000),
    "recurring_composition": (4_000, 8_000, 12_000, 16_000),
}


def load_runs():
    results = []
    for path in sorted(RUNS.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.sum(np.isfinite(values), axis=0)
    mean = np.divide(
        np.nansum(values, axis=0), valid,
        out=np.full(values.shape[1], np.nan), where=valid > 0,
    )
    centered = values - mean
    variance = np.divide(
        np.nansum(centered * centered, axis=0), valid - 1,
        out=np.full(values.shape[1], np.nan), where=valid > 1,
    )
    ci = 1.96 * np.sqrt(variance) / np.sqrt(valid)
    return mean, ci


def dynamic_regret_figure(results):
    # Poster panels are intentionally wide and shallow: the right poster
    # column has much more horizontal than vertical space available.
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 5.8))
    for axis, scenario in zip(axes.flat, SCENARIOS):
        for method in METHODS:
            runs = [
                row for row in results
                if row["scenario"] == scenario and row["method"] == method
            ]
            steps = np.asarray([point["step"] for point in runs[0]["curves"]])
            values = np.asarray([
                [point["dynamic_regret"] for point in run["curves"]]
                for run in runs
            ])
            mean, ci = mean_ci(values)
            linestyle = "--" if method == "oracle_dyna" else "-"
            width = 2.2 if method == "cafd_lite" else 1.55
            axis.plot(
                steps, mean, color=COLORS[method], label=LABELS[method],
                linewidth=width, linestyle=linestyle,
            )
            axis.fill_between(
                steps, mean - ci, mean + ci, color=COLORS[method], alpha=0.08,
            )
        for change in CHANGES[scenario]:
            axis.axvline(change, color="#333333", linestyle=":", linewidth=0.9)
        if scenario == "loca_reward":
            axis.axvspan(10_000, 10_500, color="#777777", alpha=0.10)
        axis.set_title(TITLES[scenario], fontsize=12, fontweight="bold")
        axis.set_xlabel("Real environment steps")
        axis.set_ylabel("Exact dynamic regret (lower is better)")
        axis.grid(alpha=0.18)
        axis.set_ylim(bottom=-0.02)
        # Keep the method mapping attached to every panel so the figure stays
        # readable after it is scaled or split across a poster layout.
        axis.legend(
            loc="upper right", ncol=3, fontsize=6.4,
            frameon=True, framealpha=0.92, edgecolor="#D1D5DB",
            borderpad=0.25, handlelength=1.35, columnspacing=0.65,
            labelspacing=0.20,
        )
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.0)
    fig.savefig(OUTPUT / "dynamic_regret_selected.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def paired_bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(60_909)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return float(np.mean(values)), np.percentile(samples, (2.5, 97.5))


def average_reward_gain_figure(results):
    differences = []
    intervals = []
    for scenario in SCENARIOS:
        q = {
            row["seed"]: row["stream_average_reward"] for row in results
            if row["scenario"] == scenario and row["method"] == "q_learning"
        }
        cafd = {
            row["seed"]: row["stream_average_reward"] for row in results
            if row["scenario"] == scenario and row["method"] == "cafd_lite"
        }
        values = [cafd[seed] - q[seed] for seed in sorted(set(q) & set(cafd))]
        mean, bounds = paired_bootstrap(values)
        differences.append(mean)
        intervals.append((mean - bounds[0], bounds[1] - mean))
    y = np.arange(len(SCENARIOS))
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.barh(
        y, differences, xerr=np.asarray(intervals).T, color="#2E7D32",
        alpha=0.92, capsize=4,
    )
    axis.axvline(0.0, color="#333333", linewidth=1)
    axis.set_yticks(y, [TITLES[name] for name in SCENARIOS])
    axis.invert_yaxis()
    axis.set_xlabel("CAFD-Lite minus Q-learning\nstream average reward (paired 95% bootstrap CI)")
    axis.set_title("More online reward in every continual scenario", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    for index, value in enumerate(differences):
        axis.text(
            value * 0.52, index, "+%.3f" % value, va="center", ha="center",
            color="white", fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT / "average_reward_gain.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def grid_figure():
    fig, axis = plt.subplots(figsize=(9.2, 5.5))
    axis.set_aspect("equal")
    axis.set_xlim(-0.35, 11.2)
    axis.set_ylim(7.55, -0.65)
    axis.axis("off")
    for y in range(7):
        for x in range(7):
            face = "#DBEAFE" if x == 0 else "#F8FAFC"
            if x == 3 and y not in (1, 5):
                face = "#334155"
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor="#CBD5E1"))
    for (x, y), color, label in (
        ((6, 1), "#F59E0B", "A"), ((6, 5), "#8B5CF6", "B"),
    ):
        axis.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#334155", linewidth=1.5))
        axis.text(x + 0.5, y + 0.56, "Goal " + label, ha="center", va="center", color="white", fontweight="bold")
    axis.scatter([1.0], [3.5], s=650, color="#2563EB", edgecolor="white", linewidth=2, zorder=5)
    axis.text(1.0, 3.55, "S", ha="center", va="center", color="white", fontsize=16, fontweight="bold", zorder=6)
    axis.add_patch(FancyArrowPatch((1.3, 3.35), (5.95, 1.5), connectionstyle="arc3,rad=-.22", arrowstyle="-|>", mutation_scale=18, color="#F59E0B", linewidth=3))
    axis.add_patch(FancyArrowPatch((1.3, 3.65), (5.95, 5.5), connectionstyle="arc3,rad=.22", arrowstyle="-|>", mutation_scale=18, color="#7C3AED", linewidth=3))
    axis.plot([3, 3], [1.05, 1.95], color="#DC2626", linewidth=5, linestyle="--")
    axis.text(3.18, 1.55, "dynamic edge", color="#DC2626", fontsize=10, va="center")
    axis.text(0.0, 7.35, "uniform restart region after each goal; no termination", color="#1E40AF", fontsize=11)
    axis.text(7.65, 0.2, "Dynamic factors", fontsize=17, fontweight="bold", color="#172033")
    items = (
        ("#F59E0B", "goal-specific rewards"),
        ("#DC2626", "local corridor availability"),
        ("#2563EB", "global categorical wind"),
    )
    for index, (color, label) in enumerate(items):
        y = 1.0 + index * 0.75
        axis.scatter([7.9], [y], s=90, color=color)
        axis.text(8.2, y, label, va="center", fontsize=12)
    axis.text(7.65, 3.8, "Rewards", fontsize=17, fontweight="bold", color="#172033")
    axis.text(7.65, 4.4, "step  −0.05", fontsize=12)
    axis.text(7.65, 4.9, "collision  −0.25", fontsize=12)
    axis.text(7.65, 5.4, "goals  +1 to +6", fontsize=12)
    axis.text(7.65, 6.35, "42 states × 5 actions", fontsize=12, fontstyle="italic", color="#475569")
    fig.tight_layout()
    fig.savefig(OUTPUT / "grid_world.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def mechanism_figure():
    with (SUMMARY / "aggregate_summary.csv").open("r", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    with (SUMMARY / "stepwise_summary.csv").open("r", encoding="utf-8") as handle:
        stepwise = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    methods = ("ema_dyna", "cafd_uniform", "cafd_lite")
    width = 0.23
    x = np.arange(len(SCENARIOS))
    for offset, method in enumerate(methods):
        rows = [
            next(row for row in aggregate if row["scenario"] == scenario and row["method"] == method)
            for scenario in SCENARIOS
        ]
        axes[0].bar(
            x + (offset - 1) * width,
            [float(row["tail_model_error_mean"]) for row in rows],
            width=width, color=COLORS[method], label=LABELS[method],
            yerr=[float(row["tail_model_error_ci95"]) for row in rows], capsize=2,
        )
    axes[0].set_xticks(x, ["LoCA", "Obstacle", "Wind", "Recurring"])
    axes[0].set_ylabel("Tail world-model error (lower is better)")
    axes[0].set_title("Factorization improves model fidelity", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)

    wind_rows = [row for row in stepwise if row["scenario"] == "wind_drift"]
    truth = [row for row in wind_rows if row["method"] == "q_learning"]
    steps = np.asarray([int(row["step"]) for row in truth])
    axes[1].plot(
        steps, [float(row["true_wind_probability_mean"]) for row in truth],
        color="#222222", linestyle="--", linewidth=2.0, label="True wind probability",
    )
    for method in ("cafd_uniform", "cafd_lite"):
        rows = [row for row in wind_rows if row["method"] == method]
        axes[1].plot(
            [int(row["step"]) for row in rows],
            [float(row["estimated_wind_down_mean"]) for row in rows],
            color=COLORS[method], linewidth=1.8, label=LABELS[method],
        )
    axes[1].set_xlabel("Real environment steps")
    axes[1].set_ylabel("Down-wind probability")
    axes[1].set_title("A learned factor tracks continuous drift", fontweight="bold")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "world_model_evidence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def model_tracking_selected_figure(results):
    # Factored Dyna and CAFD-Lite use the same fixed-rate learned model; their
    # planning strategies differ, so their model-error trajectories coincide.
    methods = ("ema_dyna", "cafd_lite")
    model_labels = {
        "ema_dyna": "Unstructured EMA model",
        "cafd_lite": "Factored model (CAFD-Lite)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 5.8))
    for axis, scenario in zip(axes.flat, SCENARIOS):
        for method in methods:
            runs = [
                row for row in results
                if row["scenario"] == scenario and row["method"] == method
            ]
            steps = np.asarray([point["step"] for point in runs[0]["curves"]])
            values = np.asarray([
                [point["model_error"] for point in run["curves"]]
                for run in runs
            ])
            mean, ci = mean_ci(values)
            axis.plot(
                steps, mean, color=COLORS[method], label=model_labels[method],
                linewidth=2.2 if method == "cafd_lite" else 1.7,
            )
            axis.fill_between(
                steps, mean - ci, mean + ci, color=COLORS[method], alpha=0.09,
            )
        for change in CHANGES[scenario]:
            axis.axvline(change, color="#333333", linestyle=":", linewidth=0.9)
        if scenario == "loca_reward":
            axis.axvspan(10_000, 10_500, color="#777777", alpha=0.10)
        axis.set_title(TITLES[scenario], fontsize=12, fontweight="bold")
        axis.set_xlabel("Real environment steps")
        axis.set_ylabel("World-model error (lower is better)")
        axis.grid(alpha=0.18)
        axis.set_ylim(bottom=-0.02)
        axis.legend(
            loc="upper right", ncol=1, fontsize=7.2,
            frameon=True, framealpha=0.92, edgecolor="#D1D5DB",
            borderpad=0.25, handlelength=1.5, labelspacing=0.20,
        )
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.0)
    fig.savefig(OUTPUT / "model_tracking_selected.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def stream_average_reward_table():
    with (SUMMARY / "aggregate_summary.csv").open("r", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    methods = (
        "q_learning", "ema_dyna", "cafd_uniform", "cafd_lite", "cafd_surprise",
    )
    column_labels = ("Method", "Local reward", "Corridor block", "Wind drift", "Recurring + composition")
    fig, axis = plt.subplots(figsize=(10.8, 3.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    x_positions = (0.04, 0.34, 0.53, 0.70, 0.89)
    axis.text(
        0.5, 0.95, "Stream Average Reward ↑",
        ha="center", va="center", fontsize=18, fontweight="bold",
    )
    axis.text(
        0.5, 0.88, "mean ± 95% CI over 20 paired seeds",
        ha="center", va="center", fontsize=10, color="#475569",
    )
    axis.plot([0.02, 0.98], [0.81, 0.81], color="black", linewidth=1.8)
    for x, label in zip(x_positions, column_labels):
        axis.text(
            x, 0.75, label, ha="left" if label == "Method" else "center",
            va="center", fontsize=11, fontweight="bold",
        )
    axis.plot([0.02, 0.98], [0.69, 0.69], color="black", linewidth=1.0)
    best_by_scenario = {
        scenario: max(
            methods,
            key=lambda method: float(next(
                item for item in aggregate
                if item["scenario"] == scenario and item["method"] == method
            )["stream_average_reward_mean"]),
        )
        for scenario in SCENARIOS
    }
    row_y = (0.60, 0.505, 0.41, 0.315, 0.22)
    for y, method in zip(row_y, methods):
        is_ours = method in ("cafd_lite", "cafd_surprise")
        label = LABELS[method] + (" (ours)" if is_ours else "")
        axis.text(
            x_positions[0], y, label, ha="left", va="center", fontsize=11,
            fontweight="bold" if is_ours else "normal",
        )
        for x, scenario in zip(x_positions[1:], SCENARIOS):
            row = next(
                item for item in aggregate
                if item["scenario"] == scenario and item["method"] == method
            )
            value = float(row["stream_average_reward_mean"])
            ci = float(row["stream_average_reward_ci95"])
            is_best = method == best_by_scenario[scenario]
            axis.text(
                x, y, "%.3f ± %.3f" % (value, ci), ha="center", va="center",
                fontsize=11, fontweight="bold" if is_best else "normal",
                color="#1B5E20" if is_best else "black",
            )
    axis.plot([0.02, 0.98], [0.15, 0.15], color="black", linewidth=1.8)
    axis.text(
        0.02, 0.075,
        "Bold: highest mean. CAFD-Lite vs Q-learning paired bootstrap CI excludes zero in all four scenarios.",
        ha="left", va="center", fontsize=9.5, color="#334155",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "stream_average_reward_table.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = load_runs()
    if len(results) != 800:
        raise RuntimeError("expected 800 final runs, found %d" % len(results))
    dynamic_regret_figure(results)
    average_reward_gain_figure(results)
    mechanism_figure()
    model_tracking_selected_figure(results)
    stream_average_reward_table()
    grid_figure()
    print("Poster figures written to", OUTPUT)


if __name__ == "__main__":
    main()
