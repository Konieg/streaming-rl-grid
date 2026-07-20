"""Run eight selected algorithms under four isolated shifts and all combined."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .experiment_execution import execute, load_or_create_manifest
from .phase1_sweep import METHOD_LABELS, default_output as default_phase1_output


METHOD_ORDER = (
    "sarsa", "sarsa_lambda", "tidbd", "q_learning", "q_lambda",
    "dyna_q", "dyna_q_lambda", "dyna_q_plus",
)
METHOD_LABELS_8 = dict(METHOD_LABELS, dyna_q_plus="Dyna-Q+")
SETTINGS = {
    "wind_only": {
        "wind_changes": True, "goal_moves": False,
        "obstacle_switches": False, "reward_changes": False,
    },
    "goal_only": {
        "wind_changes": False, "goal_moves": True,
        "obstacle_switches": False, "reward_changes": False,
    },
    "obstacles_only": {
        "wind_changes": False, "goal_moves": False,
        "obstacle_switches": True, "reward_changes": False,
    },
    "reward_only": {
        "wind_changes": False, "goal_moves": False,
        "obstacle_switches": False, "reward_changes": True,
    },
    "combined": {
        "wind_changes": True, "goal_moves": True,
        "obstacle_switches": True, "reward_changes": True,
    },
}
WINNER_SOURCE_SETTING = {
    "wind_only": "transition_shift",
    "goal_only": "transition_shift",
    "obstacles_only": "transition_shift",
    "reward_only": "reward_shift",
    "combined": "combined",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_phase1_selected() -> Path:
    return project_root() / "phase1_summary" / "selected_configs.csv"


def default_dyna_plus_input() -> Path:
    return project_root() / "experiment_results" / "dyna_q_plus_sweep"


def default_dyna_plus_selected() -> Path:
    return project_root() / "dyna_q_plus_sweep_summary" / "selected_configs.csv"


def default_output() -> Path:
    return project_root() / "experiment_results" / "eight_algorithm_comparison"


def default_summary_output() -> Path:
    return project_root() / "eight_algorithm_summary"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_ids(path: Path) -> Dict[Tuple[str, str], str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (row["setting"], row["method"])
        if key in result:
            raise ValueError("Duplicate selected configuration for %r in %s" % (key, path))
        result[key] = row["config_id"]
    return result


def _configuration_index(manifest: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (config["method"], config["config_id"]): config
        for config in manifest["parameter_configurations"]
    }


def selected_parameter_configurations(
    phase1_manifest: Dict[str, Any],
    phase1_selected: Path,
    dyna_plus_manifest: Dict[str, Any],
    dyna_plus_selected: Path,
) -> List[Dict[str, Any]]:
    old_ids = _selected_ids(phase1_selected)
    plus_ids = _selected_ids(dyna_plus_selected)
    old_configs = _configuration_index(phase1_manifest)
    plus_configs = _configuration_index(dyna_plus_manifest)
    result = []
    for destination_setting, source_setting in WINNER_SOURCE_SETTING.items():
        for method in METHOD_ORDER:
            ids = plus_ids if method == "dyna_q_plus" else old_ids
            configs = plus_configs if method == "dyna_q_plus" else old_configs
            selected_key = (source_setting, method)
            if selected_key not in ids:
                raise ValueError(
                    "No selected winner for setting=%s method=%s in %s"
                    % (source_setting, method,
                       dyna_plus_selected if method == "dyna_q_plus" else phase1_selected)
                )
            config_key = (method, ids[selected_key])
            if config_key not in configs:
                raise ValueError("Selected configuration is absent from its manifest: %r" % (config_key,))
            config = dict(configs[config_key])
            config["applies_to_settings"] = [destination_setting]
            config["selected_from_setting"] = source_setting
            result.append(config)
    return result


def make_manifest(
    phase1_manifest: Dict[str, Any],
    phase1_selected: Path,
    dyna_plus_manifest: Dict[str, Any],
    dyna_plus_selected: Path,
) -> Dict[str, Any]:
    if phase1_manifest["steps"] != dyna_plus_manifest["steps"]:
        raise ValueError("Phase-one and Dyna-Q+ sweep step counts differ.")
    if phase1_manifest["seeds"] != dyna_plus_manifest["seeds"]:
        raise ValueError("Phase-one and Dyna-Q+ sweep seeds differ.")
    parameters = selected_parameter_configurations(
        phase1_manifest, phase1_selected, dyna_plus_manifest, dyna_plus_selected
    )
    seeds = [int(seed) for seed in phase1_manifest["seeds"]]
    metrics = dict(phase1_manifest["metrics"])
    metrics["post_change_window"] = 500
    return {
        "protocol_version": 1,
        "name": "eight_algorithm_five_setting_comparison",
        "steps": int(phase1_manifest["steps"]),
        "seeds": seeds,
        "expected_runs": len(SETTINGS) * len(METHOD_ORDER) * len(seeds),
        "feature_representation": phase1_manifest["feature_representation"],
        "environment": phase1_manifest["environment"],
        "agent_common": phase1_manifest["agent_common"],
        "metrics": metrics,
        "schedule": phase1_manifest["schedule"],
        "settings": SETTINGS,
        "parameter_configurations": parameters,
        "seed_manifests": phase1_manifest["seed_manifests"],
        "method_order": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS_8,
        "winner_source_setting": WINNER_SOURCE_SETTING,
        "source_selected_configs": {
            "phase1": str(phase1_selected.resolve()),
            "dyna_q_plus": str(dyna_plus_selected.resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and plot the 200-run, eight-algorithm, five-setting comparison."
    )
    parser.add_argument("--phase1-input", type=Path, default=default_phase1_output())
    parser.add_argument("--phase1-selected", type=Path, default=default_phase1_selected())
    parser.add_argument("--dyna-plus-input", type=Path, default=default_dyna_plus_input())
    parser.add_argument("--dyna-plus-selected", type=Path, default=default_dyna_plus_selected())
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--summary-output", type=Path, default=default_summary_output())
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("workers and checkpoint-every must be positive")
    phase1_manifest = _read_json(args.phase1_input.resolve() / "experiment_manifest.json")
    dyna_plus_manifest = _read_json(args.dyna_plus_input.resolve() / "experiment_manifest.json")
    requested = make_manifest(
        phase1_manifest, args.phase1_selected.resolve(), dyna_plus_manifest,
        args.dyna_plus_selected.resolve(),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(output, requested)
    execute(
        "Eight-algorithm comparison", output, args.summary_output.resolve(), manifest,
        args.workers, args.checkpoint_every, args.keep_checkpoints,
    )


if __name__ == "__main__":
    main()
