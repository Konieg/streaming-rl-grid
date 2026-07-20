"""Select all eight algorithms' main parameters with the legacy DualTileCoder."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .dyna_q_plus_sweep import parameter_configurations as dyna_plus_configurations
from .eight_algorithm_comparison import METHOD_LABELS_8, METHOD_ORDER
from .experiment_execution import execute, load_or_create_manifest
from .phase1_sweep import (
    default_output as default_phase1_output,
    parameter_configurations as seven_algorithm_configurations,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output() -> Path:
    return project_root() / "experiment_results" / "tile_coding_sweep"


def default_summary_output() -> Path:
    return project_root() / "tile_coding_sweep_summary"


def parameter_configurations() -> List[Dict[str, Any]]:
    result = [dict(config) for config in seven_algorithm_configurations()]
    result.extend(dict(config) for config in dyna_plus_configurations())
    if len(result) != 90:
        raise AssertionError("Expected 90 tile-coding parameter configurations.")
    return result


def make_manifest(source: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    parameters = parameter_configurations()
    seeds = [int(seed) for seed in source["seeds"]]
    settings = {
        name: dict(source["settings"][name])
        for name in ("transition_shift", "reward_shift", "combined")
    }
    metrics = dict(source["metrics"])
    metrics["post_change_window"] = 500
    agent_common = dict(source["agent_common"])
    agent_common.update({
        "num_tilings": 8,
        "tiles_per_dimension": 8,
        "iht_size": 65_536,
    })
    return {
        "protocol_version": 1,
        "name": "eight_algorithm_legacy_tile_coding_parameter_sweep",
        "source_phase1_manifest": str(source_path.resolve()),
        "steps": int(source["steps"]),
        "seeds": seeds,
        "expected_runs": len(settings) * len(parameters) * len(seeds),
        "feature_representation": "tile_coding",
        "environment": source["environment"],
        "agent_common": agent_common,
        "metrics": metrics,
        "schedule": source["schedule"],
        "settings": settings,
        "parameter_configurations": parameters,
        "seed_manifests": source["seed_manifests"],
        "method_order": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS_8,
        "representation_protocol": {
            "name": "DualTileCoder",
            "position_tilings": 8,
            "relative_goal_tilings": 8,
            "tiles_per_dimension": 8,
            "iht_size": 65_536,
            "nominal_active_count": 17,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot the resumable 1350-run parameter sweep for all eight "
            "algorithms using the legacy DualTileCoder."
        )
    )
    parser.add_argument("--phase1-input", type=Path, default=default_phase1_output())
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
    source_path = args.phase1_input.resolve() / "experiment_manifest.json"
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(output, make_manifest(source, source_path))
    execute(
        "Eight-algorithm tile-coding sweep",
        output,
        args.summary_output.resolve(),
        manifest,
        args.workers,
        args.checkpoint_every,
        args.keep_checkpoints,
        tolerate_failed_configs=True,
    )


if __name__ == "__main__":
    main()
