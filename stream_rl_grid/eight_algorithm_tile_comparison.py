"""Repeat the final eight-algorithm comparison with the legacy DualTileCoder."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from .eight_algorithm_comparison import (
    default_dyna_plus_input,
    default_dyna_plus_selected,
    default_phase1_output,
    default_phase1_selected,
    make_manifest as make_selected_comparison_manifest,
)
from .experiment_execution import execute, load_or_create_manifest
from .eight_algorithm_comparison import _read_json


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output() -> Path:
    return project_root() / "experiment_results" / "eight_algorithm_tile_comparison"


def default_summary_output() -> Path:
    return project_root() / "eight_algorithm_tile_summary"


def default_tuned_output() -> Path:
    return project_root() / "experiment_results" / "eight_algorithm_tile_tuned_comparison"


def default_tuned_summary_output() -> Path:
    return project_root() / "eight_algorithm_tile_tuned_summary"


def default_tile_sweep_input() -> Path:
    return project_root() / "experiment_results" / "tile_coding_sweep"


def default_tile_sweep_selected() -> Path:
    return project_root() / "tile_coding_sweep_summary" / "selected_configs.csv"


def make_manifest(
    phase1_manifest: Dict[str, Any],
    phase1_selected: Path,
    dyna_plus_manifest: Dict[str, Any],
    dyna_plus_selected: Path,
    parameter_policy: str = "reuse_selected_D55_winners",
) -> Dict[str, Any]:
    manifest = make_selected_comparison_manifest(
        phase1_manifest, phase1_selected, dyna_plus_manifest, dyna_plus_selected
    )
    manifest["name"] = "eight_algorithm_five_setting_legacy_tile_coding"
    manifest["feature_representation"] = "tile_coding"
    manifest["agent_common"] = dict(manifest["agent_common"])
    manifest["agent_common"].update({
        "num_tilings": 8,
        "tiles_per_dimension": 8,
        "iht_size": 65_536,
    })
    manifest["representation_protocol"] = {
        "name": "DualTileCoder",
        "position_tilings": 8,
        "relative_goal_tilings": 8,
        "tiles_per_dimension": 8,
        "iht_size": 65_536,
        "nominal_active_count": 17,
        "parameter_policy": parameter_policy,
    }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot the 200-run eight-algorithm comparison using the "
            "legacy DualTileCoder with transferred or tile-tuned winners."
        )
    )
    parser.add_argument("--phase1-input", type=Path, default=default_phase1_output())
    parser.add_argument("--phase1-selected", type=Path, default=default_phase1_selected())
    parser.add_argument("--dyna-plus-input", type=Path, default=default_dyna_plus_input())
    parser.add_argument("--dyna-plus-selected", type=Path, default=default_dyna_plus_selected())
    parser.add_argument(
        "--use-tuned-parameters", action="store_true",
        help="Read all eight winners from the completed tile-coding sweep",
    )
    parser.add_argument("--tile-sweep-input", type=Path, default=default_tile_sweep_input())
    parser.add_argument("--tile-sweep-selected", type=Path, default=default_tile_sweep_selected())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
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
    if args.use_tuned_parameters:
        tile_manifest = _read_json(
            args.tile_sweep_input.resolve() / "experiment_manifest.json"
        )
        tile_selected = args.tile_sweep_selected.resolve()
        requested = make_manifest(
            tile_manifest, tile_selected, tile_manifest, tile_selected,
            parameter_policy="tile_coding_sweep_winners",
        )
        output = (args.output or default_tuned_output()).resolve()
        summary_output = (
            args.summary_output or default_tuned_summary_output()
        ).resolve()
    else:
        phase1_manifest = _read_json(
            args.phase1_input.resolve() / "experiment_manifest.json"
        )
        dyna_plus_manifest = _read_json(
            args.dyna_plus_input.resolve() / "experiment_manifest.json"
        )
        requested = make_manifest(
            phase1_manifest, args.phase1_selected.resolve(), dyna_plus_manifest,
            args.dyna_plus_selected.resolve(),
        )
        output = (args.output or default_output()).resolve()
        summary_output = (args.summary_output or default_summary_output()).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(output, requested)
    execute(
        "Eight-algorithm tile-coding comparison",
        output,
        summary_output,
        manifest,
        args.workers,
        args.checkpoint_every,
        args.keep_checkpoints,
    )


if __name__ == "__main__":
    main()
