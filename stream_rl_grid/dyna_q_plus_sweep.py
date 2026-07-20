"""Select Dyna-Q+ hyperparameters under the original three settings."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .experiment_execution import execute, load_or_create_manifest
from .phase1_sweep import default_output as default_phase1_output


ALPHAS = (0.01, 0.05, 0.10)
PLANNING_STEPS = (1, 5, 20)
KAPPAS = (0.0001, 0.001, 0.01)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output() -> Path:
    return project_root() / "experiment_results" / "dyna_q_plus_sweep"


def default_summary_output() -> Path:
    return project_root() / "dyna_q_plus_sweep_summary"


def _number_tag(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def parameter_configurations() -> List[Dict[str, Any]]:
    result = []
    for alpha in ALPHAS:
        for planning in PLANNING_STEPS:
            for kappa in KAPPAS:
                result.append({
                    "method": "dyna_q_plus",
                    "method_label": "Dyna-Q+",
                    "algorithm": "dyna_q_plus",
                    "effective_initial_step": float(alpha),
                    "lambda": 0.0,
                    "planning_steps": int(planning),
                    "dyna_plus_kappa": float(kappa),
                    "config_id": "a%s_p%d_k%s" % (
                        _number_tag(alpha), planning, _number_tag(kappa)
                    ),
                })
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
    return {
        "protocol_version": 1,
        "name": "dyna_q_plus_parameter_sweep",
        "source_phase1_manifest": str(source_path.resolve()),
        "steps": int(source["steps"]),
        "seeds": seeds,
        "expected_runs": len(settings) * len(parameters) * len(seeds),
        "feature_representation": source["feature_representation"],
        "environment": source["environment"],
        "agent_common": source["agent_common"],
        "metrics": metrics,
        "schedule": source["schedule"],
        "settings": settings,
        "parameter_configurations": parameters,
        "seed_manifests": source["seed_manifests"],
        "method_order": ["dyna_q_plus"],
        "method_labels": {"dyna_q_plus": "Dyna-Q+"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and plot the 405-run Dyna-Q+ parameter sweep."
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
        "Dyna-Q+ sweep", output, args.summary_output.resolve(), manifest,
        args.workers, args.checkpoint_every, args.keep_checkpoints,
        tolerate_failed_configs=True,
    )


if __name__ == "__main__":
    main()
