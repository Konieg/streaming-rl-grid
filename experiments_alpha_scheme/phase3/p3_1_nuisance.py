"""P3.1: D=71 nuisance-feature test after the Phase 1/2 gate."""

import argparse
from pathlib import Path

from experiments.lfa_runner import all_mechanisms, run_matrix
from experiments.phase0.protocol import DEFAULT_SEEDS, DEFAULT_STEPS, condition_by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("stationary", "seasonal_wind", "hidden_context", "moving_goal"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "p3_1_nuisance")
    args = parser.parse_args()
    condition = (condition_by_name(args.condition),)
    for task in ("prediction", "control"):
        run_matrix(
            "phase3", "p3_1_nuisance_%s" % task, task, all_mechanisms(),
            args.steps, args.seeds, args.output / task, conditions=condition,
            nuisance_features=True, save_models=True,
        )


if __name__ == "__main__":
    main()

