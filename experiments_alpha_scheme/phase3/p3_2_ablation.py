"""P3.2: frozen causal ablations of D=71 adaptive models."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.features import NUISANCE_DIMENSION, NuisanceFeatureEncoder
from experiments.learners import LinearDifferentialLearner
from experiments.lfa_runner import evaluate_frozen
from experiments.plotting import plot_ablation_results
from experiments.phase0.protocol import condition_by_name
from experiments.result_schema import SCHEMA_VERSION, load_result_bundle


def _probe_offsets(condition: str) -> Tuple[Tuple[str, int], ...]:
    if condition == "stationary":
        return (("A", 0),)
    if condition == "seasonal_wind":
        return (("A", 0), ("B", 500), ("A_recurrence", 2_000))
    return (("A", 0), ("B", 500), ("A_recurrence", 1_000))


def _masks(alphas: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    dimension = len(alphas)
    count = NUISANCE_DIMENSION
    ordered = np.argsort(alphas, kind="stable")
    middle_start = (dimension - count) // 2
    selected = {
        "unablated": np.asarray([], dtype=int),
        "nuisance": np.flatnonzero(groups == "nuisance"),
        "low_alpha": ordered[:count],
        "middle_alpha": ordered[middle_start:middle_start + count],
        "high_alpha": ordered[-count:],
    }
    masks = {}
    for name, indices in selected.items():
        mask = np.ones(dimension, dtype=np.float64)
        mask[indices] = 0.0
        masks[name] = mask
    return masks


def run_ablation(summary_path: Path, output: Path) -> Path:
    bundle = load_result_bundle(summary_path)
    if bundle["phase"] != "phase3":
        raise ValueError("P3.2 requires a Phase 3 result bundle")
    task = str(bundle["task"])
    records: List[Dict[str, object]] = []
    for run in bundle["runs"]:
        if run["method"] != "adaptive" or not run.get("model_file"):
            continue
        condition = str(run["condition"])
        with np.load(summary_path.parent / run["model_file"]) as model:
            groups = model["groups"]
            encoder = NuisanceFeatureEncoder(5, 5)
            learner = LinearDifferentialLearner(encoder.dimension, groups, adaptive=True)
            learner.weights[:] = model["weights"]
            learner.beta[:] = model["beta"]
            learner.h[:] = model["h"]
            learner.average_reward = float(model["average_reward"][0])
        evaluations = []
        for ablation, mask in _masks(learner.alphas, groups).items():
            for mode, offset in _probe_offsets(condition):
                metrics = evaluate_frozen(
                    learner, encoder, condition_by_name(condition), task,
                    int(run["seed"]) + 40_000, mask=mask, burn_in_steps=offset,
                )
                evaluations.append(
                    {"ablation": ablation, "mode": mode, "burn_in_steps": offset, **metrics}
                )
        records.append(
            {
                "run_id": run["run_id"],
                "condition": condition,
                "seed": run["seed"],
                "deleted_features_per_rank_group": NUISANCE_DIMENSION,
                "evaluations": evaluations,
            }
        )
    if not records:
        raise ValueError("No adaptive model artifacts found in the supplied bundle")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase3",
        "subexperiment": "p3_2_frozen_ablation",
        "task": task,
        "source_summary": str(summary_path.resolve()),
        "frozen": ["weights", "alphas", "average_reward", "policy"],
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "ablation.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    plot_ablation_results(payload, output)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="P3.1 prediction or control summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = run_ablation(
        args.summary, args.output or args.summary.parent / "ablation"
    )
    print("Ablation written to %s" % destination)


if __name__ == "__main__":
    main()
