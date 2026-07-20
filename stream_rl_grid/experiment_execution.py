"""Shared resumable execution loop for fixed-manifest comparison experiments."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from .phase1_plot import aggregate
from .phase1_sweep import _atomic_json, build_jobs, run_job


def load_or_create_manifest(output: Path, requested: Dict[str, Any]) -> Dict[str, Any]:
    path = output / "experiment_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        comparable = (
            "protocol_version", "name", "steps", "seeds", "settings",
            "parameter_configurations", "schedule", "metrics",
            "feature_representation", "agent_common",
        )
        if any(existing.get(key) != requested.get(key) for key in comparable):
            raise ValueError(
                "Existing experiment_manifest.json differs from this protocol; "
                "choose a different --output directory."
            )
        # These fields are additive protocol metadata. In particular,
        # setting_seed_manifests can safely repair a setting whose jobs never
        # passed environment construction, without invalidating completed jobs
        # from other settings in the same output directory.
        changed = False
        for key in (
            "setting_seed_manifests", "method_order", "method_labels",
            "winner_source_setting", "source_selected_configs",
            "representation_protocol",
        ):
            if key not in requested:
                continue
            if key in existing and existing[key] != requested[key]:
                raise ValueError(
                    "Existing experiment manifest has incompatible %s metadata; "
                    "choose a different --output directory." % key
                )
            if key not in existing:
                existing[key] = requested[key]
                changed = True
        if changed:
            _atomic_json(path, existing)
        return existing
    _atomic_json(path, requested)
    return requested


def execute(
    name: str,
    output: Path,
    summary_output: Path,
    manifest: Dict[str, Any],
    workers: int,
    checkpoint_interval: int,
    keep_checkpoints: bool,
    tolerate_failed_configs: bool = False,
) -> None:
    jobs = build_jobs(manifest)
    pending = [
        job for job in jobs
        if not (output / job["relative_dir"] / "summary.json").exists()
    ]
    completed_before = len(jobs) - len(pending)
    print(
        "%s: %d total, %d already complete, %d pending, %d workers"
        % (name, len(jobs), completed_before, len(pending), workers),
        flush=True,
    )
    completed = failed = 0
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_job, str(output), manifest, job, checkpoint_interval,
                    keep_checkpoints,
                ): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed", "job": job["relative_dir"],
                        "error": "worker process error: %r" % exc,
                    }
                if result["status"] == "failed":
                    failed += 1
                    print("FAILED %s: %s" % (result["job"], result["error"]), flush=True)
                else:
                    completed += 1
                print(
                    "[%d/%d] complete_now=%d failed=%d latest=%s"
                    % (completed_before + completed + failed, len(jobs), completed,
                       failed, result["job"]),
                    flush=True,
                )
    if failed and not tolerate_failed_configs:
        raise SystemExit(
            "%d run(s) failed; inspect failure.json and execute the same command "
            "again to resume/retry." % failed
        )
    if failed:
        print(
            "%d numerically failed configuration run(s) will be excluded from selection."
            % failed,
            flush=True,
        )
    else:
        print("All runs complete.", flush=True)
    print("Creating plots in %s" % summary_output, flush=True)
    aggregate(output, summary_dir=summary_output, allow_incomplete=False)
