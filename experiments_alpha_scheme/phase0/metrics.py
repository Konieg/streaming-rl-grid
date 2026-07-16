"""Metrics shared by the two Phase 0 subexperiments."""

from typing import Dict, List, Sequence

import numpy as np

from .protocol import METRIC_WINDOW, RECOVERY_SMOOTHING, RECOVERY_TOLERANCE


def _segments(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    result: List[Dict[str, object]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            result.append({"label": labels[start], "start": start, "end": index})
            start = index
    return result


def _smoothed(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.empty(0, dtype=float)
    kernel = np.full(window, 1.0 / window)
    return np.convolve(values, kernel, mode="valid")


def _change_metrics(
    signal: np.ndarray,
    segments: List[Dict[str, object]],
    higher_is_better: bool,
) -> List[Dict[str, object]]:
    changes: List[Dict[str, object]] = []
    for segment_index, segment in enumerate(segments[1:], start=1):
        start = int(segment["start"])
        end = int(segment["end"])
        pre = signal[max(0, start - METRIC_WINDOW):start]
        post = signal[start:min(end, start + 500)]
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        if pre.size == 0 or post.size == 0:
            continue
        baseline = float(np.mean(pre))
        smooth = _smoothed(post, RECOVERY_SMOOTHING)
        tolerance = RECOVERY_TOLERANCE * max(1.0, abs(baseline))
        if higher_is_better:
            loss = np.maximum(baseline - post, 0.0)
            recovered = np.flatnonzero(smooth >= baseline - tolerance)
        else:
            loss = np.maximum(post - baseline, 0.0)
            recovered = np.flatnonzero(smooth <= baseline + tolerance)
        changes.append(
            {
                "step": start,
                "from": segments[segment_index - 1]["label"],
                "to": segment["label"],
                "prechange_baseline": baseline,
                "postchange_auec": float(np.sum(loss)),
                "recovery_steps": (
                    None if recovered.size == 0 else int(recovered[0] + RECOVERY_SMOOTHING)
                ),
            }
        )
    return changes


def summarize_trace(
    rewards: np.ndarray,
    squared_td_errors: np.ndarray,
    collisions: np.ndarray,
    goals: np.ndarray,
    mode_labels: Sequence[str],
    policy_entropies: np.ndarray,
    task: str,
) -> Dict[str, object]:
    segments = _segments(mode_labels)
    for segment in segments:
        start, end = int(segment["start"]), int(segment["end"])
        tail_start = max(start, end - METRIC_WINDOW)
        segment["mean_reward"] = float(np.nanmean(rewards[tail_start:end]))
        segment["mean_squared_td_error"] = float(
            np.nanmean(squared_td_errors[tail_start:end])
        )
        segment["goal_rate_per_1000"] = float(np.nanmean(goals[tail_start:end]) * 1000.0)

    signal = squared_td_errors if task == "prediction" else rewards
    changes = _change_metrics(signal, segments, higher_is_better=(task == "control"))
    repeated_modes: List[Dict[str, object]] = []
    first_by_label: Dict[str, Dict[str, object]] = {}
    for segment in segments:
        label = str(segment["label"])
        if label not in first_by_label:
            first_by_label[label] = segment
        else:
            first = first_by_label[label]
            repeated_modes.append(
                {
                    "label": label,
                    "first_start": int(first["start"]),
                    "recurrence_start": int(segment["start"]),
                    "first_stable_reward": float(first["mean_reward"]),
                    "recurrence_stable_reward": float(segment["mean_reward"]),
                    "first_stable_squared_td_error": float(first["mean_squared_td_error"]),
                    "recurrence_stable_squared_td_error": float(
                        segment["mean_squared_td_error"]
                    ),
                }
            )

    return {
        "mean_reward": float(np.nanmean(rewards)),
        "mean_squared_td_error": float(np.nanmean(squared_td_errors)),
        "goal_rate_per_1000": float(np.nanmean(goals) * 1000.0),
        "collision_rate": float(np.nanmean(collisions)),
        "mean_policy_entropy": float(np.nanmean(policy_entropies)),
        "segments": segments,
        "changes": changes,
        "recurrences": repeated_modes,
    }
