#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/build_fr3_phase_sampling_manifest.py --help
"""Build a phase-aware OpenPI sample-index manifest from FR3 recordings.

The manifest remaps LeRobot *start indices* only. The underlying dataset stays
complete, so LeRobot can still fetch each start's original adjacent 16 actions
without crossing an artificial temporal gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CRITICAL_PHASES = {
    "cell pickup: to hover: horizontal",
    "cell pickup: to hover: descend to travel",
    "cell pickup: to hover: descend",
    "cell pickup: descend to grasp",
    "gripper_grasp",
    "cell pickup: lift & verify",
    "place in basket: to hover: horizontal",
    "place in basket: to hover: descend to travel",
    "place in basket: to hover: descend",
    "place in basket: descend to place",
    "place in basket: retreat to hover",
}
GRIPPER_ACTIONS = {"gripper_open", "gripper_grasp"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def successful_episodes(raw_dir: Path) -> list[Path]:
    result = []
    for episode in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        metadata_path = episode / "metadata.json"
        if not metadata_path.exists():
            continue
        if json.loads(metadata_path.read_text()).get("success", False):
            result.append(episode)
    return result


def nearest_indices(sorted_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sorted_times, query_times)
    indices = np.clip(indices, 1, len(sorted_times) - 1)
    left = indices - 1
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(sorted_times[indices] - query_times)
    return np.where(choose_left, left, indices)


def phase_name(action: dict) -> str:
    return str(action.get("payload", {}).get("label", action.get("action", "episode_start")))


def gripper_closedness(state: dict, max_width: float = 0.08) -> float:
    width = state.get("gripper_command_width")
    if width is None:
        width = state.get("gripper_width", max_width)
    return float(np.clip(1.0 - float(width) / max_width, 0.0, 1.0))


def phases_at(sample_times: np.ndarray, actions: list[dict]) -> list[str]:
    if not actions:
        return ["episode_start"] * len(sample_times)
    action_times = np.asarray([row["time_ns"] for row in actions], dtype=np.int64)
    indices = np.searchsorted(action_times, sample_times, side="right") - 1
    return ["episode_start" if index < 0 else phase_name(actions[int(index)]) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/fr3_real_pick_place_droid_v2")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--output-fps", type=int, default=15)
    parser.add_argument("--idle-velocity-threshold", type=float, default=0.02)
    parser.add_argument("--gripper-constant-threshold", type=float, default=0.02)
    parser.add_argument("--transition-context-seconds", type=float, default=1.0)
    parser.add_argument("--critical-weight", type=int, default=2)
    args = parser.parse_args()

    if args.action_horizon < 1:
        raise SystemExit("--action-horizon must be positive")
    if args.critical_weight < 1:
        raise SystemExit("--critical-weight must be at least 1")

    raw_dir = args.raw_dir.expanduser().resolve()
    episodes = successful_episodes(raw_dir)
    if not episodes:
        raise SystemExit(f"No successful episodes found under {raw_dir}")

    sample_indices: list[int] = []
    unique_indices: list[int] = []
    episode_summaries: list[dict] = []
    global_offset = 0
    dropped_idle = 0
    critical_unique = 0
    kept_transition = 0

    for episode_index, episode in enumerate(episodes):
        metadata = json.loads((episode / "metadata.json").read_text())
        states = read_jsonl(episode / "robot_state.jsonl")
        actions = read_jsonl(episode / "actions.jsonl")
        timestamp_files = sorted(episode.glob("camera_*_timestamps.jsonl"))
        if len(states) < 2 or not timestamp_files:
            raise RuntimeError(f"Incomplete episode: {episode}")

        timestamp_rows = [read_jsonl(path) for path in timestamp_files]
        frame_count = min(len(rows) for rows in timestamp_rows)
        source_fps = int(metadata.get("fps", 60))
        frame_stride = max(1, round(source_fps / args.output_fps))
        selected_frames = list(range(0, frame_count, frame_stride))
        sample_times = np.asarray(
            [timestamp_rows[0][frame]["time_ns"] for frame in selected_frames], dtype=np.int64
        )
        state_times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)
        state_indices = nearest_indices(state_times, sample_times)
        velocities = np.asarray(
            [states[int(index)].get("dq_d", states[int(index)]["dq"]) for index in state_indices],
            dtype=np.float64,
        )
        gripper = np.asarray(
            [
                gripper_closedness(states[min(int(index) + 1, len(states) - 1)])
                for index in state_indices
            ],
            dtype=np.float64,
        )
        phases = phases_at(sample_times, actions)
        transition_times = np.asarray(
            [row["time_ns"] for row in actions if row.get("action") in GRIPPER_ACTIONS], dtype=np.int64
        )
        transition_context_ns = int(args.transition_context_seconds * 1e9)

        episode_kept = 0
        episode_dropped = 0
        episode_critical = 0
        # Do not use a short tail as though it were a full action-horizon chunk.
        for local_index in range(max(0, len(sample_times) - args.action_horizon + 1)):
            chunk_end = local_index + args.action_horizon
            velocity_chunk = velocities[local_index:chunk_end]
            gripper_chunk = gripper[local_index:chunk_end]
            arm_is_idle = float(np.max(np.abs(velocity_chunk), initial=0.0)) < args.idle_velocity_threshold
            gripper_is_constant = (
                float(np.ptp(gripper_chunk)) < args.gripper_constant_threshold if len(gripper_chunk) else True
            )
            near_transition = bool(
                len(transition_times)
                and np.min(np.abs(transition_times - sample_times[local_index])) <= transition_context_ns
            )
            critical = phases[local_index] in CRITICAL_PHASES or near_transition

            if arm_is_idle and gripper_is_constant and not near_transition:
                dropped_idle += 1
                episode_dropped += 1
                continue

            dataset_index = global_offset + local_index
            unique_indices.append(dataset_index)
            weight = args.critical_weight if critical else 1
            sample_indices.extend([dataset_index] * weight)
            episode_kept += 1
            if critical:
                critical_unique += 1
                episode_critical += 1
            if near_transition:
                kept_transition += 1

        episode_summaries.append(
            {
                "episode_index": episode_index,
                "episode_name": episode.name,
                "global_start": global_offset,
                "frames": len(sample_times),
                "kept_unique": episode_kept,
                "dropped_idle": episode_dropped,
                "critical_unique": episode_critical,
            }
        )
        global_offset += len(sample_times)

    manifest = {
        "version": 1,
        "repo_id": args.repo_id,
        "raw_dir": str(raw_dir),
        "dataset_total_frames": global_offset,
        "successful_episodes": len(episodes),
        "action_horizon": args.action_horizon,
        "output_fps": args.output_fps,
        "idle_velocity_threshold": args.idle_velocity_threshold,
        "gripper_constant_threshold": args.gripper_constant_threshold,
        "transition_context_seconds": args.transition_context_seconds,
        "critical_weight": args.critical_weight,
        "kept_unique_count": len(unique_indices),
        "dropped_idle_count": dropped_idle,
        "critical_unique_count": critical_unique,
        "kept_transition_count": kept_transition,
        "sample_count_with_repeats": len(sample_indices),
        "sample_indices": sample_indices,
        "episodes": episode_summaries,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[manifest] episodes={len(episodes)} total_frames={global_offset}")
    print(
        f"[manifest] kept_unique={len(unique_indices)} dropped_idle={dropped_idle} "
        f"critical_unique={critical_unique}"
    )
    print(f"[manifest] weighted_samples={len(sample_indices)} output={output}")


if __name__ == "__main__":
    main()
