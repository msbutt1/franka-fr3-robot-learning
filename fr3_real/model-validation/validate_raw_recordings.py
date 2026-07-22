#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/validate_raw_recordings.py --help
"""Validate raw FR3 episodes before LeRobot conversion or training."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def first_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                return json.loads(line)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--min_state_hz", type=float, default=10.0)
    args = parser.parse_args()

    episodes = []
    excluded = []
    episode_dirs = sorted(path for path in args.raw_dir.iterdir() if path.is_dir())
    for episode_dir in episode_dirs:
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            excluded.append((episode_dir.name, "missing metadata.json", ""))
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            excluded.append((episode_dir.name, f"invalid metadata: {type(exc).__name__}", ""))
            continue
        if not metadata.get("success", False):
            status = "failed" if metadata.get("success") is False else "unfinished"
            excluded.append((episode_dir.name, status, str(metadata.get("note", ""))))
            continue
        start = int(metadata.get("record_start_time_ns", 0))
        stop = int(metadata.get("record_stop_time_ns", 0))
        duration = (stop - start) / 1e9 if stop > start else 0.0
        state_path = episode_dir / "robot_state.jsonl"
        state_count = jsonl_count(state_path)
        first_state = first_jsonl(state_path)
        required_state_keys = {"q", "dq", "q_d", "dq_d", "ee_translation", "gripper_width", "gripper_command_width"}
        missing_state_keys = sorted(required_state_keys - first_state.keys())
        timestamp_files = sorted(episode_dir.glob("camera_*_timestamps.jsonl"))
        camera_counts = [jsonl_count(path) for path in timestamp_files]
        state_hz = state_count / duration if duration > 0 else 0.0
        camera_hz = [count / duration for count in camera_counts] if duration > 0 else []
        valid = (
            duration > 0
            and state_count >= 2
            and state_hz >= args.min_state_hz
            and not missing_state_keys
            and len(camera_counts) >= 2
            and all(count >= 2 for count in camera_counts)
        )
        episodes.append(
            (valid, episode_dir.name, duration, state_count, state_hz, camera_counts, camera_hz, missing_state_keys)
        )

    valid_count = sum(row[0] for row in episodes)
    print(f"episode folders: {len(episode_dirs)}")
    print(f"excluded failed/unfinished folders: {len(excluded)}")
    print(f"successful episodes: {len(episodes)}")
    print(f"valid for robot-policy conversion: {valid_count}")
    print(f"invalid: {len(episodes) - valid_count}")
    if episodes:
        durations = [row[2] for row in episodes]
        print(
            "duration seconds: "
            f"min={min(durations):.2f} median={statistics.median(durations):.2f} max={max(durations):.2f}"
        )
    for valid, name, duration, state_count, state_hz, camera_counts, camera_hz, missing_state_keys in episodes:
        if valid:
            continue
        camera_summary = ", ".join(
            f"{count} ({hz:.1f} Hz)" for count, hz in zip(camera_counts, camera_hz, strict=False)
        )
        print(
            f"[INVALID] {name}: duration={duration:.2f}s states={state_count} "
            f"({state_hz:.1f} Hz) missing_state_keys={missing_state_keys} cameras=[{camera_summary}]"
        )
    for name, status, note in excluded:
        suffix = f": {note}" if note else ""
        print(f"[EXCLUDED] {name}: {status}{suffix}")

    if valid_count != len(episodes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
