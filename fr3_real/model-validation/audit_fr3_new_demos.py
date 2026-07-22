#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/audit_fr3_new_demos.py --help
"""Validate new FR3 demonstrations and detect held-out spatial leakage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def episode_xy(metadata: dict) -> np.ndarray:
    if "cell_xyz" in metadata:
        return np.asarray(metadata["cell_xyz"][:2], dtype=np.float64)
    source = metadata.get("source", {})
    if "x" in source and "y" in source:
        return np.asarray([source["x"], source["y"]], dtype=np.float64)
    raise ValueError("metadata has neither cell_xyz nor source x/y")


def manifest_positions(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data.get("positions", []))


def finite_vector(row: dict, key: str, size: int) -> bool:
    value = np.asarray(row.get(key, []), dtype=np.float64)
    return value.shape == (size,) and bool(np.isfinite(value).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--protected-manifest", type=Path, action="append", required=True)
    parser.add_argument("--min-distance", type=float, default=0.03, help="Minimum XY distance in meters")
    parser.add_argument("--min-episodes", type=int, default=60)
    parser.add_argument(
        "--verify-video-decode",
        action="store_true",
        help="Verify that OpenCV can decode the first frame of each camera stream.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cv2 = None
    if args.verify_video_decode:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise SystemExit("--verify-video-decode requires OpenCV in the active Python environment") from exc
        cv2 = cv2_module

    raw_dir = args.raw_dir.expanduser().resolve()
    protected: list[dict] = []
    for manifest_path in args.protected_manifest:
        manifest_path = manifest_path.expanduser().resolve()
        for position in manifest_positions(manifest_path):
            protected.append({**position, "manifest": str(manifest_path)})
    if not protected:
        raise SystemExit("Protected manifests contain no positions")

    reports: list[dict] = []
    failures: list[str] = []
    excluded_unsuccessful: list[str] = []
    for episode in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        metadata_path = episode / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("success", False):
            excluded_unsuccessful.append(episode.name)
            continue
        episode_failures: list[str] = []
        try:
            xy = episode_xy(metadata)
        except ValueError as exc:
            episode_failures.append(str(exc))
            xy = np.asarray([math.nan, math.nan])

        nearest = None
        nearest_distance = math.inf
        if np.isfinite(xy).all():
            for position in protected:
                distance = float(np.linalg.norm(xy - [position["x"], position["y"]]))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = position
            if nearest_distance < args.min_distance:
                episode_failures.append(
                    f"held-out leakage: {nearest_distance:.4f} m from {nearest['eval_id']} "
                    f"in {Path(nearest['manifest']).name}"
                )

        required_files = [episode / "robot_state.jsonl", episode / "actions.jsonl"]
        missing = [path.name for path in required_files if not path.is_file()]
        if missing:
            episode_failures.append(f"missing files: {', '.join(missing)}")
            states = []
            actions = []
        else:
            states = read_jsonl(required_files[0])
            actions = read_jsonl(required_files[1])
        if len(states) < 2:
            episode_failures.append("fewer than two robot states")
        else:
            times = np.asarray([row.get("time_ns", -1) for row in states], dtype=np.int64)
            if np.any(np.diff(times) <= 0):
                episode_failures.append("robot timestamps are not strictly increasing")
            if not all(
                finite_vector(row, "q", 7)
                and finite_vector(row, "dq", 7)
                and finite_vector(row, "dq_d", 7)
                for row in states
            ):
                episode_failures.append("invalid or non-finite q/dq/dq_d")
        action_names = {str(row.get("action")) for row in actions}
        for required_action in ("gripper_grasp", "gripper_open"):
            if required_action not in action_names:
                episode_failures.append(f"missing {required_action} action marker")
        timestamp_files = sorted(episode.glob("camera_*_timestamps.jsonl"))
        video_files = sorted(episode.glob("camera_*_rgb.mp4"))
        if len(timestamp_files) != 2 or len(video_files) != 2:
            episode_failures.append(
                f"expected two camera timestamp/video streams, found {len(timestamp_files)}/{len(video_files)}"
            )
        elif cv2 is not None:
            for video_path in video_files:
                capture = cv2.VideoCapture(str(video_path))
                readable, _ = capture.read()
                capture.release()
                if not readable:
                    episode_failures.append(f"cannot decode first frame of {video_path.name}")

        report = {
            "episode": episode.name,
            "xy": xy.tolist(),
            "nearest_protected_distance": nearest_distance,
            "nearest_protected_id": None if nearest is None else nearest.get("eval_id"),
            "failures": episode_failures,
        }
        reports.append(report)
        failures.extend(f"{episode.name}: {message}" for message in episode_failures)

    if len(reports) < args.min_episodes:
        failures.append(f"only {len(reports)} episodes; require at least {args.min_episodes}")
    result = {
        "version": 1,
        "raw_dir": str(raw_dir),
        "protected_manifests": [str(path.expanduser().resolve()) for path in args.protected_manifest],
        "minimum_distance_m": args.min_distance,
        "episode_count": len(reports),
        "excluded_unsuccessful_episodes": excluded_unsuccessful,
        "passed": not failures,
        "failures": failures,
        "episodes": reports,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[audit] successful={len(reports)} excluded_unsuccessful={len(excluded_unsuccessful)} "
        f"failures={len(failures)} output={output}"
    )
    if failures:
        for failure in failures[:20]:
            print(f"[audit:error] {failure}")
        if len(failures) > 20:
            print(f"[audit:error] ... and {len(failures) - 20} more")
        raise SystemExit(1)
    print("[audit] new demonstrations passed integrity and held-out leakage checks")


if __name__ == "__main__":
    main()
