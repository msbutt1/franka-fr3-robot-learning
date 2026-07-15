#!/usr/bin/env python3
"""Validate the FR3 LeRobot dataset layout expected by OpenPI pi0.5-DROID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_FEATURES = {
    "exterior_image_1_left": None,
    "wrist_image_left": None,
    "joint_position": (7,),
    "gripper_position": (1,),
    "actions": (8,),
}


def shape_tuple(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(x) for x in value)


def load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expected-fps", type=int, default=15)
    parser.add_argument("--strict-fps", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    info = load_info(dataset_dir)
    features = info.get("features", {})
    errors = []
    warnings = []

    for name, expected_shape in REQUIRED_FEATURES.items():
        feature = features.get(name)
        if feature is None:
            errors.append(f"missing feature {name!r}")
            continue
        actual_shape = shape_tuple(feature.get("shape"))
        if expected_shape is not None and actual_shape != expected_shape:
            errors.append(f"{name!r} shape is {actual_shape}, expected {expected_shape}")

    fps = info.get("fps")
    if fps != args.expected_fps:
        message = f"dataset fps is {fps}, expected {args.expected_fps}"
        if args.strict_fps:
            errors.append(message)
        else:
            warnings.append(message)

    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    if not total_episodes:
        errors.append("dataset has no episodes")
    if not total_frames:
        errors.append("dataset has no frames")

    print(f"dataset={dataset_dir}")
    print(f"repo_id={info.get('repo_id')}")
    print(f"robot_type={info.get('robot_type')}")
    print(f"fps={fps}")
    print(f"total_episodes={total_episodes}")
    print(f"total_frames={total_frames}")
    print("features=" + ", ".join(sorted(features)))

    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(2)

    print("[ok] dataset has the required OpenPI pi0.5-DROID FR3 features")


if __name__ == "__main__":
    main()
