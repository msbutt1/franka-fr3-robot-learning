#!/usr/bin/env python3
"""Convert FR3 raw recording episodes into a LeRobotDataset.

This converter expects episode folders produced by realsense_recorder.py:
metadata.json, robot_state.jsonl, actions.jsonl, and camera_*_rgb.mp4 files.
It uses LeRobot's writer API so the output is a normal LeRobot dataset with
Parquet state/action data and MP4 camera observations.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


FR3_FULL_STATE_NAMES = (
    [f"joint_{i}.pos" for i in range(7)]
    + [f"joint_{i}.vel" for i in range(7)]
    + ["ee.x", "ee.y", "ee.z"]
    + ["ee.qx", "ee.qy", "ee.qz", "ee.qw"]
    + ["gripper.width"]
)
FR3_FULL_ACTION_NAMES = ["target.ee.x", "target.ee.y", "target.ee.z", "target.ee.qx", "target.ee.qy", "target.ee.qz", "target.ee.qw", "target.gripper.width"]
DROID_STATE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.closedness"]
DROID_ACTION_NAMES = ["delta.ee.x", "delta.ee.y", "delta.ee.z", "delta.ee.rx", "delta.ee.ry", "delta.ee.rz", "gripper.closedness"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_dirs(raw_dir: Path, include_failed: bool) -> list[Path]:
    dirs = []
    for path in sorted(raw_dir.iterdir()):
        metadata_path = path / "metadata.json"
        if not path.is_dir() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("success", False) or include_failed:
            dirs.append(path)
    return dirs


def camera_files(ep_dir: Path) -> list[Path]:
    return sorted(ep_dir.glob("camera_*_rgb.mp4"))


def camera_key(video_path: Path) -> str:
    return "observation.images." + video_path.stem.removesuffix("_rgb").replace("-", "_")


def normalize_quaternion(q: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / norm
    if q[3] < 0:
        q = -q
    return q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def quaternion_delta_to_rotvec(q_current: list[float], q_next: list[float]) -> np.ndarray:
    q_current = normalize_quaternion(q_current)
    q_next = normalize_quaternion(q_next)
    q_delta = normalize_quaternion(quaternion_multiply(q_next, quaternion_inverse(q_current)))
    xyz = q_delta[:3]
    w = float(np.clip(q_delta[3], -1.0, 1.0))
    norm_xyz = float(np.linalg.norm(xyz))
    if norm_xyz < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(norm_xyz, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (xyz / norm_xyz * angle).astype(np.float32)


def gripper_closedness(sample: dict, gripper_max_width: float) -> float:
    width = float(sample.get("gripper_width", gripper_max_width))
    return float(np.clip(1.0 - width / gripper_max_width, 0.0, 1.0))


def fr3_full_state_vector(sample: dict) -> np.ndarray:
    return np.asarray(
        sample["q"]
        + sample["dq"]
        + sample["ee_translation"]
        + sample["ee_quaternion"]
        + [float(sample.get("gripper_width", 0.0))],
        dtype=np.float32,
    )


def fr3_full_action_vector(sample: dict) -> np.ndarray:
    return np.asarray(
        sample["ee_translation"]
        + sample["ee_quaternion"]
        + [float(sample.get("gripper_width", 0.0))],
        dtype=np.float32,
    )


def droid_state_vector(sample: dict, gripper_max_width: float) -> np.ndarray:
    return np.asarray(
        sample["ee_translation"]
        + sample["ee_quaternion"]
        + [gripper_closedness(sample, gripper_max_width)],
        dtype=np.float32,
    )


def droid_delta_action_vector(current: dict, next_sample: dict, gripper_max_width: float) -> np.ndarray:
    delta_xyz = np.asarray(next_sample["ee_translation"], dtype=np.float32) - np.asarray(current["ee_translation"], dtype=np.float32)
    delta_rot = quaternion_delta_to_rotvec(current["ee_quaternion"], next_sample["ee_quaternion"])
    return np.asarray(
        delta_xyz.tolist() + delta_rot.tolist() + [gripper_closedness(next_sample, gripper_max_width)],
        dtype=np.float32,
    )


def nearest_indices(sorted_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_times, query_times)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = idx - 1
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(sorted_times[idx] - query_times)
    return np.where(choose_left, left, idx)


def read_video_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_features(first_ep: Path, fps: int, schema: str) -> tuple[dict, list[str]]:
    video_paths = camera_files(first_ep)
    if not video_paths:
        raise RuntimeError(f"No camera videos found in {first_ep}")

    if schema == "droid_delta":
        state_names = DROID_STATE_NAMES
        action_names = DROID_ACTION_NAMES
    else:
        state_names = FR3_FULL_STATE_NAMES
        action_names = FR3_FULL_ACTION_NAMES

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
    }
    camera_keys = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {video_path}")
        height, width = frame.shape[:2]
        key = camera_key(video_path)
        camera_keys.append(key)
        features[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
            "info": {"fps": fps},
        }
    return features, camera_keys


def convert_episode(
    dataset,
    ep_dir: Path,
    camera_keys: list[str],
    task: str,
    max_frames: int | None,
    schema: str,
    gripper_max_width: float,
) -> int:
    states = read_jsonl(ep_dir / "robot_state.jsonl")
    if len(states) < 2:
        raise RuntimeError(f"{ep_dir} has too few robot_state samples")
    state_times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)

    video_paths = camera_files(ep_dir)
    if len(video_paths) != len(camera_keys):
        raise RuntimeError(f"{ep_dir} camera count changed: expected {len(camera_keys)}, got {len(video_paths)}")

    timestamp_rows = []
    for video_path in video_paths:
        name = video_path.stem.removesuffix("_rgb")
        rows = read_jsonl(ep_dir / f"{name}_timestamps.jsonl")
        if not rows:
            raise RuntimeError(f"Missing timestamps for {video_path}")
        timestamp_rows.append(rows)

    frame_count = min(len(rows) for rows in timestamp_rows)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count < 2:
        raise RuntimeError(f"{ep_dir} has too few synchronized frames")

    ref_times = np.asarray([row["time_ns"] for row in timestamp_rows[0][:frame_count]], dtype=np.int64)
    state_indices = nearest_indices(state_times, ref_times)

    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        for frame_idx in range(frame_count):
            images = [read_video_frame(cap) for cap in caps]
            if any(image is None for image in images):
                break
            state_idx = int(state_indices[frame_idx])
            action_idx = min(state_idx + 1, len(states) - 1)
            if schema == "droid_delta":
                state = droid_state_vector(states[state_idx], gripper_max_width)
                action = droid_delta_action_vector(states[state_idx], states[action_idx], gripper_max_width)
            else:
                state = fr3_full_state_vector(states[state_idx])
                action = fr3_full_action_vector(states[action_idx])
            frame = {
                "task": task,
                "observation.state": state,
                "action": action,
            }
            for key, image in zip(camera_keys, images, strict=True):
                frame[key] = image
            dataset.add_frame(frame)
    finally:
        for cap in caps:
            cap.release()
    dataset.save_episode()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot repo id, e.g. msbutt1/fr3-pick-place")
    parser.add_argument("--task", type=str, default="Pick up the cube from the cell and place it in the basket.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument("--max_frames_per_episode", type=int, default=None)
    parser.add_argument("--schema", choices=["droid_delta", "fr3_full"], default="droid_delta",
                        help="droid_delta uses 8D proprio state and 7D relative EE delta action.")
    parser.add_argument("--gripper_max_width", type=float, default=0.08,
                        help="Meters, used to normalize gripper closedness for DROID-style schema.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Could not import modern LeRobot. Install/run this in a LeRobot >=0.4 environment, "
            "then retry the converter."
        ) from exc

    episodes = episode_dirs(args.raw_dir, args.include_failed)
    if not episodes:
        raise SystemExit(f"No usable episodes found under {args.raw_dir}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    features, camera_keys = build_features(episodes[0], args.fps, args.schema)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=args.fps,
        robot_type="franka_fr3",
        features=features,
        use_videos=True,
    )

    converted = 0
    for ep_dir in episodes:
        print(f"[convert] {ep_dir.name}")
        n_frames = convert_episode(
            dataset,
            ep_dir,
            camera_keys,
            args.task,
            args.max_frames_per_episode,
            args.schema,
            args.gripper_max_width,
        )
        print(f"          saved {n_frames} frames")
        converted += 1
    dataset.finalize()
    print(f"[done] converted {converted} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
