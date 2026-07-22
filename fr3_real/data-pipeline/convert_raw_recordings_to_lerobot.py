#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/convert_raw_recordings_to_lerobot.py --help
"""Convert FR3 raw recording episodes into a LeRobotDataset.

This converter expects episode folders produced by realsense_recorder.py:
metadata.json, robot_state.jsonl, actions.jsonl, and camera_*_rgb.mp4 files.
It uses LeRobot's writer API so the output is a normal LeRobot dataset with
Parquet state/action data and MP4 camera observations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
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
DROID_JOINT_STATE_NAMES = [f"joint_{i}.pos" for i in range(7)]
DROID_GRIPPER_STATE_NAMES = ["gripper.closedness"]
DROID_ACTION_NAMES = [f"joint_{i}.velocity" for i in range(7)] + ["gripper.closedness"]


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


def gripper_command_closedness(sample: dict, gripper_max_width: float) -> float:
    width = sample.get("gripper_command_width")
    if width is None:
        return gripper_closedness(sample, gripper_max_width)
    return float(np.clip(1.0 - float(width) / gripper_max_width, 0.0, 1.0))


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


def droid_joint_state_vector(sample: dict) -> np.ndarray:
    return np.asarray(sample["q"], dtype=np.float32)


def droid_joint_velocity_action_vector(sample: dict, next_sample: dict, gripper_max_width: float) -> np.ndarray:
    """DROID-compatible action: 7 measured joint velocities and next gripper state.

    Franky's dq_d is libfranka's desired joint velocity at the sampled control
    instant, which is a closer match to the original DROID velocity command than
    measured dq. The gripper channel is the explicit persisted command.
    """
    return np.asarray(
        list(sample.get("dq_d", sample["dq"])) + [gripper_command_closedness(next_sample, gripper_max_width)],
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


def build_features(
    first_ep: Path, schema: str, image_width: int | None, image_height: int | None
) -> tuple[dict, list[tuple[str, int]]]:
    video_paths = camera_files(first_ep)
    if not video_paths:
        raise RuntimeError(f"No camera videos found in {first_ep}")

    if schema == "droid_joint_velocity":
        state_names = DROID_JOINT_STATE_NAMES
        action_names = DROID_ACTION_NAMES
    else:
        state_names = FR3_FULL_STATE_NAMES
        action_names = FR3_FULL_ACTION_NAMES

    if schema == "droid_joint_velocity":
        # These exact names are consumed by OpenPI's LeRobotDROIDDataConfig.
        features = {
            "joint_position": {"dtype": "float32", "shape": (7,), "names": DROID_JOINT_STATE_NAMES},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": DROID_GRIPPER_STATE_NAMES},
            "actions": {"dtype": "float32", "shape": (8,), "names": DROID_ACTION_NAMES},
        }
        required_cameras = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    else:
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)},
            "action": {"dtype": "float32", "shape": (len(action_names),), "names": list(action_names)},
        }
        required_cameras = []

    camera_sources: list[tuple[str, int]] = []
    if schema != "droid_joint_velocity":
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {video_path}")
            height, width = frame.shape[:2]
            if image_width is not None and image_height is not None:
                width, height = image_width, image_height
            key = camera_key(video_path)
            camera_sources.append((key, len(camera_sources)))
            features[key] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
    if schema == "droid_joint_velocity":
        if len(video_paths) < 2:
            raise RuntimeError("pi05_droid conversion requires the two RealSense camera videos.")
        # OpenPI's DROID adapter requires three image keys. The second FR3 view
        # fills the wrist slot; exterior_image_2_left duplicates camera 0.
        camera_sources = [
            (required_cameras[0], 0),
            (required_cameras[1], 0),
            (required_cameras[2], 1),
        ]
        for key, source_idx in camera_sources:
            source_path = video_paths[source_idx]
            cap = cv2.VideoCapture(str(source_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {source_path}")
            height, width = frame.shape[:2]
            if image_width is not None and image_height is not None:
                width, height = image_width, image_height
            features[key] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
    return features, camera_sources


def convert_episode(
    dataset,
    ep_dir: Path,
    camera_sources: list[tuple[str, int]],
    task: str,
    max_frames: int | None,
    schema: str,
    gripper_max_width: float,
    image_width: int | None,
    image_height: int | None,
) -> int:
    states = read_jsonl(ep_dir / "robot_state.jsonl")
    if len(states) < 2:
        raise RuntimeError(f"{ep_dir} has too few robot_state samples")
    state_times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)

    video_paths = camera_files(ep_dir)
    if schema != "droid_joint_velocity" and len(video_paths) != len(camera_sources):
        raise RuntimeError(f"{ep_dir} camera count changed: expected {len(camera_sources)}, got {len(video_paths)}")

    timestamp_rows = []
    for video_path in video_paths:
        name = video_path.stem.removesuffix("_rgb")
        rows = read_jsonl(ep_dir / f"{name}_timestamps.jsonl")
        if not rows:
            raise RuntimeError(f"Missing timestamps for {video_path}")
        timestamp_rows.append(rows)

    source_fps = int(json.loads((ep_dir / "metadata.json").read_text()).get("fps", 60))
    frame_stride = max(1, round(source_fps / 15)) if schema == "droid_joint_velocity" else 1
    frame_count = min(len(rows) for rows in timestamp_rows)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count < 2:
        raise RuntimeError(f"{ep_dir} has too few synchronized frames")

    selected_frame_indices = list(range(0, frame_count, frame_stride))
    if max_frames is not None:
        selected_frame_indices = selected_frame_indices[:max_frames]
    if len(selected_frame_indices) < 2:
        raise RuntimeError(f"{ep_dir} has too few frames after resampling")
    ref_times = np.asarray([timestamp_rows[0][idx]["time_ns"] for idx in selected_frame_indices], dtype=np.int64)
    state_indices = nearest_indices(state_times, ref_times)

    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        saved_frames = 0
        selected_set = set(selected_frame_indices)
        for frame_idx in range(frame_count):
            images = [read_video_frame(cap) for cap in caps]
            if any(image is None for image in images):
                break
            if frame_idx not in selected_set:
                continue
            if image_width is not None and image_height is not None:
                images = [cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA) for image in images]
            state_idx = int(state_indices[saved_frames])
            action_idx = min(state_idx + 1, len(states) - 1)
            if schema == "droid_joint_velocity":
                frame = {
                    "task": task,
                    "joint_position": droid_joint_state_vector(states[state_idx]),
                    "gripper_position": np.asarray([gripper_closedness(states[state_idx], gripper_max_width)], dtype=np.float32),
                    "actions": droid_joint_velocity_action_vector(states[state_idx], states[action_idx], gripper_max_width),
                }
            else:
                state = fr3_full_state_vector(states[state_idx])
                action = fr3_full_action_vector(states[action_idx])
                frame = {"task": task, "observation.state": state, "action": action}
            for key, source_idx in camera_sources:
                frame[key] = images[source_idx]
            dataset.add_frame(frame)
            saved_frames += 1
    finally:
        for cap in caps:
            cap.release()
    if saved_frames == 0:
        raise RuntimeError(
            f"{ep_dir}: OpenCV decoded zero synchronized frames from "
            f"{', '.join(path.name for path in video_paths)}"
        )
    dataset.save_episode()
    return saved_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot repo id, e.g. msbutt1/fr3-pick-place")
    parser.add_argument("--task", type=str, default="Pick up the blue cube and place it in the basket.")
    parser.add_argument("--fps", type=int, default=15, help="Output dataset FPS. pi05_droid expects 15 Hz.")
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument("--max_frames_per_episode", type=int, default=None)
    parser.add_argument("--schema", choices=["droid_joint_velocity", "fr3_full"], default="droid_joint_velocity",
                        help="droid_joint_velocity matches OpenPI pi05_droid; fr3_full is diagnostic only.")
    parser.add_argument("--gripper_max_width", type=float, default=0.08,
                        help="Meters, used to normalize gripper closedness for DROID-style schema.")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Converted image width. Default preserves 640x480's 4:3 aspect at half size.")
    parser.add_argument("--image_height", type=int, default=240,
                        help="Converted image height. OpenPI performs the final model resize.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # OpenPI's pinned legacy LeRobot resolves this at import time. Treat
    # --output_dir as the cache root, exactly as the existing VM converters do.
    os.environ["HF_LEROBOT_HOME"] = str(args.output_dir.expanduser().resolve())

    legacy_root = None
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            # This is the API used by the LeRobot revision pinned by OpenPI.
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            legacy_root = Path(HF_LEROBOT_HOME)
        except ImportError as exc:
            raise SystemExit(
                "Could not import LeRobot. Run this from the OpenPI environment, "
                "which installs the compatible LeRobot revision."
            ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Could not import LeRobot: {exc}"
        ) from exc

    episodes = episode_dirs(args.raw_dir, args.include_failed)
    if not episodes:
        raise SystemExit(f"No usable episodes found under {args.raw_dir}")
    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    supports_root = "root" in create_parameters
    # Both the VM's legacy LeRobot and its existing conversion scripts use
    # --output_dir as a cache parent, with the actual dataset at <root>/<repo>.
    destination = args.output_dir / args.repo_id
    if destination.exists():
        if not args.overwrite:
            raise SystemExit(f"{destination} exists. Pass --overwrite to replace it.")
        shutil.rmtree(destination)

    features, camera_sources = build_features(episodes[0], args.schema, args.image_width, args.image_height)
    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        # DROID/OpenPI uses Panda kinematic conventions. FR3 shares the
        # relevant seven-joint Franka arm interface for this data adapter.
        "robot_type": "panda" if args.schema == "droid_joint_velocity" else "franka_fr3",
        "features": features,
    }
    if legacy_root is not None:
        create_kwargs.update({
            "root": str(destination),
            # LeRobot 0.1's multiprocess image writer can report completion
            # before its first PNG is visible on a local workstation filesystem.
            # That leaves a sparse episode image directory and video encoding
            # then fails on frame_000000.png. Synchronous writes are slower but
            # deterministic for this one-time conversion.
            "image_writer_threads": 0,
            "image_writer_processes": 0,
        })
    elif supports_root:
        create_kwargs.update({"root": destination, "use_videos": True})
    dataset = LeRobotDataset.create(**create_kwargs)

    converted = 0
    for ep_dir in episodes:
        print(f"[convert] {ep_dir.name}")
        n_frames = convert_episode(
            dataset,
            ep_dir,
            camera_sources,
            args.task,
            args.max_frames_per_episode,
            args.schema,
            args.gripper_max_width,
            args.image_width,
            args.image_height,
        )
        print(f"          saved {n_frames} frames")
        converted += 1
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"[done] converted {converted} episodes to {destination}")


if __name__ == "__main__":
    main()
