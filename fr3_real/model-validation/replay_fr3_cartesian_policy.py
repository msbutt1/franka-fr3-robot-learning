#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/replay_fr3_cartesian_policy.py --help
"""Replay raw FR3 observations through a Cartesian policy without robot access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import RECORDINGS_DIR, SHADOW_LOGS_DIR


DEFAULT_EPISODE = "20260713_091547_printed_cell_001"
DEFAULT_PROMPT = "Pick up the cube from the cell and place it in the basket."
EXTERIOR_CAMERA = "camera_0_336222074819"
WRIST_CAMERA = "camera_1_243322074869"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def nearest_index(records: list[dict], time_ns: int) -> int:
    timestamps = np.fromiter((row["time_ns"] for row in records), dtype=np.int64)
    return int(np.argmin(np.abs(timestamps - time_ns)))


def read_video_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {index} from {path}")
        return np.ascontiguousarray(bgr[..., ::-1])
    finally:
        capture.release()


def closedness(row: dict, max_width: float) -> float:
    width = row.get("gripper_width")
    if width is None:
        width = row.get("gripper_command_width", max_width)
    return float(np.clip(1.0 - float(width) / max_width, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", type=Path, default=RECORDINGS_DIR / "droid_raw_full_v3")
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--host", default="10.6.38.133")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--samples_per_phase", type=int, default=5)
    parser.add_argument("--gripper_max_width", type=float, default=0.08)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        help="Replay only this 1-based phase number; may be supplied repeatedly",
    )
    parser.add_argument(
        "--swap_cameras",
        action="store_true",
        help="Send the recorded wrist view as base and exterior view as wrist",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.horizon < 1 or args.samples_per_phase < 1:
        raise SystemExit("--horizon and --samples_per_phase must be positive")

    episode_dir = args.record_dir / args.episode
    states = load_jsonl(episode_dir / "robot_state.jsonl")
    commands = load_jsonl(episode_dir / "actions.jsonl")
    if len(states) <= args.horizon:
        raise RuntimeError("Episode is shorter than the requested horizon")

    camera_records = {
        EXTERIOR_CAMERA: load_jsonl(episode_dir / f"{EXTERIOR_CAMERA}_timestamps.jsonl"),
        WRIST_CAMERA: load_jsonl(episode_dir / f"{WRIST_CAMERA}_timestamps.jsonl"),
    }
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    output_suffix = "_swapped" if args.swap_cameras else ""
    output_path = args.output or SHADOW_LOGS_DIR / (
        f"replay_{args.episode}{output_suffix}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phases = [
        {
            "time_ns": states[0]["time_ns"],
            "action": "episode_start",
            "payload": {},
        },
        *commands,
    ]
    if args.phase:
        invalid_phases = [number for number in args.phase if not 1 <= number <= len(phases)]
        if invalid_phases:
            raise SystemExit(
                f"--phase values must be between 1 and {len(phases)}: {invalid_phases}"
            )
        selected_phase_indices = {number - 1 for number in args.phase}
        phases_to_run = [
            (index, phase)
            for index, phase in enumerate(phases)
            if index in selected_phase_indices
        ]
    else:
        phases_to_run = list(enumerate(phases))
    print("[REPLAY] Offline only: this script does not connect to the robot.")
    print(
        f"[REPLAY] episode={args.episode} phases={len(phases_to_run)}/{len(phases)} "
        f"samples/phase={args.samples_per_phase} swap_cameras={args.swap_cameras} "
        f"server={args.host}:{args.port}"
    )

    with output_path.open("w") as output_file:
        for run_index, (phase_index, phase) in enumerate(phases_to_run):
            state_index = nearest_index(states, phase["time_ns"])
            future_index = min(state_index + args.horizon, len(states) - 1)
            state_row = states[state_index]
            tcp = np.asarray(state_row["ee_translation"], dtype=np.float32)
            quaternion = np.asarray(state_row["ee_quaternion"], dtype=np.float32)
            model_state = np.concatenate(
                [
                    tcp,
                    quaternion,
                    np.asarray(
                        [closedness(state_row, args.gripper_max_width)],
                        dtype=np.float32,
                    ),
                ]
            )

            images = {}
            camera_indices = {}
            for camera_name, timestamps in camera_records.items():
                camera_index = nearest_index(timestamps, state_row["time_ns"])
                camera_indices[camera_name] = camera_index
                rgb = read_video_frame(
                    episode_dir / f"{camera_name}_rgb.mp4", camera_index
                )
                images[camera_name] = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(rgb, 224, 224)
                )

            observation = {
                "observation/state": model_state,
                "observation/image": images[
                    WRIST_CAMERA if args.swap_cameras else EXTERIOR_CAMERA
                ],
                "observation/wrist_image": images[
                    EXTERIOR_CAMERA if args.swap_cameras else WRIST_CAMERA
                ],
                "prompt": args.prompt,
            }
            predictions = []
            latencies = []
            for _ in range(args.samples_per_phase):
                started = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latencies.append(time.perf_counter() - started)
                if actions.shape != (args.horizon, 7):
                    raise RuntimeError(
                        f"Expected ({args.horizon}, 7), received {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy returned NaN or infinite actions")
                predictions.append(actions)

            predictions_array = np.asarray(predictions)
            predicted_displacements = predictions_array[:, :, :3].sum(axis=1)
            actual_displacement = (
                np.asarray(states[future_index]["ee_translation"], dtype=float) - tcp
            )
            predicted_mean = predicted_displacements.mean(axis=0)
            errors = np.linalg.norm(
                predicted_displacements - actual_displacement[None, :], axis=1
            )
            gripper_maxima = predictions_array[:, :, 6].max(axis=1)
            label = phase["payload"].get("label", phase["action"])
            record = {
                "phase_index": phase_index,
                "label": label,
                "state_index": state_index,
                "future_index": future_index,
                "camera_indices": camera_indices,
                "state": model_state.tolist(),
                "actual_displacement": actual_displacement.tolist(),
                "predicted_displacements": predicted_displacements.tolist(),
                "predicted_mean_displacement": predicted_mean.tolist(),
                "translation_errors_m": errors.tolist(),
                "gripper_maxima": gripper_maxima.tolist(),
                "latencies_s": latencies,
            }
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            print(
                f"[{run_index + 1:02d}/{len(phases_to_run)} phase={phase_index + 1:02d}] "
                f"{label} frame={state_index} "
                f"actual={np.array2string(actual_displacement, precision=3)} "
                f"pred_mean={np.array2string(predicted_mean, precision=3)} "
                f"error={errors.mean():.3f}+/-{errors.std():.3f}m "
                f"grip_max={gripper_maxima.mean():.3f}"
            )

    print(f"[REPLAY] complete; log: {output_path}")


if __name__ == "__main__":
    main()
