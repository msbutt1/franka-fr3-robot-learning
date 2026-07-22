#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/replay_fr3_joint_velocity_policy.py --help
"""Replay recorded FR3 observations through the 15 Hz joint-velocity policy."""

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
DEFAULT_PROMPT = "Pick up the blue cube and place it in the basket."
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


def gripper_closedness(sample: dict, max_width: float) -> float:
    width = float(sample.get("gripper_width", max_width))
    return float(np.clip(1.0 - width / max_width, 0.0, 1.0))


def gripper_command_closedness(sample: dict, max_width: float) -> float:
    width = float(sample.get("gripper_command_width", sample.get("gripper_width", max_width)))
    return float(np.clip(1.0 - width / max_width, 0.0, 1.0))


def action_vector(state: dict, next_state: dict, max_width: float) -> np.ndarray:
    return np.asarray(
        list(state.get("dq_d", state["dq"]))
        + [gripper_command_closedness(next_state, max_width)],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", type=Path, default=RECORDINGS_DIR / "droid_raw_full_v3")
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--host", default="10.6.38.133")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--samples_per_phase", type=int, default=5)
    parser.add_argument("--gripper_max_width", type=float, default=0.08)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        help="Replay only this 1-based phase number; may be supplied repeatedly",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.horizon < 1 or args.samples_per_phase < 1 or args.fps <= 0:
        raise SystemExit("--horizon, --samples_per_phase, and --fps must be positive")

    episode_dir = args.record_dir / args.episode
    states = load_jsonl(episode_dir / "robot_state.jsonl")
    commands = load_jsonl(episode_dir / "actions.jsonl")
    exterior_timestamps = load_jsonl(episode_dir / f"{EXTERIOR_CAMERA}_timestamps.jsonl")
    wrist_timestamps = load_jsonl(episode_dir / f"{WRIST_CAMERA}_timestamps.jsonl")
    source_fps = int(json.loads((episode_dir / "metadata.json").read_text()).get("fps", 60))
    stride = max(1, round(source_fps / args.fps))
    selected_frames = list(range(0, min(len(exterior_timestamps), len(wrist_timestamps)), stride))
    state_times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)
    selected_times = np.asarray(
        [exterior_timestamps[index]["time_ns"] for index in selected_frames], dtype=np.int64
    )
    selected_state_indices = np.searchsorted(state_times, selected_times)
    selected_state_indices = np.clip(selected_state_indices, 1, len(states) - 1)
    left = selected_state_indices - 1
    selected_state_indices = np.where(
        np.abs(selected_times - state_times[left])
        <= np.abs(selected_times - state_times[selected_state_indices]),
        left,
        selected_state_indices,
    )

    phases = [
        {"time_ns": states[0]["time_ns"], "action": "episode_start", "payload": {}},
        *commands,
    ]
    if args.phase:
        invalid_phases = [number for number in args.phase if not 1 <= number <= len(phases)]
        if invalid_phases:
            raise SystemExit(
                f"--phase values must be between 1 and {len(phases)}: {invalid_phases}"
            )
        phases_to_run = [
            (index, phase)
            for index, phase in enumerate(phases)
            if index in {number - 1 for number in args.phase}
        ]
    else:
        phases_to_run = list(enumerate(phases))

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    output_path = args.output or SHADOW_LOGS_DIR / f"joint_velocity_replay_{args.episode}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[JOINT REPLAY] Offline only: this script does not connect to the robot.")
    print(
        f"[JOINT REPLAY] episode={args.episode} phases={len(phases_to_run)}/{len(phases)} "
        f"horizon={args.horizon}@{args.fps:g}Hz samples/phase={args.samples_per_phase}"
    )

    with output_path.open("w") as output_file:
        for run_index, (phase_index, phase) in enumerate(phases_to_run):
            dataset_index = int(np.argmin(np.abs(selected_times - phase["time_ns"])))
            if dataset_index + args.horizon >= len(selected_frames):
                continue
            state_index = int(selected_state_indices[dataset_index])
            state = states[state_index]
            exterior_frame = selected_frames[dataset_index]
            wrist_frame = nearest_index(wrist_timestamps, selected_times[dataset_index])
            exterior_rgb = read_video_frame(
                episode_dir / f"{EXTERIOR_CAMERA}_rgb.mp4", exterior_frame
            )
            wrist_rgb = read_video_frame(
                episode_dir / f"{WRIST_CAMERA}_rgb.mp4", wrist_frame
            )
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(exterior_rgb, 224, 224)
                ),
                "observation/exterior_image_2_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(exterior_rgb, 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_rgb, 224, 224)
                ),
                "observation/joint_position": np.asarray(state["q"], dtype=np.float32),
                "observation/gripper_position": np.asarray(
                    [gripper_closedness(state, args.gripper_max_width)], dtype=np.float32
                ),
                "prompt": args.prompt,
            }
            actual_actions = np.asarray(
                [
                    action_vector(
                        states[int(selected_state_indices[dataset_index + step])],
                        states[
                            min(
                                int(selected_state_indices[dataset_index + step]) + 1,
                                len(states) - 1,
                            )
                        ],
                        args.gripper_max_width,
                    )
                    for step in range(args.horizon)
                ],
                dtype=float,
            )
            predictions = []
            latencies = []
            for _ in range(args.samples_per_phase):
                started = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latencies.append(time.perf_counter() - started)
                if actions.shape != (args.horizon, 8):
                    raise RuntimeError(
                        f"Expected ({args.horizon}, 8), received {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy returned NaN or infinite actions")
                predictions.append(actions)

            predictions_array = np.asarray(predictions)
            velocity_rmse = np.sqrt(
                np.mean((predictions_array[:, :, :7] - actual_actions[None, :, :7]) ** 2, axis=(1, 2))
            )
            predicted_joint_delta = predictions_array[:, :, :7].sum(axis=1) / args.fps
            end_state = states[int(selected_state_indices[dataset_index + args.horizon])]
            actual_joint_delta = np.asarray(end_state["q"], dtype=float) - np.asarray(state["q"], dtype=float)
            label = phase["payload"].get("label", phase["action"])
            record = {
                "phase_index": phase_index,
                "label": label,
                "dataset_index": dataset_index,
                "state_index": state_index,
                "source_frame_indices": {"exterior": exterior_frame, "wrist": wrist_frame},
                "actual_actions": actual_actions.tolist(),
                "actual_joint_delta": actual_joint_delta.tolist(),
                "predicted_joint_deltas": predicted_joint_delta.tolist(),
                "velocity_rmse": velocity_rmse.tolist(),
                "predicted_gripper_maxima": predictions_array[:, :, 7].max(axis=1).tolist(),
                "latencies_s": latencies,
            }
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            print(
                f"[{run_index + 1:02d}/{len(phases_to_run)} phase={phase_index + 1:02d}] "
                f"{label} sample={dataset_index} "
                f"actual_dq={np.array2string(actual_joint_delta, precision=3)} "
                f"pred_dq={np.array2string(predicted_joint_delta.mean(axis=0), precision=3)} "
                f"vel_rmse={velocity_rmse.mean():.3f} "
                f"grip_max={predictions_array[:, :, 7].max(axis=1).mean():.3f}"
            )

    print(f"[JOINT REPLAY] complete; log: {output_path}")


if __name__ == "__main__":
    main()
