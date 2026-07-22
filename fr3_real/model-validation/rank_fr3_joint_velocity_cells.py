#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/rank_fr3_joint_velocity_cells.py --help
"""Rank recorded FR3 cells by joint-policy replay agreement without robot access.

The score measures how closely a checkpoint reproduces the demonstrated action
chunks at the phases required for pick-and-place. It is useful for choosing a
conservative cell to demonstrate, but it is not a physical success probability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import EVAL_LOGS_DIR, RECORDINGS_DIR


DEFAULT_PROMPT = "Pick up the blue cube and place it in the basket."
EXTERIOR_CAMERA = "camera_0_336222074819"
WRIST_CAMERA = "camera_1_243322074869"


@dataclass(frozen=True)
class Phase:
    name: str
    label_fragment: str
    kind: str
    weight: float
    last_match: bool = False


PHASES = (
    Phase("approach", "cell pickup: to hover: horizontal", "velocity", 1.5),
    Phase("grasp_descent", "cell pickup: descend to grasp", "velocity", 1.5),
    Phase("grasp", "gripper_grasp", "grasp", 1.0),
    Phase("lift", "cell pickup: lift & verify", "velocity", 1.0),
    Phase("transport", "place in basket: to hover: horizontal", "velocity", 1.5),
    Phase("place_descent", "place in basket: descend to place", "velocity", 1.5),
    Phase("release", "gripper_open", "release", 1.0, last_match=True),
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def nearest_index(times: np.ndarray, time_ns: int) -> int:
    index = int(np.searchsorted(times, time_ns))
    index = int(np.clip(index, 1, len(times) - 1))
    return index - 1 if abs(time_ns - times[index - 1]) <= abs(times[index] - time_ns) else index


def read_frame(capture: cv2.VideoCapture, index: int, name: str) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode {name} frame {index}")
    return np.ascontiguousarray(bgr[..., ::-1])


def gripper_closedness(sample: dict, max_width: float) -> float:
    return float(np.clip(1.0 - float(sample.get("gripper_width", max_width)) / max_width, 0.0, 1.0))


def gripper_command_closedness(sample: dict, max_width: float) -> float:
    width = sample.get("gripper_command_width", sample.get("gripper_width", max_width))
    return float(np.clip(1.0 - float(width) / max_width, 0.0, 1.0))


def action_vector(state: dict, next_state: dict, max_width: float) -> np.ndarray:
    return np.asarray(
        list(state.get("dq_d", state["dq"])) + [gripper_command_closedness(next_state, max_width)],
        dtype=np.float32,
    )


def phase_event(commands: list[dict], phase: Phase) -> dict | None:
    matches = [
        command
        for command in commands
        if phase.label_fragment in command.get("payload", {}).get("label", command.get("action", ""))
        or phase.label_fragment == command.get("action")
    ]
    if not matches:
        return None
    return matches[-1] if phase.last_match else matches[0]


def velocity_score(predicted: np.ndarray, actual: np.ndarray, fps: float) -> tuple[float, dict]:
    predicted_delta = predicted[:, :7].sum(axis=0) / fps
    actual_delta = actual[:, :7].sum(axis=0) / fps
    predicted_norm = float(np.linalg.norm(predicted_delta))
    actual_norm = float(np.linalg.norm(actual_delta))
    if predicted_norm < 1e-9 or actual_norm < 1e-9:
        cosine = 1.0 if predicted_norm < 1e-5 and actual_norm < 1e-5 else 0.0
        magnitude_ratio = 1.0 if cosine == 1.0 else 0.0
    else:
        cosine = float(np.clip(np.dot(predicted_delta, actual_delta) / (predicted_norm * actual_norm), -1.0, 1.0))
        magnitude_ratio = min(predicted_norm / actual_norm, actual_norm / predicted_norm)
    rmse = float(np.sqrt(np.mean((predicted[:, :7] - actual[:, :7]) ** 2)))
    action_rms = float(np.sqrt(np.mean(actual[:, :7] ** 2)))
    rmse_score = max(0.0, 1.0 - rmse / max(action_rms, 0.05))
    score = 0.45 * ((cosine + 1.0) / 2.0) + 0.35 * magnitude_ratio + 0.20 * rmse_score
    return score, {
        "score": score,
        "cosine": cosine,
        "magnitude_ratio": magnitude_ratio,
        "velocity_rmse": rmse,
        "predicted_joint_delta": predicted_delta.tolist(),
        "actual_joint_delta": actual_delta.tolist(),
    }


def gripper_score(predicted: np.ndarray, target: float) -> tuple[float, dict]:
    value = float(predicted[:, 7].max()) if target > 0 else float(predicted[:, 7].mean())
    score = max(0.0, 1.0 - abs(value - target) / 0.5)
    return score, {"score": score, "prediction": value, "target": target}


def successful_episodes(record_dir: Path) -> list[Path]:
    episodes = []
    for episode in sorted(record_dir.iterdir()):
        metadata_path = episode / "metadata.json"
        if not metadata_path.exists():
            continue
        if json.loads(metadata_path.read_text()).get("success", False):
            episodes.append(episode)
    return episodes


def rank_episode(
    episode: Path,
    client: websocket_client_policy.WebsocketClientPolicy,
    horizon: int,
    fps: float,
    samples_per_phase: int,
    gripper_max_width: float,
    prompt: str,
) -> dict:
    metadata = json.loads((episode / "metadata.json").read_text())
    states = load_jsonl(episode / "robot_state.jsonl")
    commands = load_jsonl(episode / "actions.jsonl")
    exterior_timestamps = load_jsonl(episode / f"{EXTERIOR_CAMERA}_timestamps.jsonl")
    wrist_timestamps = load_jsonl(episode / f"{WRIST_CAMERA}_timestamps.jsonl")
    source_fps = int(metadata.get("fps", 60))
    stride = max(1, round(source_fps / fps))
    selected_frames = list(range(0, min(len(exterior_timestamps), len(wrist_timestamps)), stride))
    state_times = np.asarray([state["time_ns"] for state in states], dtype=np.int64)
    selected_times = np.asarray(
        [exterior_timestamps[index]["time_ns"] for index in selected_frames], dtype=np.int64
    )
    selected_state_indices = np.asarray(
        [nearest_index(state_times, int(time_ns)) for time_ns in selected_times], dtype=np.int64
    )

    exterior_capture = cv2.VideoCapture(str(episode / f"{EXTERIOR_CAMERA}_rgb.mp4"))
    wrist_capture = cv2.VideoCapture(str(episode / f"{WRIST_CAMERA}_rgb.mp4"))
    if not exterior_capture.isOpened() or not wrist_capture.isOpened():
        raise RuntimeError("Could not open one or both episode videos")

    details = []
    try:
        for phase in PHASES:
            event = phase_event(commands, phase)
            if event is None:
                raise RuntimeError(f"Missing required phase: {phase.name}")
            dataset_index = nearest_index(selected_times, int(event["time_ns"]))
            if dataset_index + horizon >= len(selected_frames):
                raise RuntimeError(f"Not enough frames after phase: {phase.name}")
            state_index = int(selected_state_indices[dataset_index])
            state = states[state_index]
            exterior_frame = selected_frames[dataset_index]
            wrist_frame = nearest_index(
                np.asarray([row["time_ns"] for row in wrist_timestamps], dtype=np.int64),
                int(selected_times[dataset_index]),
            )
            exterior_rgb = read_frame(exterior_capture, exterior_frame, "exterior")
            wrist_rgb = read_frame(wrist_capture, wrist_frame, "wrist")
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
                    [gripper_closedness(state, gripper_max_width)], dtype=np.float32
                ),
                "prompt": prompt,
            }
            actual = np.asarray(
                [
                    action_vector(
                        states[int(selected_state_indices[dataset_index + step])],
                        states[
                            min(int(selected_state_indices[dataset_index + step]) + 1, len(states) - 1)
                        ],
                        gripper_max_width,
                    )
                    for step in range(horizon)
                ],
                dtype=float,
            )
            samples = []
            for _ in range(samples_per_phase):
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                if actions.shape != (horizon, 8) or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid policy actions at {phase.name}: {actions.shape}")
                samples.append(actions)
            predicted = np.mean(np.asarray(samples), axis=0)
            if phase.kind == "velocity":
                phase_score, metrics = velocity_score(predicted, actual, fps)
            elif phase.kind == "grasp":
                phase_score, metrics = gripper_score(predicted, target=0.5)
            else:
                phase_score, metrics = gripper_score(predicted, target=0.0)
            details.append({"phase": phase.name, "weight": phase.weight, **metrics})
    finally:
        exterior_capture.release()
        wrist_capture.release()

    total_weight = sum(detail["weight"] for detail in details)
    score = sum(detail["weight"] * detail["score"] for detail in details) / total_weight
    cell = metadata.get("source", {})
    return {
        "episode": episode.name,
        "printed_cell": cell.get("printed_cell"),
        "cell_xyz": metadata.get("cell_xyz"),
        "score": score,
        "phase_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", type=Path, default=RECORDINGS_DIR / "droid_raw_full_v3")
    parser.add_argument("--host", default="10.6.38.133")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--samples_per_phase", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gripper_max_width", type=float, default=0.08)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=EVAL_LOGS_DIR / "joint_velocity_cell_ranking.json")
    args = parser.parse_args()

    if args.horizon < 1 or args.samples_per_phase < 1 or args.top_k < 1:
        raise SystemExit("--horizon, --samples_per_phase, and --top_k must be positive")

    episodes = successful_episodes(args.record_dir)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"No successful episodes under {args.record_dir}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("[RANK] Offline replay only: this script never connects to or moves the robot.")
    print(
        f"[RANK] episodes={len(episodes)} phases={len(PHASES)} "
        f"samples/phase={args.samples_per_phase} server={args.host}:{args.port}"
    )

    results = []
    failures = []
    started = time.perf_counter()
    for index, episode in enumerate(episodes, start=1):
        try:
            result = rank_episode(
                episode,
                client,
                args.horizon,
                args.fps,
                args.samples_per_phase,
                args.gripper_max_width,
                args.prompt,
            )
            results.append(result)
            print(
                f"[{index:03d}/{len(episodes)}] cell={result['printed_cell']} "
                f"score={result['score']:.3f} episode={result['episode']}"
            )
        except Exception as exc:
            failures.append({"episode": episode.name, "error": str(exc)})
            print(f"[{index:03d}/{len(episodes)}] SKIP {episode.name}: {exc}")

    results.sort(key=lambda result: result["score"], reverse=True)
    payload = {
        "record_dir": str(args.record_dir),
        "server": f"{args.host}:{args.port}",
        "episodes_scored": len(results),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "ranking": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n[RANK] Best replay-agreement cells:")
    for rank, result in enumerate(results[: args.top_k], start=1):
        xyz = result["cell_xyz"] or [float("nan")] * 3
        print(
            f"  {rank:02d}. cell={result['printed_cell']} score={result['score']:.3f} "
            f"xyz=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) "
            f"episode={result['episode']}"
        )
    print(f"[RANK] detailed ranking: {args.output}")


if __name__ == "__main__":
    main()
