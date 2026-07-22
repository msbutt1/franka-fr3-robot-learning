#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/shadow_fr3_cartesian_policy.py --help
"""Query the 50-step FR3 Cartesian policy with live data without commanding motion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import SHADOW_LOGS_DIR


DEFAULT_PROMPT = "Pick up the cube from the cell and place it in the basket."
DEFAULT_EXTERIOR_SERIAL = "336222074819"
DEFAULT_WRIST_SERIAL = "243322074869"

# Quantiles embedded in pi05_fr3_checkpoint_29999/assets/training_dataset.
# The quaternion entries intentionally retain the order persisted during recording.
STATE_Q01 = np.asarray(
    [
        0.2611988689,
        -0.3907777965,
        0.0254766578,
        0.9919339890,
        -0.0447265347,
        -0.1236408460,
        -0.0236538497,
        0.0018830635,
    ],
    dtype=np.float32,
)
STATE_Q99 = np.asarray(
    [
        0.6554354029,
        0.3674626052,
        0.2964669579,
        0.9993208519,
        0.0359131887,
        -0.0329297329,
        0.0211130113,
        0.9956003428,
    ],
    dtype=np.float32,
)


def start_camera(serial: str, width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def read_rgb(pipeline, timeout_ms: int = 2000) -> np.ndarray:
    frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("RealSense did not return a color frame")
    bgr = np.asanyarray(color.get_data())
    return np.ascontiguousarray(bgr[..., ::-1])


def vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--server_host", default="10.6.38.133")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--exterior_serial", default=DEFAULT_EXTERIOR_SERIAL)
    parser.add_argument("--wrist_serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gripper_max_width", type=float, default=0.08)
    parser.add_argument("--log_dir", type=Path, default=SHADOW_LOGS_DIR)
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.gripper_max_width <= 0:
        raise SystemExit("--gripper_max_width must be positive")

    print("[CARTESIAN SHADOW] Read-only: no robot or gripper commands are issued.")
    print(f"[CARTESIAN SHADOW] server={args.server_host}:{args.server_port}")
    print(
        f"[CARTESIAN SHADOW] exterior={args.exterior_serial} "
        f"wrist={args.wrist_serial} "
        f"camera={args.camera_width}x{args.camera_height}@{args.camera_fps}"
    )
    input("Confirm the workspace is clear, then press Enter to connect read-only...")

    from franky import Gripper, Robot

    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)
    client = websocket_client_policy.WebsocketClientPolicy(
        args.server_host, args.server_port
    )
    exterior = None
    wrist = None

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / (
        f"cartesian_shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    try:
        exterior = start_camera(
            args.exterior_serial,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
        )
        wrist = start_camera(
            args.wrist_serial,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
        )

        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)

        previous_actions = None
        with log_path.open("w") as log_file:
            for index in range(args.iterations):
                exterior_rgb = read_rgb(exterior)
                wrist_rgb = read_rgb(wrist)
                if index == 0:
                    Image.fromarray(exterior_rgb).save(
                        log_path.with_name(f"{log_path.stem}_exterior.png")
                    )
                    Image.fromarray(wrist_rgb).save(
                        log_path.with_name(f"{log_path.stem}_wrist.png")
                    )

                robot_state = robot.state
                pose = robot_state.O_T_EE
                tcp = vector(pose.translation)
                quaternion = vector(pose.quaternion)
                if tcp.shape != (3,) or quaternion.shape != (4,):
                    raise RuntimeError(
                        f"Unexpected pose shapes: translation={tcp.shape}, "
                        f"quaternion={quaternion.shape}"
                    )

                gripper_width = float(gripper.state.width)
                gripper_closedness = float(
                    np.clip(1.0 - gripper_width / args.gripper_max_width, 0.0, 1.0)
                )
                state = np.concatenate(
                    [tcp, quaternion, np.asarray([gripper_closedness], dtype=np.float32)]
                ).astype(np.float32, copy=False)
                outside_quantiles = np.flatnonzero(
                    (state < STATE_Q01) | (state > STATE_Q99)
                ).tolist()

                observation = {
                    "observation/state": state,
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior_rgb, 224, 224)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_rgb, 224, 224)
                    ),
                    "prompt": args.prompt,
                }

                started = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latency = time.perf_counter() - started
                if actions.shape != (50, 7):
                    raise RuntimeError(
                        f"Expected action shape (50, 7), received {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy returned NaN or infinite actions")

                translation_deltas = actions[:, :3]
                rotation_deltas = actions[:, 3:6]
                predicted_path = tcp[None, :] + np.cumsum(translation_deltas, axis=0)
                translation_step_norms = np.linalg.norm(translation_deltas, axis=1)
                rotation_step_norms = np.linalg.norm(rotation_deltas, axis=1)
                first_step_jump = None
                if previous_actions is not None:
                    first_step_jump = float(
                        np.max(np.abs(actions[0, :6] - previous_actions[0, :6]))
                    )

                record = {
                    "time_ns": time.time_ns(),
                    "iteration": index,
                    "latency_s": latency,
                    "state": state.tolist(),
                    "tcp": tcp.tolist(),
                    "quaternion_recorded_order": quaternion.tolist(),
                    "gripper_width": gripper_width,
                    "gripper_closedness": gripper_closedness,
                    "state_outside_q01_q99_dimensions": outside_quantiles,
                    "actions": actions.tolist(),
                    "predicted_xyz_end": predicted_path[-1].tolist(),
                    "predicted_xyz_min": predicted_path.min(axis=0).tolist(),
                    "predicted_xyz_max": predicted_path.max(axis=0).tolist(),
                    "max_translation_step_m": float(translation_step_norms.max()),
                    "max_rotation_step": float(rotation_step_norms.max()),
                    "first_step_jump": first_step_jump,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

                print(
                    f"[{index + 1:02d}/{args.iterations}] latency={latency:.3f}s "
                    f"tcp={np.array2string(tcp, precision=3)} "
                    f"end={np.array2string(predicted_path[-1], precision=3)}"
                )
                print(
                    f"             max|dxyz|={translation_step_norms.max():.6f}m "
                    f"max|drot|={rotation_step_norms.max():.6f} "
                    f"gripper={actions[:, 6].min():.3f}..{actions[:, 6].max():.3f} "
                    f"jump={first_step_jump if first_step_jump is not None else float('nan'):.6f} "
                    f"outside_q01_q99={outside_quantiles}"
                )
                previous_actions = actions
    finally:
        if exterior is not None:
            exterior.stop()
        if wrist is not None:
            wrist.stop()

    print(
        "[CARTESIAN SHADOW] completed without commanding motion; "
        f"log: {log_path}"
    )


if __name__ == "__main__":
    main()
