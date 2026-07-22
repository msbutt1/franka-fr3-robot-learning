#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/shadow_fr3_policy.py --help
"""Query an OpenPI FR3 policy with live observations without moving the robot."""

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


DEFAULT_PROMPT = "Pick up the blue cube and place it in the basket."
DEFAULT_EXTERIOR_SERIAL = "336222074819"
DEFAULT_WRIST_SERIAL = "243322074869"


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

    print("[SHADOW] Read-only policy test: no robot or gripper motions are commanded.")
    print(f"[SHADOW] server={args.server_host}:{args.server_port}")
    print(
        f"[SHADOW] exterior={args.exterior_serial} wrist={args.wrist_serial} "
        f"camera={args.camera_width}x{args.camera_height}@{args.camera_fps}"
    )
    input("Confirm the workspace is clear, then press Enter to connect read-only...")

    from franky import Gripper, Robot

    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)
    client = websocket_client_policy.WebsocketClientPolicy(
        args.server_host, args.server_port
    )
    exterior = start_camera(
        args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps
    )
    wrist = start_camera(
        args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    try:
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
                state = robot.state
                q = vector(state.q)
                width = float(gripper.state.width)
                closedness = float(
                    np.clip(1.0 - width / args.gripper_max_width, 0.0, 1.0)
                )

                observation = {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior_rgb, 224, 224)
                    ),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_rgb, 224, 224)
                    ),
                    "observation/joint_position": q,
                    "observation/gripper_position": np.asarray(
                        [closedness], dtype=np.float32
                    ),
                    "prompt": args.prompt,
                }

                started = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latency = time.perf_counter() - started
                if actions.shape != (16, 8):
                    raise RuntimeError(
                        f"Expected action shape (16, 8), received {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy returned NaN or infinite actions")

                first_step_jump = None
                if previous_actions is not None:
                    first_step_jump = float(
                        np.max(np.abs(actions[0, :7] - previous_actions[0, :7]))
                    )
                record = {
                    "time_ns": time.time_ns(),
                    "iteration": index,
                    "latency_s": latency,
                    "q": q.tolist(),
                    "gripper_width": width,
                    "gripper_closedness": closedness,
                    "actions": actions.tolist(),
                    "max_abs_joint_velocity": float(np.max(np.abs(actions[:, :7]))),
                    "first_step_jump": first_step_jump,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(
                    f"[{index + 1:02d}/{args.iterations}] latency={latency:.3f}s "
                    f"max|dq|={record['max_abs_joint_velocity']:.3f} "
                    f"gripper={actions[:, 7].min():.3f}..{actions[:, 7].max():.3f} "
                    f"jump={first_step_jump if first_step_jump is not None else float('nan'):.3f}"
                )
                previous_actions = actions
    finally:
        exterior.stop()
        wrist.stop()

    print(f"[SHADOW] completed without commanding motion; log: {log_path}")


if __name__ == "__main__":
    main()
