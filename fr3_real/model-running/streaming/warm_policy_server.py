#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/streaming/warm_policy_server.py --help
"""Warm and validate the live FR3 policy server before FCI control starts."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_paths import EVAL_LOGS_DIR
from shadow_fr3_policy import (
    DEFAULT_EXTERIOR_SERIAL,
    DEFAULT_PROMPT,
    DEFAULT_WRIST_SERIAL,
    read_rgb,
    start_camera,
    vector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--server-host", default="10.6.38.133")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--exterior-serial", default=DEFAULT_EXTERIOR_SERIAL)
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-steady-latency", type=float, default=0.75)
    parser.add_argument("--gripper-max-width", type=float, default=0.08)
    parser.add_argument("--log-dir", type=Path, default=EVAL_LOGS_DIR / "policy_warmups")
    args = parser.parse_args()
    if args.iterations < 2 or args.max_steady_latency <= 0:
        raise SystemExit("Use at least two iterations and a positive steady-latency limit")

    from franky import Gripper, Robot

    run_dir = args.log_dir / (
        f"{args.checkpoint_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "warmup.jsonl"
    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    metadata = client.get_server_metadata()
    exterior = start_camera(
        args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps
    )
    wrist = start_camera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps)
    failures: list[float] = []

    print("[WARMUP] Read-only policy warmup; no arm or gripper motion is commanded.")
    print(f"[WARMUP] expected_checkpoint={args.checkpoint_label} metadata={metadata!r}")
    metadata_text = json.dumps(metadata, default=str)
    if "checkpoint" in metadata_text.lower() and args.checkpoint_label not in metadata_text:
        raise RuntimeError(
            f"Server checkpoint metadata does not match {args.checkpoint_label!r}: {metadata!r}"
        )
    if "checkpoint" not in metadata_text.lower():
        print("[WARMUP] WARNING server does not advertise checkpoint identity; label remains operator-supplied.")

    try:
        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)
        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "server_metadata",
                        "checkpoint_label": args.checkpoint_label,
                        "server": f"{args.server_host}:{args.server_port}",
                        "metadata": metadata,
                    },
                    default=str,
                )
                + "\n"
            )
            for index in range(args.iterations):
                exterior_rgb = read_rgb(exterior)
                wrist_rgb = read_rgb(wrist)
                exterior_path = run_dir / f"{index + 1:02d}_exterior.png"
                wrist_path = run_dir / f"{index + 1:02d}_wrist.png"
                Image.fromarray(exterior_rgb).save(exterior_path)
                Image.fromarray(wrist_rgb).save(wrist_path)
                q = vector(robot.state.q)
                width = float(gripper.state.width)
                closedness = float(np.clip(1.0 - width / args.gripper_max_width, 0.0, 1.0))
                observation = {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior_rgb, 224, 224)
                    ),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_rgb, 224, 224)
                    ),
                    "observation/joint_position": q,
                    "observation/gripper_position": np.asarray([closedness], dtype=np.float32),
                    "prompt": args.prompt,
                }
                started = time.perf_counter()
                response = client.infer(observation)
                latency = time.perf_counter() - started
                actions = np.asarray(response["actions"], dtype=float)
                if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= actions.shape[0] <= 16 or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid warmup response shape={actions.shape}")
                if index > 0 and latency > args.max_steady_latency:
                    failures.append(latency)
                record = {
                    "event": "warmup_inference",
                    "iteration": index + 1,
                    "latency_s": latency,
                    "q": q.tolist(),
                    "gripper_closedness": closedness,
                    "exterior_image": str(exterior_path),
                    "wrist_image": str(wrist_path),
                    "actions": actions.tolist(),
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(
                    f"[WARMUP {index + 1}/{args.iterations}] latency={latency:.3f}s "
                    f"peak={np.max(np.abs(actions[:, :7])):.3f}"
                )
    finally:
        exterior.stop()
        wrist.stop()

    if failures:
        raise RuntimeError(
            "Steady policy latency exceeded the pre-FCI limit: "
            + ", ".join(f"{latency:.3f}s" for latency in failures)
        )
    print(f"[WARMUP] ready; log={log_path}")


if __name__ == "__main__":
    main()
