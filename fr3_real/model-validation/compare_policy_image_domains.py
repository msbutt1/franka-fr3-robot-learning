#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/compare_policy_image_domains.py --help
"""Compare policy actions for recorded and live images at one fixed FR3 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(image)


def read_video_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame} from {path}")
        return np.ascontiguousarray(bgr[..., ::-1])
    finally:
        capture.release()


def policy_observation(exterior: np.ndarray, wrist: np.ndarray, q: np.ndarray) -> dict:
    return {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(exterior, 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, 224, 224)
        ),
        "observation/joint_position": q,
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "Pick up the blue cube and place it in the basket.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline only: compare recorded and saved-live image policy outputs."
    )
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--live_shadow", type=Path, required=True,
                        help="JSONL shadow log; matching _exterior.png and _wrist.png are used")
    parser.add_argument("--recorded_frame", type=int, default=0)
    parser.add_argument("--host", default="10.6.38.133")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    rows = [json.loads(line) for line in args.live_shadow.read_text().splitlines() if line]
    if not rows:
        raise RuntimeError(f"No observations in {args.live_shadow}")
    q = np.asarray(rows[0]["q"], dtype=np.float32)
    live_exterior = load_rgb(args.live_shadow.with_name(args.live_shadow.stem + "_exterior.png"))
    live_wrist = load_rgb(args.live_shadow.with_name(args.live_shadow.stem + "_wrist.png"))
    recorded_exterior = read_video_frame(
        args.episode_dir / "camera_0_336222074819_rgb.mp4", args.recorded_frame
    )
    recorded_wrist = read_video_frame(
        args.episode_dir / "camera_1_243322074869_rgb.mp4", args.recorded_frame
    )

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    observations = {
        "recorded": policy_observation(recorded_exterior, recorded_wrist, q),
        "live": policy_observation(live_exterior, live_wrist, q),
    }
    actions: dict[str, np.ndarray] = {}
    print("[IMAGE A/B] Offline only: no robot or gripper commands are issued.")
    print(f"[IMAGE A/B] fixed q={np.array2string(q, precision=4)}")
    for name, observation in observations.items():
        samples = np.asarray([client.infer(observation)["actions"] for _ in range(args.samples)], dtype=float)
        if samples.shape[1:] != (16, 8) or not np.isfinite(samples).all():
            raise RuntimeError(f"{name}: invalid policy output {samples.shape}")
        actions[name] = samples.mean(axis=0)
        print(
            f"[IMAGE A/B] {name:8s} peak={np.abs(actions[name][:, :7]).max():.3f} "
            f"chunk_dq={np.array2string(actions[name][:, :7].sum(axis=0) / 15.0, precision=4)} "
            f"grip_max={actions[name][:, 7].max():.3f}"
        )
    difference = actions["live"] - actions["recorded"]
    print(
        f"[IMAGE A/B] live-vs-recorded mean|dq|={np.abs(difference[:, :7]).mean():.4f} "
        f"max|dq|={np.abs(difference[:, :7]).max():.4f} "
        f"chunk_dq_delta={np.array2string(difference[:, :7].sum(axis=0) / 15.0, precision=4)}"
    )


if __name__ == "__main__":
    main()
