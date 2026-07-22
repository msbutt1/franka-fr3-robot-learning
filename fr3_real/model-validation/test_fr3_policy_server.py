#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/test_fr3_policy_server.py --help
"""Smoke-test an OpenPI FR3 policy server without commanding the robot."""

from __future__ import annotations

import argparse

import numpy as np
from openpi_client import websocket_client_policy


DEFAULT_PROMPT = "Pick up the blue cube and place it in the basket."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image.copy(),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": args.prompt,
    }

    actions = np.asarray(client.infer(observation)["actions"])
    finite = bool(np.isfinite(actions).all())

    print(f"server: {args.host}:{args.port}")
    print(f"shape: {actions.shape}")
    print(f"finite: {finite}")
    print(f"range: [{float(actions.min()):.6f}, {float(actions.max()):.6f}]")
    print(np.array2string(actions, precision=5, suppress_small=True))

    if actions.shape != (16, 8):
        raise RuntimeError(f"Expected action shape (16, 8), received {actions.shape}")
    if not finite:
        raise RuntimeError("Policy returned NaN or infinite actions")

    print("POLICY SERVER SMOKE TEST PASSED")
    print("These synthetic-input actions must not be sent to the robot.")


if __name__ == "__main__":
    main()
