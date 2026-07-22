#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/test_fr3_cartesian_policy_server.py --help
"""Synthetic smoke test for the isolated 50-step FR3 Cartesian policy."""

from __future__ import annotations

import argparse

import numpy as np
from openpi_client import websocket_client_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Representative state from the recorded FR3 workspace. Quaternion order
    # intentionally matches the persisted training vector.
    state = np.asarray(
        [0.6211, -0.0696, 0.2878, 0.9957, -0.0166, -0.0880, 0.0225, 0.0],
        dtype=np.float32,
    )
    observation = {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": image.copy(),
        "prompt": "Pick up the cube from the cell and place it in the basket.",
    }
    actions = np.asarray(client.infer(observation)["actions"])

    print(f"server: {args.host}:{args.port}")
    print("shape:", actions.shape)
    print("finite:", bool(np.isfinite(actions).all()))
    print("translation range:", float(actions[:, :3].min()), float(actions[:, :3].max()))
    print("rotation range:", float(actions[:, 3:6].min()), float(actions[:, 3:6].max()))
    print("gripper range:", float(actions[:, 6].min()), float(actions[:, 6].max()))

    assert actions.shape == (50, 7), actions.shape
    assert np.isfinite(actions).all()
    print("CARTESIAN POLICY SERVER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
