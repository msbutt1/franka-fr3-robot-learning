#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/streaming/test_stream_protocol.py --help
"""Exercise timestamped activation and STOP against a dry-run streamer."""

from __future__ import annotations

import argparse
import socket
import time

import numpy as np

from stream_protocol import VERSION, make_command, receive_latest_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=51020)
    parser.add_argument("--state-port", type=int, default=51021)
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM
    ) as state_sock:
        state_sock.bind(("127.0.0.1", args.state_port))
        state_sock.settimeout(2.0)
        state, _ = receive_latest_state(state_sock)
        print(f"received protocol-v{VERSION} state timestamp={state.monotonic_ns}")

        velocities = np.zeros((8, 7), dtype=np.float64)
        velocities[:, [0, 1, 3, 5, 6]] = [0.01, -0.01, 0.02, -0.02, 0.01]
        # A 150 ms-old observation should activate around action index 2 at 15 Hz.
        observation_ns = time.monotonic_ns() - 150_000_000
        sock.sendto(make_command(1, observation_ns, velocities), ("127.0.0.1", args.port))
        activated, _ = receive_latest_state(state_sock)
        while activated.active_sequence != 1 and not activated.stop_reason:
            activated, _ = receive_latest_state(state_sock)
        if activated.activation_offset not in (2, 3):
            raise RuntimeError(f"unexpected activation offset {activated.activation_offset}")
        if activated.plan_steps != len(velocities):
            raise RuntimeError(f"unexpected accepted plan length {activated.plan_steps}")
        print(
            f"activated sequence=1 offset={activated.activation_offset} "
            f"step={activated.plan_step}"
        )

        sock.sendto(
            make_command(2, time.monotonic_ns(), None), ("127.0.0.1", args.port)
        )
        stopped, _ = receive_latest_state(state_sock)
        while stopped.stop_reason == 0:
            stopped, _ = receive_latest_state(state_sock)
        if stopped.stop_reason_name != "command":
            raise RuntimeError(f"unexpected stop reason {stopped.stop_reason_name}")
        print("STOP acknowledged reason=command")


if __name__ == "__main__":
    main()
