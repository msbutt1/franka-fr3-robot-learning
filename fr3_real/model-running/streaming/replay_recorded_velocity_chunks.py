#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/streaming/replay_recorded_velocity_chunks.py --help
"""Replay a known recorded arm trajectory through the timestamped streamer.

This bypasses policy inference, cameras, and the gripper. Overlapping receding
windows replace the old stop-at-each-16-action playback. ``--through-grasp``
executes the full recorded approach through the arm action aligned with the
recorded gripper-grasp marker, but never closes the physical gripper.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_paths import RECORDINGS_DIR
from stream_protocol import MAX_STEPS, StreamState, make_command, receive_latest_state


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def grasp_aligned_step(sampled: list[dict], actions: list[dict]) -> int:
    markers = [row for row in actions if row.get("action") == "gripper_grasp"]
    if len(markers) != 1:
        raise RuntimeError(f"Expected one gripper_grasp marker, found {len(markers)}")
    marker_ns = int(markers[0]["time_ns"])
    for index, state in enumerate(sampled):
        if int(state["time_ns"]) >= marker_ns:
            return index
    raise RuntimeError("Grasp marker occurs after the sampled robot states")


def write_states(handle, states: list[StreamState]) -> None:
    if handle is None:
        return
    for state in states:
        handle.write(
            json.dumps(
                {
                    "stream_monotonic_ns": state.monotonic_ns,
                    "q": state.q.tolist(),
                    "dq": state.dq.tolist(),
                    "actual_tcp": state.tcp.tolist(),
                    "active_sequence": state.active_sequence,
                    "plan_step": state.plan_step,
                    "plan_steps": state.plan_steps,
                    "activation_offset": state.activation_offset,
                    "stop_reason": state.stop_reason_name,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()


def receive_and_record(sock: socket.socket, handle) -> StreamState:
    state, received = receive_latest_state(sock)
    write_states(handle, received)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORDINGS_DIR / "droid_raw_full_v3")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--command-port", type=int, default=51000)
    parser.add_argument("--state-port", type=int, default=51001)
    parser.add_argument("--chunks", type=int, help="Legacy fixed replay length in 16-action units.")
    parser.add_argument("--through-grasp", action="store_true")
    parser.add_argument("--replan-at-step", type=int, default=8)
    parser.add_argument("--start-step", type=int, default=0, help="15 Hz source action index")
    parser.add_argument("--start-joint-tolerance", type=float, default=0.015)
    parser.add_argument("--max-final-joint-error", type=float, default=0.030)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.chunks is not None and args.through_grasp:
        raise SystemExit("Use either --chunks or --through-grasp")
    if args.chunks is not None and args.chunks < 1:
        raise SystemExit("--chunks must be positive")
    if not 1 <= args.replan_at_step < MAX_STEPS:
        raise SystemExit("--replan-at-step must be in [1, 15]")
    if args.start_step < 0 or args.start_joint_tolerance <= 0 or args.max_final_joint_error <= 0:
        raise SystemExit("--start-step must be nonnegative and tolerances positive")

    episode_dir = args.record_dir / args.episode
    states = load_jsonl(episode_dir / "robot_state.jsonl")
    sampled = states[::4]  # 60 Hz recorder -> 15 Hz policy sequence.
    if args.through_grasp:
        final_source_step = grasp_aligned_step(sampled, load_jsonl(episode_dir / "actions.jsonl"))
        total_steps = final_source_step - args.start_step + 1
    else:
        total_steps = (args.chunks or 3) * MAX_STEPS
    if total_steps < 1 or args.start_step + total_steps >= len(sampled):
        raise SystemExit("Requested replay extends beyond the recorded episode")

    expected_start = np.asarray(sampled[args.start_step]["q"], dtype=float)
    expected_end = np.asarray(sampled[args.start_step + total_steps]["q"], dtype=float)
    velocities = np.asarray(
        [
            row.get("dq_d", row["dq"])
            for row in sampled[args.start_step : args.start_step + total_steps]
        ],
        dtype=float,
    )
    if velocities.shape != (total_steps, 7) or not np.isfinite(velocities).all():
        raise RuntimeError(f"Invalid recorded velocity shape: {velocities.shape}")

    windows = max(1, math.ceil(max(0, total_steps - MAX_STEPS) / args.replan_at_step) + 1)
    mode = "through grasp boundary" if args.through_grasp else f"{args.chunks or 3} chunks"
    print("[RECORDED REPLAY] Arm-only: policy, cameras, and gripper are bypassed.")
    print(
        f"[RECORDED REPLAY] episode={args.episode} mode={mode} "
        f"steps={total_steps}@15Hz overlapping_windows~{windows}"
    )

    state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    state_socket.bind(("127.0.0.1", args.state_port))
    state_socket.settimeout(2.0)
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", args.command_port)
    sequence = 0
    source_base = 0
    log_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.output.open("w", encoding="utf-8", buffering=1)
    try:
        state = receive_and_record(state_socket, log_handle)
        start_error = float(np.max(np.abs(state.q - expected_start)))
        print(f"[RECORDED REPLAY] start max|q-live-q-recorded|={start_error:.4f} rad")
        if start_error > args.start_joint_tolerance:
            raise RuntimeError("Live arm is not at the recorded start state; refusing playback")

        while source_base < total_steps:
            plan = velocities[source_base : min(source_base + MAX_STEPS, total_steps)]
            sequence += 1
            command_socket.sendto(
                make_command(sequence, state.monotonic_ns, plan), target
            )
            print(
                f"[RECORDED REPLAY] sent seq={sequence} source={source_base}:"
                f"{source_base + len(plan)} peak={np.abs(plan).max():.3f}"
            )
            while True:
                state = receive_and_record(state_socket, log_handle)
                if state.stop_reason:
                    raise RuntimeError(f"Streamer stopped: {state.stop_reason_name}")
                if state.active_sequence == sequence:
                    break
            print(
                f"[RECORDED REPLAY] activated seq={sequence} "
                f"offset={state.activation_offset}"
            )

            final_window = source_base + len(plan) >= total_steps
            target_step = len(plan) if final_window else args.replan_at_step
            while state.active_sequence != sequence or state.plan_step < target_step:
                state = receive_and_record(state_socket, log_handle)
                if state.stop_reason:
                    raise RuntimeError(f"Streamer stopped: {state.stop_reason_name}")
            if final_window:
                source_base = total_steps
            else:
                # Align the next recorded window to the actual action index at
                # the fresh state packet, not to an assumed exact wakeup time.
                source_base += state.plan_step

        final_error = state.q - expected_end
        print(f"[RECORDED REPLAY] final q error={np.array2string(final_error, precision=4)}")
        final_max_error = float(np.max(np.abs(final_error)))
        print(f"[RECORDED REPLAY] final max|q error|={final_max_error:.4f} rad")
        if final_max_error > args.max_final_joint_error:
            raise RuntimeError(
                "Recorded executor tracking exceeded the final joint-error limit: "
                f"{final_max_error:.4f} > {args.max_final_joint_error:.4f} rad"
            )
        if args.through_grasp:
            print("[RECORDED REPLAY] complete approach reached recorded grasp boundary; gripper remained open.")
    finally:
        try:
            sequence += 1
            command_socket.sendto(make_command(sequence, time.monotonic_ns(), None), target)
        except OSError:
            pass
        state_socket.close()
        command_socket.close()
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
