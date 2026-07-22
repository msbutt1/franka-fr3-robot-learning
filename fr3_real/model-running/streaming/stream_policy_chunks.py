#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/streaming/stream_policy_chunks.py --help
"""Run timestamped receding-horizon OpenPI inference for the FR3 streamer.

This process never commands the robot or gripper directly. It continuously
records 60 Hz streamer states, captures every policy image, and tags each plan
with the exact robot-state timestamp used for inference. The streamer skips
actions made stale by inference latency and activates each replacement plan
immediately.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

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
)
from streaming.stream_protocol import StreamState, make_command, unpack_state


EXIT_INVALID_INFRASTRUCTURE = 3
EXIT_STREAM_STOP = 4
INFRASTRUCTURE_STOP_REASONS = {"watchdog", "stale_plan"}
TRUNCATED_PLAN_RESERVE_STEPS = 5


def json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class EventLog:
    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, event: str, **values: object) -> None:
        record = {
            "event": event,
            "host_time_ns": time.time_ns(),
            "host_monotonic_ns": time.monotonic_ns(),
            **values,
        }
        with self._lock:
            self._handle.write(json.dumps(json_value(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def state_record(state: StreamState) -> dict[str, object]:
    return {
        "stream_monotonic_ns": state.monotonic_ns,
        "q": state.q.tolist(),
        "dq": state.dq.tolist(),
        "actual_tcp": state.tcp.tolist(),
        "active_sequence": state.active_sequence,
        "plan_step": state.plan_step,
        "plan_steps": state.plan_steps,
        "activation_offset": state.activation_offset,
        "stop_reason": state.stop_reason_name,
    }


class StateMonitor:
    """Continuously drain state UDP so inference never observes a backlog."""

    def __init__(self, port: int, log: EventLog):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", port))
        self._socket.settimeout(0.2)
        self._log = log
        self._condition = threading.Condition()
        self._latest: StreamState | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="stream-state-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._closed:
            try:
                payload, _ = self._socket.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                return
            state = unpack_state(payload)
            if state is None:
                continue
            self._log.write("stream_state", **state_record(state))
            with self._condition:
                self._latest = state
                self._condition.notify_all()

    def wait_for(
        self,
        predicate: Callable[[StreamState], bool],
        timeout: float,
        description: str,
    ) -> StreamState:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest is not None and predicate(self._latest):
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {description}")
                self._condition.wait(remaining)

    def latest(self) -> StreamState | None:
        with self._condition:
            return self._latest

    def close(self) -> None:
        self._closed = True
        self._socket.close()
        self._thread.join(timeout=1.0)


def first_transition(values: np.ndarray, threshold: float, *, closing: bool) -> int | None:
    matches = np.flatnonzero(values >= threshold if closing else values <= threshold)
    return int(matches[0]) if len(matches) else None


def is_websocket_failure(exception: Exception) -> bool:
    return exception.__class__.__module__.startswith("websockets.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="10.6.38.133")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--command-port", type=int, default=51000)
    parser.add_argument("--state-port", type=int, default=51001)
    parser.add_argument("--exterior-serial", default=DEFAULT_EXTERIOR_SERIAL)
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--policy-hz", type=float, default=15.0)
    parser.add_argument("--execute-steps", type=int, default=16)
    parser.add_argument(
        "--replan-at-step",
        "--prefetch-at-step",
        dest="replan_at_step",
        type=int,
        default=8,
        help="Observe and infer again after this active action index.",
    )
    parser.add_argument("--gripper-closedness", type=float, default=0.0)
    parser.add_argument("--stop-on-gripper-threshold", type=float)
    parser.add_argument("--holding", action="store_true")
    parser.add_argument("--stop-on-release-threshold", type=float)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 runs until a transition or stop.")
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--activation-timeout", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=EVAL_LOGS_DIR / "stream_sessions")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if not 1 <= args.execute_steps <= 16:
        raise SystemExit("--execute-steps must be in [1, 16]")
    if not 1 <= args.replan_at_step < args.execute_steps:
        raise SystemExit("--replan-at-step must be in [1, execute-steps - 1]")
    if args.policy_hz <= 0 or args.state_timeout <= 0 or args.activation_timeout <= 0:
        raise SystemExit("Timing values must be positive")
    if not 0.0 <= args.gripper_closedness <= 1.0:
        raise SystemExit("--gripper-closedness must be in [0, 1]")
    if args.stop_on_release_threshold is not None and not args.holding:
        raise SystemExit("--stop-on-release-threshold requires --holding")

    run_id = args.run_id or datetime.now().strftime("stream_%Y%m%d_%H%M%S")
    run_dir = args.log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    event_log = EventLog(run_dir / "events.jsonl")
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_target = ("127.0.0.1", args.command_port)
    monitor: StateMonitor | None = None
    exterior = None
    wrist = None
    sequence = 0
    exit_code = 0

    def send_stop() -> None:
        nonlocal sequence
        sequence += 1
        command_socket.sendto(make_command(sequence, time.monotonic_ns(), None), command_target)
        event_log.write("stop_command", sequence=sequence)

    def interrupted(*_unused: object) -> None:
        send_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    try:
        client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
        metadata = client.get_server_metadata()
        event_log.write(
            "session_start",
            run_id=run_id,
            server=f"{args.server_host}:{args.server_port}",
            server_metadata=metadata,
            policy_hz=args.policy_hz,
            replan_at_step=args.replan_at_step,
            execute_steps=args.execute_steps,
            prompt=args.prompt,
        )
        exterior = start_camera(
            args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps
        )
        wrist = start_camera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps)
        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)
        monitor = StateMonitor(args.state_port, event_log)
        print(
            "[FEEDER] Timestamped immediate receding horizon. Arm and gripper commands remain "
            "owned by the streamer and operator."
        )
        print(f"[FEEDER] telemetry={run_dir / 'events.jsonl'} images={run_dir}")

        chunk_count = 0
        current_sequence = 0
        current_replan_step = args.replan_at_step
        while args.max_chunks == 0 or chunk_count < args.max_chunks:
            if current_sequence == 0:
                trigger_state = monitor.wait_for(
                    lambda state: True, args.state_timeout, "the first streamer state"
                )
            else:
                trigger_state = monitor.wait_for(
                    lambda state: state.stop_reason != 0
                    or (
                        state.active_sequence == current_sequence
                        and state.plan_step >= current_replan_step
                    ),
                    args.state_timeout,
                    f"sequence {current_sequence} replan point",
                )
            if trigger_state.stop_reason:
                print(f"[FEEDER] STREAM_STOP reason={trigger_state.stop_reason_name}")
                event_log.write("stream_stop_observed", **state_record(trigger_state))
                exit_code = EXIT_STREAM_STOP
                break

            state_before_images = trigger_state.monotonic_ns
            image_started_ns = time.monotonic_ns()
            exterior_rgb = read_rgb(exterior)
            wrist_rgb = read_rgb(wrist)
            image_finished_ns = time.monotonic_ns()
            observation_state = monitor.wait_for(
                lambda state: state.monotonic_ns > state_before_images or state.stop_reason != 0,
                args.state_timeout,
                "a post-image robot state",
            )
            if observation_state.stop_reason:
                print(f"[FEEDER] STREAM_STOP reason={observation_state.stop_reason_name}")
                exit_code = EXIT_STREAM_STOP
                break

            image_index = chunk_count + 1
            exterior_path = run_dir / f"{image_index:04d}_exterior.png"
            wrist_path = run_dir / f"{image_index:04d}_wrist.png"
            Image.fromarray(exterior_rgb).save(exterior_path)
            Image.fromarray(wrist_rgb).save(wrist_path)
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(exterior_rgb, 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_rgb, 224, 224)
                ),
                "observation/joint_position": observation_state.q.astype(np.float32),
                "observation/gripper_position": np.asarray(
                    [args.gripper_closedness], dtype=np.float32
                ),
                "prompt": args.prompt,
            }
            event_log.write(
                "observation",
                observation_index=image_index,
                image_capture_started_ns=image_started_ns,
                image_capture_finished_ns=image_finished_ns,
                exterior_image=str(exterior_path),
                wrist_image=str(wrist_path),
                gripper_closedness=args.gripper_closedness,
                **state_record(observation_state),
            )

            started = time.perf_counter()
            try:
                response = client.infer(observation)
            except (ConnectionError, OSError, TimeoutError) as exc:
                event_log.write("policy_connection_failure", error=repr(exc))
                print(f"[FEEDER] INVALID policy connection failed: {exc}")
                exit_code = EXIT_INVALID_INFRASTRUCTURE
                break
            except Exception as exc:
                if not is_websocket_failure(exc):
                    raise
                event_log.write("policy_connection_failure", error=repr(exc))
                print(f"[FEEDER] INVALID policy WebSocket failed: {exc}")
                exit_code = EXIT_INVALID_INFRASTRUCTURE
                break
            latency = time.perf_counter() - started
            actions = np.asarray(response["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= actions.shape[0] <= 16 or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid policy chunk: shape={actions.shape}")

            latest_after_inference = monitor.latest()
            if latest_after_inference is not None and latest_after_inference.stop_reason:
                event_log.write(
                    "inference_returned_after_stop",
                    latency_s=latency,
                    actions=actions.tolist(),
                    **state_record(latest_after_inference),
                )
                if latest_after_inference.stop_reason_name in INFRASTRUCTURE_STOP_REASONS:
                    print(
                        "[FEEDER] INVALID inference returned after infrastructure stop "
                        f"reason={latest_after_inference.stop_reason_name} latency={latency:.3f}s"
                    )
                    exit_code = EXIT_INVALID_INFRASTRUCTURE
                else:
                    print(
                        "[FEEDER] STREAM_STOP inference returned after safety stop "
                        f"reason={latest_after_inference.stop_reason_name} latency={latency:.3f}s"
                    )
                    exit_code = EXIT_STREAM_STOP
                break

            plan_budget = None
            if observation_state.active_sequence != 0 and observation_state.plan_steps:
                plan_budget = max(
                    0.0,
                    (observation_state.plan_steps - observation_state.plan_step) / args.policy_hz,
                )
            if plan_budget is not None and latency >= plan_budget:
                event_log.write(
                    "inference_budget_violation",
                    latency_s=latency,
                    plan_budget_s=plan_budget,
                    actions=actions.tolist(),
                    **state_record(observation_state),
                )
                print(
                    "[FEEDER] INVALID inference latency exceeded remaining plan budget: "
                    f"latency={latency:.3f}s budget={plan_budget:.3f}s"
                )
                exit_code = EXIT_INVALID_INFRASTRUCTURE
                break

            gripper = actions[: args.execute_steps, 7]
            transition_kind = None
            transition_index = None
            if args.holding and args.stop_on_release_threshold is not None:
                transition_index = first_transition(
                    gripper, args.stop_on_release_threshold, closing=False
                )
                transition_kind = "release" if transition_index is not None else None
            elif not args.holding and args.stop_on_gripper_threshold is not None:
                transition_index = first_transition(
                    gripper, args.stop_on_gripper_threshold, closing=True
                )
                transition_kind = "closure" if transition_index is not None else None

            transmitted_steps = (
                transition_index + 1 if transition_index is not None else args.execute_steps
            )
            velocities = actions[:transmitted_steps, :7]
            estimated_offset = int(
                max(0, time.monotonic_ns() - observation_state.monotonic_ns)
                * args.policy_hz
                // 1_000_000_000
            )
            if estimated_offset >= transmitted_steps:
                event_log.write(
                    "transition_or_plan_stale_before_send",
                    latency_s=latency,
                    estimated_activation_offset=estimated_offset,
                    transmitted_steps=transmitted_steps,
                    transition_kind=transition_kind,
                    transition_index=transition_index,
                    actions=actions.tolist(),
                    **state_record(observation_state),
                )
                print(
                    "[FEEDER] INVALID requested action prefix was already stale before send: "
                    f"offset={estimated_offset} steps={transmitted_steps}"
                )
                exit_code = EXIT_INVALID_INFRASTRUCTURE
                break

            sequence += 1
            command_socket.sendto(
                make_command(sequence, observation_state.monotonic_ns, velocities), command_target
            )
            chunk_count += 1
            event_log.write(
                "action_chunk_sent",
                sequence=sequence,
                observation_index=image_index,
                latency_s=latency,
                plan_budget_s=plan_budget,
                estimated_activation_offset=estimated_offset,
                transmitted_steps=transmitted_steps,
                transition_kind=transition_kind,
                transition_index=transition_index,
                actions=actions.tolist(),
                response={key: value for key, value in response.items() if key != "actions"},
                **state_record(observation_state),
            )
            activation = monitor.wait_for(
                lambda state: state.stop_reason != 0 or state.active_sequence == sequence,
                args.activation_timeout,
                f"sequence {sequence} activation",
            )
            if activation.stop_reason:
                event_log.write("plan_rejected", sequence=sequence, **state_record(activation))
                print(f"[FEEDER] STREAM_STOP reason={activation.stop_reason_name}")
                exit_code = EXIT_STREAM_STOP
                break
            event_log.write(
                "plan_activated",
                sequence=sequence,
                activation_offset=activation.activation_offset,
                acknowledged_plan_step=activation.plan_step,
                accepted_steps=activation.plan_steps,
                actual_tcp=activation.tcp.tolist(),
            )
            accepted_steps = activation.plan_steps
            if not 1 <= accepted_steps <= transmitted_steps:
                event_log.write(
                    "invalid_plan_length_acknowledgement",
                    sequence=sequence,
                    transmitted_steps=transmitted_steps,
                    accepted_steps=accepted_steps,
                    **state_record(activation),
                )
                print(
                    "[FEEDER] INVALID streamer acknowledged an invalid plan length: "
                    f"accepted={accepted_steps} transmitted={transmitted_steps}"
                )
                exit_code = EXIT_INVALID_INFRASTRUCTURE
                break
            plan_truncated = accepted_steps < transmitted_steps
            current_replan_step = min(
                args.replan_at_step,
                max(1, accepted_steps - TRUNCATED_PLAN_RESERVE_STEPS),
            )
            print(
                f"[FEEDER {chunk_count}] seq={sequence} observed_step={observation_state.plan_step} "
                f"latency={latency:.3f}s budget="
                f"{plan_budget if plan_budget is not None else float('nan'):.3f}s "
                f"activation_offset={activation.activation_offset} accepted_steps={accepted_steps} raw_peak="
                f"{np.max(np.abs(velocities)):.3f} gripper={gripper.min():.3f}..{gripper.max():.3f}"
            )
            current_sequence = sequence

            if plan_truncated:
                event_log.write(
                    "plan_truncated_by_guard",
                    sequence=sequence,
                    transmitted_steps=transmitted_steps,
                    accepted_steps=accepted_steps,
                    next_replan_step=current_replan_step,
                    transition_kind=transition_kind,
                    transition_index=transition_index,
                    **state_record(activation),
                )
                print(
                    "[FEEDER] guard truncated the plan before its first unsafe action; "
                    f"replanning at step {current_replan_step}."
                )
                continue

            if transition_kind is not None:
                completed = monitor.wait_for(
                    lambda state: state.stop_reason != 0
                    or (
                        state.active_sequence == sequence
                        and state.plan_step >= transmitted_steps
                    ),
                    max(args.state_timeout, transmitted_steps / args.policy_hz + 0.5),
                    f"{transition_kind} action prefix completion",
                )
                if completed.stop_reason:
                    print(f"[FEEDER] STREAM_STOP reason={completed.stop_reason_name}")
                    exit_code = EXIT_STREAM_STOP
                    break
                event_log.write(
                    "gripper_transition_reached",
                    sequence=sequence,
                    transition_kind=transition_kind,
                    transition_index=transition_index,
                    transmitted_steps=transmitted_steps,
                    **state_record(completed),
                )
                print(
                    f"[FEEDER] policy requested gripper {transition_kind} at action "
                    f"index={transition_index}; executed arm prefix through that exact action."
                )
                break

            if args.max_chunks and chunk_count >= args.max_chunks:
                completed = monitor.wait_for(
                    lambda state: state.stop_reason != 0
                    or (
                        state.active_sequence == sequence
                        and state.plan_step >= transmitted_steps
                    ),
                    transmitted_steps / args.policy_hz + args.state_timeout,
                    "final requested plan completion",
                )
                if completed.stop_reason:
                    print(f"[FEEDER] STREAM_STOP reason={completed.stop_reason_name}")
                    exit_code = EXIT_STREAM_STOP
                else:
                    print("[FEEDER] final requested plan completed.")
                break
    except TimeoutError as exc:
        event_log.write("state_timeout", error=str(exc))
        print(f"[FEEDER] STREAM_STOP no state acknowledgement: {exc}")
        exit_code = EXIT_STREAM_STOP
    finally:
        try:
            send_stop()
        except OSError:
            pass
        if monitor is not None:
            time.sleep(0.1)
            monitor.close()
        command_socket.close()
        if exterior is not None:
            exterior.stop()
        if wrist is not None:
            wrist.stop()
        event_log.write("session_end", exit_code=exit_code)
        event_log.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
