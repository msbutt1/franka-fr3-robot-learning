#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/guarded_fr3_policy_arm_rollout.py --help
"""Execute guarded OpenPI joint-velocity rollouts for FR3.

The policy was trained at 15 Hz.  This controller uses a short receding
horizon: it executes a prefix of each prediction, obtains a new observation,
and then replans.  Rate limits are applied per joint, never by scaling an
entire chunk from its largest element.
"""

from __future__ import annotations

import argparse
import itertools
import signal
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from commission_fr3_policy_motion import JOINT_MAX, JOINT_MIN
from shadow_fr3_policy import (
    DEFAULT_EXTERIOR_SERIAL,
    DEFAULT_PROMPT,
    DEFAULT_WRIST_SERIAL,
    read_rgb,
    start_camera,
    vector,
)


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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="Number of chunks; use 0 (default) to keep prompting until you stop.",
    )
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=8,
        help="Policy actions to execute before taking a new image/state and replanning.",
    )
    parser.add_argument(
        "--velocity_cap",
        type=float,
        default=0.10,
        help="Conservative symmetric per-joint cap in rad/s when --joint_velocity_caps is omitted.",
    )
    parser.add_argument(
        "--joint_velocity_caps",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Optional seven positive per-joint caps in rad/s. Values clip only their own joint.",
    )
    # Match the demonstrations: changing Franky's dynamics factor changes how
    # the velocity waypoints are physically realized, even with identical
    # policy actions.
    parser.add_argument("--dynamics_factor", type=float, default=0.085)
    parser.add_argument("--guard_displacement_scale", type=float, default=2.0)
    parser.add_argument(
        "--gripper_stop_threshold",
        type=float,
        default=0.5,
        help="Stop before executing a chunk when the predicted closedness reaches this value.",
    )
    parser.add_argument(
        "--enable_gripper_grasp",
        action="store_true",
        help="After an explicitly confirmed final descent chunk, issue one stationary grasp. Disabled by default.",
    )
    parser.add_argument("--grasp_width", type=float, default=0.04)
    parser.add_argument("--grasp_speed", type=float, default=0.02)
    parser.add_argument("--grasp_force", type=float, default=15.0)
    parser.add_argument(
        "--enter_to_move",
        action="store_true",
        help="Accept Enter for normal arm-only chunks. The final grasp still requires GRASP.",
    )
    parser.add_argument(
        "--auto_move",
        action="store_true",
        help="Automatically execute normal arm-only chunks. Ctrl+C still calls robot.stop(); grasp remains manual.",
    )
    parser.add_argument("--joint_margin", type=float, default=0.12)
    parser.add_argument("--min_x", type=float, default=0.18)
    parser.add_argument("--max_x", type=float, default=0.75)
    parser.add_argument("--min_y", type=float, default=-0.50)
    parser.add_argument("--max_y", type=float, default=0.45)
    # Cell-1 recording reaches TCP z=0.0370 m at grasp over a z=0.0271 m
    # table. Keep 7 mm below the demonstrated minimum for this task.
    parser.add_argument("--min_z", type=float, default=0.03)
    parser.add_argument("--max_z", type=float, default=0.65)
    args = parser.parse_args()

    if args.chunks < 0:
        raise SystemExit("--chunks must be non-negative")
    if args.enter_to_move and args.auto_move:
        raise SystemExit("Use only one of --enter_to_move or --auto_move")
    if not 1 <= args.execute_steps <= 16:
        raise SystemExit("--execute_steps must be in [1, 16]")
    if args.control_hz <= 0 or args.velocity_cap <= 0 or args.guard_displacement_scale < 1:
        raise SystemExit("Invalid control or guard settings")
    velocity_caps = (
        np.full(7, args.velocity_cap, dtype=float)
        if args.joint_velocity_caps is None
        else np.asarray(args.joint_velocity_caps, dtype=float)
    )
    if velocity_caps.shape != (7,) or np.any(velocity_caps <= 0):
        raise SystemExit("--joint_velocity_caps must contain seven positive values")

    from franky import (
        Duration,
        Frame,
        Gripper,
        JointVelocityStopMotion,
        JointVelocityWaypoint,
        JointVelocityWaypointMotion,
        Robot,
    )

    print("[ROLLOUT] Guarded arm-only policy rollout.")
    print(
        f"[ROLLOUT] Replan every {args.execute_steps} actions at {args.control_hz:.1f} Hz; "
        "gripper commands are disabled unless explicitly enabled."
    )
    print(
        "[ROLLOUT] Per-joint velocity caps="
        f"{np.array2string(velocity_caps, precision=3)} rad/s; no global chunk scaling."
    )
    print(
        f"[ROLLOUT] Stops before a predicted gripper closedness >= {args.gripper_stop_threshold:.2f}."
    )
    input("Keep the workspace clear and E-stop reachable, then press Enter to connect...")

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor
    gripper = Gripper(args.robot_ip)

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    exterior = start_camera(args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps)
    wrist = start_camera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps)

    xyz_lower = np.asarray([args.min_x, args.min_y, args.min_z])
    xyz_upper = np.asarray([args.max_x, args.max_y, args.max_z])
    joint_lower = JOINT_MIN + args.joint_margin
    joint_upper = JOINT_MAX - args.joint_margin
    dt = 1.0 / args.control_hz
    waypoint_ms = max(1, round(1000.0 / args.control_hz))

    try:
        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)

        chunk_numbers = itertools.count(1) if args.chunks == 0 else range(1, args.chunks + 1)
        for chunk in chunk_numbers:
            exterior_rgb = read_rgb(exterior)
            wrist_rgb = read_rgb(wrist)
            state = robot.state
            q = vector(state.q).astype(float)
            closedness = float(np.clip(1.0 - float(gripper.state.width) / 0.08, 0.0, 1.0))
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(exterior_rgb, 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_rgb, 224, 224)
                ),
                "observation/joint_position": q.astype(np.float32),
                "observation/gripper_position": np.asarray([closedness], dtype=np.float32),
                "prompt": args.prompt,
            }
            began = time.perf_counter()
            actions = np.asarray(client.infer(observation)["actions"], dtype=float)
            latency = time.perf_counter() - began
            if actions.shape != (16, 8) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid policy action chunk: shape={actions.shape}")

            raw_velocities = actions[: args.execute_steps, :7].copy()
            raw_peak = float(np.max(np.abs(raw_velocities)))
            velocities = np.clip(raw_velocities, -velocity_caps, velocity_caps)
            clipped_values = int(np.count_nonzero(np.abs(raw_velocities) > velocity_caps))
            predicted_q = q + np.cumsum(
                velocities * dt * args.guard_displacement_scale, axis=0
            )
            if np.any(predicted_q < joint_lower) or np.any(predicted_q > joint_upper):
                raise RuntimeError("Predicted chunk violates guarded joint limits")

            tcp_current = np.asarray(state.O_T_EE.matrix, dtype=float)
            joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=float)
            joint7_to_tcp = np.linalg.solve(joint7_current, tcp_current)
            predicted_xyz = np.asarray([
                (
                    np.asarray(
                        robot.model.pose(Frame.Joint7, q_step, state.F_T_EE, state.EE_T_K).matrix,
                        dtype=float,
                    )
                    @ joint7_to_tcp
                )[:3, 3]
                for q_step in predicted_q
            ])
            if np.any(predicted_xyz < xyz_lower) or np.any(predicted_xyz > xyz_upper):
                raise RuntimeError(
                    "Predicted chunk leaves guarded Cartesian bounds: "
                    f"min={predicted_xyz.min(axis=0)}, max={predicted_xyz.max(axis=0)}"
                )

            chunk_label = f"{chunk}/continuous" if args.chunks == 0 else f"{chunk}/{args.chunks}"
            print(
                f"[CHUNK {chunk_label}] latency={latency:.3f}s raw_peak={raw_peak:.3f} "
                f"commanded_peak={np.max(np.abs(velocities)):.3f} "
                f"per_joint_clips={clipped_values}"
            )
            print(f"[CHUNK {chunk_label}] tcp={np.array2string(tcp_current[:3, 3], precision=3)} "
                  f"predicted_end={np.array2string(predicted_xyz[-1], precision=3)}")
            gripper_actions = actions[: args.execute_steps, 7]
            gripper_max = float(gripper_actions.max())
            print(f"[CHUNK {chunk_label}] model gripper={gripper_actions.min():.3f}..{gripper_max:.3f} "
                  "(ignored)")
            grasp_requested = gripper_max >= args.gripper_stop_threshold
            if grasp_requested and not args.enable_gripper_grasp:
                print(
                    "[ROLLOUT] Policy is requesting gripper closure; stopping before this chunk. "
                    "No gripper command was sent."
                )
                return
            if grasp_requested:
                print(
                    "[ROLLOUT] Policy requests closure in this chunk. The arm will execute the guarded "
                    "descent first, then the stationary gripper will grasp after the arm stops."
                )
                confirmation = input(
                    f"Type GRASP exactly to execute descent + grasp for chunk {chunk}, anything else stops: "
                )
                required_confirmation = "GRASP"
            else:
                if args.auto_move:
                    print(f"[CHUNK {chunk_label}] auto-executing guarded arm-only chunk; Ctrl+C stops.")
                    confirmation = "MOVE"
                else:
                    prompt = (
                        f"Press Enter to execute arm-only chunk {chunk}, anything else stops: "
                        if args.enter_to_move
                        else f"Type MOVE exactly to execute arm-only chunk {chunk}, anything else stops: "
                    )
                    confirmation = input(prompt)
                required_confirmation = "MOVE"
            accepted = (
                confirmation == required_confirmation
                or (args.enter_to_move and not grasp_requested and confirmation == "")
            )
            if not accepted:
                print("[ROLLOUT] stopped without executing this chunk.")
                return

            waypoints = [
                JointVelocityWaypoint(v, minimum_time=Duration(waypoint_ms)) for v in velocities
            ]
            robot.move(JointVelocityWaypointMotion(waypoints))
            end = robot.state
            end_xyz = np.asarray(end.O_T_EE.translation, dtype=float)
            if np.any(end_xyz < xyz_lower) or np.any(end_xyz > xyz_upper):
                robot.stop()
                raise RuntimeError(f"Actual TCP outside guarded bounds after chunk: {end_xyz}")
            print(f"[CHUNK {chunk_label}] final tcp={np.array2string(end_xyz, precision=3)}")
            if grasp_requested:
                grasped = gripper.grasp(
                    args.grasp_width,
                    args.grasp_speed,
                    args.grasp_force,
                    0.01,
                    0.01,
                )
                print(
                    f"[GRIPPER] grasp()={grasped} width={gripper.state.width:.4f} "
                    f"is_grasped={gripper.is_grasped}"
                )
                print("[ROLLOUT] Stopping after first policy-requested grasp; no lift was commanded.")
                return
    except BaseException:
        try:
            robot.move(JointVelocityStopMotion())
        except Exception:
            robot.stop()
        raise
    finally:
        exterior.stop()
        wrist.stop()

    print("[ROLLOUT] Finished all arm-only chunks. No gripper command was issued.")


if __name__ == "__main__":
    main()
