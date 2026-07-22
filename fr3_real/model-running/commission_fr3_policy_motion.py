#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/commission_fr3_policy_motion.py --help
"""Execute one guarded OpenPI velocity segment for FR3 commissioning."""

from __future__ import annotations

import argparse
import signal
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from shadow_fr3_policy import (
    DEFAULT_EXTERIOR_SERIAL,
    DEFAULT_PROMPT,
    DEFAULT_WRIST_SERIAL,
    read_rgb,
    start_camera,
    vector,
)


JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


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
    parser.add_argument("--execute_steps", type=int, default=8)
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--velocity_cap", type=float, default=0.20)
    parser.add_argument("--dynamics_factor", type=float, default=0.08)
    parser.add_argument(
        "--guard_displacement_scale",
        type=float,
        default=2.0,
        help="Conservative multiplier used only for joint/workspace prediction. "
        "The executed velocity command is unchanged.",
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

    if not 1 <= args.execute_steps <= 16:
        raise SystemExit("--execute_steps must be in [1, 16]")
    if args.control_hz <= 0 or args.velocity_cap <= 0 or args.guard_displacement_scale < 1:
        raise SystemExit(
            "--control_hz and --velocity_cap must be positive; "
            "--guard_displacement_scale must be >= 1"
        )
    if not 0 < args.dynamics_factor <= 1:
        raise SystemExit("--dynamics_factor must be in (0, 1]")

    print("[COMMISSION] Exactly one guarded arm segment will be executed.")
    print("[COMMISSION] Gripper commands are disabled in this script.")
    print("[COMMISSION] Keep the physical E-stop within immediate reach.")
    input("Confirm the workspace is clear, then press Enter to connect...")

    from franky import (
        Duration,
        Frame,
        Gripper,
        JointVelocityStopMotion,
        JointVelocityWaypoint,
        JointVelocityWaypointMotion,
        Robot,
    )

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor
    gripper = Gripper(args.robot_ip)

    def sigint_stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        try:
            robot.stop()
            print("[KILLSWITCH] robot.stop() returned.")
        finally:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, sigint_stop)

    client = websocket_client_policy.WebsocketClientPolicy(
        args.server_host, args.server_port
    )
    exterior = start_camera(
        args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps
    )
    wrist = start_camera(
        args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps
    )

    def stop_velocity_mode() -> None:
        try:
            robot.move(JointVelocityStopMotion())
        except Exception:
            robot.stop()

    try:
        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)

        exterior_rgb = read_rgb(exterior)
        wrist_rgb = read_rgb(wrist)
        state = robot.state
        q = vector(state.q).astype(float)
        gripper_width = float(gripper.state.width)
        closedness = float(np.clip(1.0 - gripper_width / 0.08, 0.0, 1.0))
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

        started = time.perf_counter()
        actions = np.asarray(client.infer(observation)["actions"], dtype=float)
        latency = time.perf_counter() - started
        if actions.shape != (16, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid policy action chunk: shape={actions.shape}")

        velocities = actions[: args.execute_steps, :7].copy()
        raw_peak = float(np.max(np.abs(velocities)))
        velocity_scale = min(1.0, args.velocity_cap / max(raw_peak, 1e-9))
        velocities *= velocity_scale
        dt = 1.0 / args.control_hz
        # Franky's velocity-waypoint generator includes acceleration/deceleration
        # timing. Keep a conservative envelope around its nominal displacement;
        # this changes checks only, never the velocity sent to the robot.
        predicted_q = q + np.cumsum(
            velocities * dt * args.guard_displacement_scale, axis=0
        )
        lower = JOINT_MIN + args.joint_margin
        upper = JOINT_MAX - args.joint_margin
        if np.any(predicted_q < lower) or np.any(predicted_q > upper):
            raise RuntimeError("Predicted segment violates guarded joint limits")

        # This Franky build returns a local transform (and zero Jacobians) for
        # Frame.EndEffector. Frame.Joint7 is base-frame FK, so calibrate the
        # fixed Joint7 -> measured TCP transform at the current pose and use
        # that for each short-horizon workspace prediction.
        measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=float)
        joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=float)
        if measured_tcp.shape != (4, 4) or joint7_current.shape != (4, 4):
            raise RuntimeError("Franky pose matrices must be 4x4")
        joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
        measured_xyz = measured_tcp[:3, 3]
        calibrated_xyz = (joint7_current @ joint7_to_tcp)[:3, 3]
        fk_self_error = float(np.max(np.abs(calibrated_xyz - measured_xyz)))
        if fk_self_error > 1e-8:
            raise RuntimeError(
                "Franky FK self-check failed before motion: "
                f"measured={measured_xyz}, calibrated={calibrated_xyz}, error={fk_self_error:.3e}"
            )

        predicted_xyz = []
        for q_step in predicted_q:
            joint7_step = np.asarray(
                robot.model.pose(Frame.Joint7, q_step, state.F_T_EE, state.EE_T_K).matrix,
                dtype=float,
            )
            predicted_xyz.append((joint7_step @ joint7_to_tcp)[:3, 3])
        predicted_xyz = np.asarray(predicted_xyz)
        xyz_min = np.asarray([args.min_x, args.min_y, args.min_z])
        xyz_max = np.asarray([args.max_x, args.max_y, args.max_z])
        if np.any(predicted_xyz < xyz_min) or np.any(predicted_xyz > xyz_max):
            raise RuntimeError(
                "Predicted segment leaves guarded Cartesian bounds: "
                f"min={predicted_xyz.min(axis=0)}, max={predicted_xyz.max(axis=0)}"
            )

        print(f"[CHECK] inference latency={latency:.3f}s")
        print(
            f"[CHECK] policy peak={raw_peak:.3f} rad/s scale={velocity_scale:.3f} "
            f"executed peak={np.max(np.abs(velocities)):.3f} rad/s"
        )
        print(
            f"[CHECK] workspace/joint prediction uses {args.guard_displacement_scale:.2f}x "
            "nominal displacement envelope"
        )
        print(f"[CHECK] q start={np.array2string(q, precision=3)}")
        print(
            f"[CHECK] FK measured={np.array2string(measured_xyz, precision=3)} "
            f"calibrated={np.array2string(calibrated_xyz, precision=3)} error={fk_self_error:.3e}"
        )
        print(
            f"[CHECK] predicted xyz start={np.array2string(predicted_xyz[0], precision=3)} "
            f"end={np.array2string(predicted_xyz[-1], precision=3)}"
        )
        print("[CHECK] selected joint-velocity waypoints:")
        print(np.array2string(velocities, precision=4, suppress_small=True))
        confirmation = input(
            "Type MOVE exactly to execute this one arm-only segment, anything else aborts: "
        )
        if confirmation != "MOVE":
            print("[COMMISSION] aborted without motion.")
            return

        waypoint_ms = max(1, round(1000.0 / args.control_hz))
        waypoints = [
            JointVelocityWaypoint(v, minimum_time=Duration(waypoint_ms))
            for v in velocities
        ]
        waypoints.append(
            JointVelocityWaypoint(
                np.zeros(7),
                minimum_time=Duration(waypoint_ms),
                hold_target_duration=Duration(waypoint_ms),
            )
        )
        robot.move(JointVelocityWaypointMotion(waypoints))
        end_state = robot.state
        end_q = vector(end_state.q)
        end_xyz = vector(end_state.O_T_EE.translation)
        print(f"[DONE] q delta={np.array2string(end_q - q, precision=4)}")
        print(f"[DONE] final xyz={np.array2string(end_xyz, precision=4)}")
        print("[DONE] one segment complete; no gripper command was issued.")
    except BaseException:
        stop_velocity_mode()
        raise
    finally:
        exterior.stop()
        wrist.stop()


if __name__ == "__main__":
    main()
