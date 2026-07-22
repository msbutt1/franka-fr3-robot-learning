#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/run_fr3_policy_eval_trial.py --help
"""Run one guarded, autonomous FR3 policy evaluation trial.

The scripted portion moves one cube from the basket to a held-out position and
returns home. The policy then controls arm joint velocities in an eight-action
receding horizon. A human approves or skips its one requested grasp and its
release. Ctrl+C calls robot.stop.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openpi_client import image_tools, websocket_client_policy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from commission_fr3_policy_motion import JOINT_MAX, JOINT_MIN
from franka_motion import (
    DEFAULT_FRANKA_HAND_TCP_OFFSET,
    MotionPlanner,
    assert_safe_flange_z,
    clear_robot_errors,
    compute_travel_z_floor,
    configure_collision_behavior,
    flange_z_for_tcp_z,
)
from shadow_fr3_policy import (
    DEFAULT_EXTERIOR_SERIAL,
    DEFAULT_PROMPT,
    DEFAULT_WRIST_SERIAL,
    read_rgb,
    start_camera,
    vector,
)
from repo_paths import CONFIGS_DIR, EVAL_LOGS_DIR


FR3_ROOT = Path(__file__).resolve().parent.parent


def load_eval_position(path: Path, eval_id: str) -> dict:
    positions = json.loads(path.read_text()).get("positions", [])
    matches = [position for position in positions if position.get("eval_id") == eval_id]
    if len(matches) != 1:
        raise SystemExit(f"Could not find exactly one eval_id={eval_id!r} in {path}")
    return matches[0]


def first_state(path: Path) -> dict:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"No robot state in {path}")


def find_policy_start_episode(record_dir: Path, source_cells: list[int]) -> Path:
    for printed_cell in source_cells:
        matches = []
        for episode in sorted(record_dir.glob("*_printed_cell_*")):
            metadata_path = episode / "metadata.json"
            state_path = episode / "robot_state.jsonl"
            if not metadata_path.exists() or not state_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            source = metadata.get("source", {})
            if metadata.get("success", False) and int(source.get("printed_cell", -1)) == printed_cell:
                matches.append(episode)
        if matches:
            return matches[-1]
    raise RuntimeError(f"No successful recorded episode found for source cells {source_cells}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--server_host", default="10.6.38.133")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--manifest", type=Path, default=CONFIGS_DIR / "fr3_interstitial_eval_v1.json")
    parser.add_argument("--eval_id", required=True)
    parser.add_argument(
        "--record_dir",
        type=Path,
        default=FR3_ROOT / "recordings/droid_raw_full_v3",
    )
    parser.add_argument(
        "--policy_start_episode",
        help="Optional recorded episode directory name used for exact policy-start reset.",
    )
    parser.add_argument("--points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument("--exterior_serial", default=DEFAULT_EXTERIOR_SERIAL)
    parser.add_argument("--wrist_serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=60)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--execute_steps", type=int, default=8)
    parser.add_argument("--max_chunks", type=int, default=80)
    parser.add_argument(
        "--joint_velocity_caps",
        type=float,
        nargs=7,
        default=[0.10] * 7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    # Match the value used while recording the demonstrations.
    parser.add_argument("--dynamics_factor", type=float, default=0.085)
    parser.add_argument("--guard_displacement_scale", type=float, default=2.0)
    parser.add_argument("--joint_margin", type=float, default=0.12)
    parser.add_argument("--min_x", type=float, default=0.18)
    parser.add_argument("--max_x", type=float, default=0.75)
    parser.add_argument("--min_y", type=float, default=-0.50)
    parser.add_argument("--max_y", type=float, default=0.45)
    parser.add_argument("--min_z", type=float, default=0.03)
    parser.add_argument("--max_z", type=float, default=0.65)
    parser.add_argument("--cube_height", type=float, default=0.04)
    parser.add_argument("--grasp_lowering", type=float, default=0.010)
    parser.add_argument("--hover_clearance", type=float, default=0.11)
    parser.add_argument("--cell_drop_clearance", type=float, default=0.017)
    parser.add_argument("--transport_clearance", type=float, default=0.05)
    parser.add_argument("--travel_extra_clearance", type=float, default=0.05)
    # The recordings used --max_step 0: one continuous Cartesian command per
    # travel leg. Segmenting at 3 cm plus settle time makes setup visibly
    # stop-and-go and creates a train/eval motion mismatch.
    parser.add_argument("--max_step", type=float, default=0.0)
    parser.add_argument("--settle_time", type=float, default=0.30)
    parser.add_argument("--grasp_width", type=float, default=0.04)
    parser.add_argument("--grasp_speed", type=float, default=0.02)
    parser.add_argument("--grasp_force", type=float, default=15.0)
    parser.add_argument("--open_speed", type=float, default=0.05)
    parser.add_argument("--close_threshold", type=float, default=0.50)
    parser.add_argument("--release_threshold", type=float, default=0.10)
    parser.add_argument("--tcp_offset", type=float, default=DEFAULT_FRANKA_HAND_TCP_OFFSET)
    parser.add_argument("--min_flange_z", type=float, default=0.03)
    parser.add_argument("--log_dir", type=Path, default=EVAL_LOGS_DIR)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required acknowledgement for scripted placement and automatic arm rollout.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Perform basket-to-cell setup and demonstrated-start reset, then exit before policy control.",
    )
    parser.add_argument(
        "--manual-target",
        action="store_true",
        help="Use a manually placed cube and already-positioned arm; skip scripted basket setup.",
    )
    parser.add_argument(
        "--manual-live",
        action="store_true",
        help="Use an arbitrary manually placed cube; reset arm to policy start and skip target placement.",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="Recover a cube from this eval cell to the basket, then reset to the demonstrated policy start.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Return the arm from a successful basket release to the demonstrated policy start.",
    )
    parser.add_argument(
        "--assume-confirmed",
        action="store_true",
        help="Skip the local START prompt; valid only for an orchestrated --reset-only move.",
    )
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("Refusing motion without --run")
    if (args.manual_target or args.manual_live) and (args.recover_only or args.reset_only):
        raise SystemExit("manual-target/manual-live cannot be combined with recover/reset-only")
    if args.manual_target and args.manual_live:
        raise SystemExit("Use only one of --manual-target or --manual-live")
    if sum((args.setup_only, args.recover_only, args.reset_only)) > 1:
        raise SystemExit("Use only one of --setup-only, --recover-only, or --reset-only")
    if args.assume_confirmed and not args.reset_only:
        raise SystemExit("--assume-confirmed is valid only with --reset-only")
    if not 1 <= args.execute_steps <= 16 or args.max_chunks < 1:
        raise SystemExit("--execute_steps must be in [1, 16] and --max_chunks must be positive")
    if args.control_hz <= 0 or args.guard_displacement_scale < 1 or not 0 < args.dynamics_factor <= 1:
        raise SystemExit("Invalid control settings")
    caps = np.asarray(args.joint_velocity_caps, dtype=float)
    if caps.shape != (7,) or np.any(caps <= 0):
        raise SystemExit("--joint_velocity_caps requires seven positive values")
    if not 0 < args.release_threshold < args.close_threshold <= 1:
        raise SystemExit("Require 0 < --release_threshold < --close_threshold <= 1")

    target = load_eval_position(args.manifest, args.eval_id)
    start_episode = (
        args.record_dir / args.policy_start_episode
        if args.policy_start_episode
        else find_policy_start_episode(args.record_dir, list(target["source_cells"]))
    )
    start_state_path = start_episode / "robot_state.jsonl"
    if not start_state_path.exists():
        raise SystemExit(f"Missing recorded start state: {start_state_path}")
    policy_start_q = np.asarray(first_state(start_state_path)["q"], dtype=float)
    if policy_start_q.shape != (7,):
        raise SystemExit(f"Expected seven joints in {start_state_path}, got {policy_start_q.shape}")
    points = json.loads(args.points.read_text())
    required = ["pad_center", "basket_rim"]
    missing = [key for key in required if key not in points]
    if missing:
        raise SystemExit(f"{args.points} missing: {missing}")
    pad = np.asarray(points["pad_center"], dtype=float)
    rim = np.asarray(points["basket_rim"], dtype=float)
    target_xyz = np.asarray([target["x"], target["y"], target["table_z"]], dtype=float)

    print("[EVAL] One autonomous guarded evaluation trial.")
    print(f"[EVAL] manifest={args.manifest} eval_id={args.eval_id} role={target.get('role')}")
    print(f"[EVAL] target xyz={np.array2string(target_xyz, precision=4)} source_cells={target['source_cells']}")
    print(f"[EVAL] recorded policy-start episode={start_episode.name}")
    print("[EVAL] Scripted setup: basket -> target cell -> home.")
    print(
        f"[EVAL] Policy: replan every {args.execute_steps} steps at {args.control_hz:.1f} Hz; "
        f"per-joint caps={np.array2string(caps, precision=3)} rad/s."
    )
    print("[EVAL] At predicted release, RELEASE opens only after visual approval, then returns home.")
    print("[EVAL] At closure: CLOSE continues after one grasp; SKIP recycles cube to basket and homes.")
    if args.setup_only:
        print("[EVAL] setup-only: no policy or gripper decision loop will start.")
    if args.manual_target:
        print("[EVAL] manual-target: cube placement and arm positioning are operator-owned; setup motion skipped.")
    if args.manual_live:
        print("[EVAL] manual-live: cube placement is operator-owned; resetting arm to policy start, target placement skipped.")
    if args.recover_only:
        print("[EVAL] recover-only: cell -> basket -> demonstrated policy start; no policy control will start.")
    if args.reset_only:
        print("[EVAL] reset-only: basket release -> demonstrated policy start; no policy control or gripper command.")
    cube_location = (
        "cube on the target cell"
        if args.recover_only
        else "cube released in the basket"
        if args.reset_only
        else "cube manually placed on the measured target and arm at policy start"
        if args.manual_target
        else "cube manually placed anywhere in the safe workspace"
        if args.manual_live
        else "cube correctly oriented in basket"
    )
    if args.assume_confirmed:
        print("[EVAL] reset-only motion authorized by the active trial orchestrator.")
    else:
        confirmation = input(
            f"Workspace/path clear, {cube_location}, E-stop reachable. Type START to begin: "
        )
        if confirmation.strip().upper() != "START":
            print("[EVAL] aborted before connection.")
            raise SystemExit(2)

    from franky import (
        Duration,
        Frame,
        Gripper,
        JointMotion,
        JointState,
        JointVelocityStopMotion,
        JointVelocityWaypoint,
        JointVelocityWaypointMotion,
        Robot,
    )

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor
    gripper = Gripper(args.robot_ip)
    exterior = None
    wrist = None

    def stop_robot() -> None:
        try:
            robot.move(JointVelocityStopMotion())
        except Exception:
            robot.stop()

    def on_sigint(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"policy_eval_{args.eval_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    result = {"status": "started", "eval_id": args.eval_id, "target": target}

    try:
        clear_robot_errors(robot)
        configure_collision_behavior(robot)
        if not args.reset_only:
            gripper.homing()
            if args.grasp_width >= gripper.max_width:
                raise RuntimeError("Configured grasp width is not below gripper maximum width")

        home_xyz = np.asarray(robot.current_pose.end_effector_pose.translation, dtype=float)
        basket_grasp_z = float(pad[2] + args.cube_height / 2 - args.grasp_lowering)
        transport_z = float(rim[2] + args.transport_clearance)
        hover_z = float(target_xyz[2] + args.hover_clearance)
        grasp_z = float(target_xyz[2] + args.cube_height / 2 - args.grasp_lowering)
        drop_z = min(hover_z, grasp_z + args.cell_drop_clearance)
        basket_release_z = float(pad[2] + args.cube_height / 2)
        for label, z in (
            ("basket grasp", basket_grasp_z),
            ("basket transport", transport_z),
            ("basket release", basket_release_z),
            ("cell hover", hover_z),
            ("cell grasp", grasp_z),
            ("cell drop", drop_z),
        ):
            assert_safe_flange_z(label, flange_z_for_tcp_z(z, args.tcp_offset), args.min_flange_z)
        travel_z = compute_travel_z_floor(
            home_xyz, [transport_z, hover_z, drop_z], args.travel_extra_clearance, include_home_z=False
        )
        motion = MotionPlanner(robot, args.dynamics_factor, args.max_step, travel_z, args.settle_time)

        def reset_to_recorded_policy_start() -> None:
            """Move to a demonstrated initial joint state after scripted setup."""
            current = robot.state
            current_q = vector(current.q).astype(float)
            lower = JOINT_MIN + args.joint_margin
            upper = JOINT_MAX - args.joint_margin
            if np.any(policy_start_q < lower) or np.any(policy_start_q > upper):
                raise RuntimeError("Recorded policy start violates guarded joint limits")
            measured_tcp = np.asarray(current.O_T_EE.matrix, dtype=float)
            joint7_current = np.asarray(robot.model.pose(Frame.Joint7, current).matrix, dtype=float)
            joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
            path_q = np.linspace(current_q, policy_start_q, num=41)
            path_xyz = np.asarray([
                (
                    np.asarray(
                        robot.model.pose(Frame.Joint7, q_step, current.F_T_EE, current.EE_T_K).matrix,
                        dtype=float,
                    )
                    @ joint7_to_tcp
                )[:3, 3]
                for q_step in path_q
            ])
            xyz_lower = np.asarray([args.min_x, args.min_y, args.min_z])
            xyz_upper = np.asarray([args.max_x, args.max_y, args.max_z])
            if np.any(path_xyz < xyz_lower) or np.any(path_xyz > xyz_upper):
                raise RuntimeError(
                    "Recorded-start reset leaves guarded Cartesian bounds: "
                    f"min={path_xyz.min(axis=0)}, max={path_xyz.max(axis=0)}"
                )
            print(
                f"[SETUP] resetting to demonstrated policy start; q_delta="
                f"{np.array2string(policy_start_q - current_q, precision=3)}"
            )
            robot.move(JointMotion(JointState(policy_start_q)))
            final_error = float(np.max(np.abs(vector(robot.state.q) - policy_start_q)))
            if final_error > 0.003:
                raise RuntimeError(f"Recorded-start reset did not converge: joint error={final_error:.4f}")

        def open_gripper() -> None:
            gripper.open(args.open_speed)

        def return_to_recorded_policy_start() -> None:
            """Use a raised Cartesian transit before the exact guarded joint reset."""
            current = robot.state
            current_tcp = np.asarray(current.O_T_EE.matrix, dtype=float)
            joint7_current = np.asarray(robot.model.pose(Frame.Joint7, current).matrix, dtype=float)
            joint7_to_tcp = np.linalg.solve(joint7_current, current_tcp)
            policy_start_tcp = (
                np.asarray(
                    robot.model.pose(
                        Frame.Joint7,
                        policy_start_q,
                        current.F_T_EE,
                        current.EE_T_K,
                    ).matrix,
                    dtype=float,
                )
                @ joint7_to_tcp
            )
            current_xyz = current_tcp[:3, 3]
            policy_start_xyz = policy_start_tcp[:3, 3]
            xyz_lower = np.asarray([args.min_x, args.min_y, args.min_z])
            xyz_upper = np.asarray([args.max_x, args.max_y, args.max_z])
            if np.any(current_xyz < xyz_lower) or np.any(current_xyz > xyz_upper):
                raise RuntimeError(f"Post-release TCP is outside guarded bounds: {current_xyz}")
            if np.any(policy_start_xyz < xyz_lower) or np.any(policy_start_xyz > xyz_upper):
                raise RuntimeError(
                    f"Recorded policy-start TCP is outside guarded bounds: {policy_start_xyz}"
                )
            return_travel_z = min(
                args.max_z,
                max(float(current_xyz[2]), float(policy_start_xyz[2]), transport_z)
                + args.travel_extra_clearance,
            )
            if return_travel_z <= max(float(current_xyz[2]), float(policy_start_xyz[2])):
                raise RuntimeError("No Cartesian clearance remains for the post-release reset")
            return_motion = MotionPlanner(
                robot,
                args.dynamics_factor,
                args.max_step,
                return_travel_z,
                args.settle_time,
            )
            print(
                "[RESET] returning from basket release to demonstrated policy start; "
                f"tcp_target={np.array2string(policy_start_xyz, precision=3)} "
                f"travel_z={return_travel_z:.3f}"
            )
            return_motion.travel_to(
                policy_start_xyz,
                float(policy_start_xyz[2]),
                "post-release return home",
            )
            reset_to_recorded_policy_start()

        def place_from_basket() -> None:
            print("[SETUP] picking cube from basket.")
            open_gripper()
            motion.travel_to(pad, transport_z, "setup basket pickup")
            motion.move_in_steps(np.asarray([pad[0], pad[1], basket_grasp_z]), "setup basket descend")
            accepted = gripper.grasp(args.grasp_width, args.grasp_speed, args.grasp_force, 0.01, 0.01)
            motion.move_in_steps(np.asarray([pad[0], pad[1], transport_z]), "setup basket lift")
            if not accepted or not gripper.is_grasped:
                raise RuntimeError("Basket pickup did not verify; target cell was not attempted")
            print("[SETUP] placing cube on held-out target.")
            motion.travel_to(target_xyz, hover_z, "setup cell hover")
            motion.move_in_steps(np.asarray([target_xyz[0], target_xyz[1], drop_z]), "setup cell drop")
            open_gripper()
            motion.move_in_steps(np.asarray([target_xyz[0], target_xyz[1], hover_z]), "setup cell retreat")
            motion.travel_to(home_xyz, float(home_xyz[2]), "setup return home")
            print(f"[SETUP] cube placed; policy start tcp={np.array2string(robot.state.O_T_EE.translation, precision=3)}")

        def skip_and_recycle_cell(return_home: bool = True) -> bool:
            """Return a still-ungrasped cube from the eval cell to the basket."""
            print("[SKIP] recovering cube: cell -> basket -> home.")
            open_gripper()
            motion.travel_to(target_xyz, hover_z, "skip recovery cell hover")
            motion.move_in_steps(
                np.asarray([target_xyz[0], target_xyz[1], grasp_z]),
                "skip recovery cell descend",
            )
            accepted = gripper.grasp(
                args.grasp_width, args.grasp_speed, args.grasp_force, 0.01, 0.01
            )
            motion.move_in_steps(
                np.asarray([target_xyz[0], target_xyz[1], transport_z]),
                "skip recovery cell lift",
            )
            if not accepted or not gripper.is_grasped:
                print("[SKIP] recovery grasp did not verify; stopping without basket transport.")
                return False
            motion.travel_to(pad, transport_z, "skip recovery basket hover")
            motion.move_in_steps(
                np.asarray([pad[0], pad[1], basket_release_z]),
                "skip recovery basket descend",
            )
            open_gripper()
            motion.move_in_steps(
                np.asarray([pad[0], pad[1], transport_z]),
                "skip recovery basket retreat",
            )
            if return_home:
                motion.travel_to(home_xyz, float(home_xyz[2]), "skip recovery home")
                print("[SKIP] cube returned to basket; arm is home.")
            else:
                print("[SKIP] cube returned to basket; awaiting demonstrated-start reset.")
            return True

        if args.recover_only:
            recovered = skip_and_recycle_cell(return_home=False)
            if recovered:
                reset_to_recorded_policy_start()
            result = {
                "status": "recovered_to_basket" if recovered else "recovery_grasp_failed",
                "eval_id": args.eval_id,
                "target": target,
            }
            return

        if args.reset_only:
            return_to_recorded_policy_start()
            result = {
                "status": "post_release_reset_complete",
                "eval_id": args.eval_id,
                "target": target,
                "policy_start_episode": start_episode.name,
            }
            print("[RESET] post-release trial turnover complete; arm is at the demonstrated policy start.")
            return

        if not args.manual_target and not args.manual_live:
            place_from_basket()
            reset_to_recorded_policy_start()
        elif args.manual_live:
            reset_to_recorded_policy_start()
        else:
            print("[MANUAL] Confirm cube is on the measured target and arm is at the recorded policy-start pose.")
        if args.manual_live:
            print("[MANUAL] Cube placement remains operator-owned; arm is at the recorded policy-start pose.")
        if args.setup_only or args.manual_target or args.manual_live:
            result = {
                "status": "setup_complete_for_streaming",
                "eval_id": args.eval_id,
                "target": target,
                "policy_start_episode": start_episode.name,
            }
            print("[SETUP] complete. Cube is on target and arm is at demonstrated policy start.")
            return
        client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
        exterior = start_camera(args.exterior_serial, args.camera_width, args.camera_height, args.camera_fps)
        wrist = start_camera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps)
        for _ in range(15):
            read_rgb(exterior)
            read_rgb(wrist)

        dt = 1.0 / args.control_hz
        waypoint_ms = max(1, round(1000.0 / args.control_hz))
        xyz_lower = np.asarray([args.min_x, args.min_y, args.min_z])
        xyz_upper = np.asarray([args.max_x, args.max_y, args.max_z])
        joint_lower = JOINT_MIN + args.joint_margin
        joint_upper = JOINT_MAX - args.joint_margin
        holding = False

        with log_path.open("w") as log_file:
            for chunk in range(1, args.max_chunks + 1):
                exterior_rgb = read_rgb(exterior)
                wrist_rgb = read_rgb(wrist)
                state = robot.state
                q = vector(state.q).astype(float)
                closedness = float(np.clip(1.0 - float(gripper.state.width) / 0.08, 0.0, 1.0))
                observation = {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(exterior_rgb, 224, 224)),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_rgb, 224, 224)),
                    "observation/joint_position": q.astype(np.float32),
                    "observation/gripper_position": np.asarray([closedness], dtype=np.float32),
                    "prompt": args.prompt,
                }
                began = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latency = time.perf_counter() - began
                if actions.shape != (16, 8) or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid policy actions: {actions.shape}")
                gripper_plan = actions[: args.execute_steps, 7]
                close_requested = float(gripper_plan.max()) >= args.close_threshold
                release_requested = holding and float(gripper_plan.max()) <= args.release_threshold
                if release_requested:
                    decision = input(
                        "[POLICY] Release requested. Type RELEASE only if the cube is visibly over the basket; "
                        "anything else leaves the robot stationary: "
                    ).strip().upper()
                    if decision == "RELEASE":
                        open_gripper()
                        motion.travel_to(home_xyz, float(home_xyz[2]), "policy release home")
                        assessment = input("[EVAL] Cube released. Type PASS or FAIL for this trial: ").strip().upper()
                        result = {
                            "status": "released_homed",
                            "assessment": assessment if assessment in {"PASS", "FAIL"} else "UNSCORED",
                            "eval_id": args.eval_id,
                            "chunk": chunk,
                            "log": str(log_path),
                        }
                    else:
                        result = {"status": "release_requested", "eval_id": args.eval_id, "chunk": chunk, "log": str(log_path)}
                        print("[POLICY] Release not approved. Arm and cube remain stationary.")
                    break
                if close_requested and not holding:
                    decision = input(
                        "[POLICY] Closure requested. Type CLOSE to execute this descent + grasp and continue; "
                        "type SKIP to return cube to basket and home: "
                    ).strip().upper()
                    if decision == "SKIP":
                        recovered = skip_and_recycle_cell()
                        result = {
                            "status": "skipped_recycled" if recovered else "skip_recovery_failed",
                            "eval_id": args.eval_id,
                            "chunk": chunk,
                            "log": str(log_path),
                        }
                        break
                    if decision != "CLOSE":
                        result = {
                            "status": "grasp_declined",
                            "eval_id": args.eval_id,
                            "chunk": chunk,
                            "log": str(log_path),
                        }
                        print("[POLICY] Closure declined; stopping without gripper motion.")
                        break

                raw_velocities = actions[: args.execute_steps, :7]
                velocities = np.clip(raw_velocities, -caps, caps)
                predicted_q = q + np.cumsum(velocities * dt * args.guard_displacement_scale, axis=0)
                if np.any(predicted_q < joint_lower) or np.any(predicted_q > joint_upper):
                    raise RuntimeError("Predicted policy chunk violates guarded joint limits")
                tcp = np.asarray(state.O_T_EE.matrix, dtype=float)
                joint7 = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=float)
                joint7_to_tcp = np.linalg.solve(joint7, tcp)
                predicted_xyz = np.asarray([
                    (np.asarray(robot.model.pose(Frame.Joint7, q_step, state.F_T_EE, state.EE_T_K).matrix, dtype=float) @ joint7_to_tcp)[:3, 3]
                    for q_step in predicted_q
                ])
                if np.any(predicted_xyz < xyz_lower) or np.any(predicted_xyz > xyz_upper):
                    raise RuntimeError(
                        "Predicted policy chunk leaves guarded Cartesian bounds: "
                        f"min={predicted_xyz.min(axis=0)}, max={predicted_xyz.max(axis=0)}"
                    )
                record = {
                    "chunk": chunk,
                    "latency_s": latency,
                    "q": q.tolist(),
                    "tcp": tcp[:3, 3].tolist(),
                    "raw_peak": float(np.max(np.abs(raw_velocities))),
                    "commanded_peak": float(np.max(np.abs(velocities))),
                    "gripper_plan": gripper_plan.tolist(),
                    "holding": holding,
                    "predicted_end": predicted_xyz[-1].tolist(),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(
                    f"[POLICY {chunk:02d}/{args.max_chunks}] latency={latency:.3f}s "
                    f"tcp={np.array2string(tcp[:3, 3], precision=3)} "
                    f"end={np.array2string(predicted_xyz[-1], precision=3)} "
                    f"gripper={gripper_plan.min():.3f}..{gripper_plan.max():.3f} holding={holding}"
                )
                waypoints = [JointVelocityWaypoint(v, minimum_time=Duration(waypoint_ms)) for v in velocities]
                robot.move(JointVelocityWaypointMotion(waypoints))
                end_xyz = np.asarray(robot.state.O_T_EE.translation, dtype=float)
                if np.any(end_xyz < xyz_lower) or np.any(end_xyz > xyz_upper):
                    raise RuntimeError(f"Actual TCP outside guarded bounds: {end_xyz}")
                if close_requested and not holding:
                    accepted = gripper.grasp(args.grasp_width, args.grasp_speed, args.grasp_force, 0.01, 0.01)
                    holding = bool(accepted and gripper.is_grasped)
                    print(f"[GRIPPER] grasp accepted={accepted} is_grasped={gripper.is_grasped} width={gripper.state.width:.4f}")
                    if not holding:
                        result = {"status": "grasp_failed", "eval_id": args.eval_id, "chunk": chunk, "log": str(log_path)}
                        print("[POLICY] Grasp did not verify. Stopping; no further policy motion.")
                        break
            else:
                result = {"status": "max_chunks_reached", "eval_id": args.eval_id, "log": str(log_path)}
                print("[POLICY] Maximum chunks reached. Stopping without gripper release.")
    except BaseException as exc:
        if result.get("status") == "started":
            result = {
                "status": "aborted_by_guard_or_exception",
                "eval_id": args.eval_id,
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path),
            }
        stop_robot()
        raise
    finally:
        if exterior is not None:
            exterior.stop()
        if wrist is not None:
            wrist.stop()
        result["finished_at"] = datetime.now().isoformat()
        result["log"] = str(log_path)
        log_path.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[EVAL] result={result['status']} result_log={log_path.with_suffix('.result.json')}")


if __name__ == "__main__":
    main()
