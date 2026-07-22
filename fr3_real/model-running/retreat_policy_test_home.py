#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/retreat_policy_test_home.py --help
"""Guardedly retreat a stationary policy-test arm to a recorded safe home pose."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from franka_motion import MotionPlanner, clear_robot_errors, configure_collision_behavior
from repo_paths import RECORDINGS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--record_dir", default=str(RECORDINGS_DIR / "droid_raw_full_v3"))
    parser.add_argument("--episode", required=True, help="Successful episode whose recorded home pose will be used.")
    parser.add_argument("--travel_z", type=float, default=0.20)
    parser.add_argument("--dynamics_factor", type=float, default=0.05)
    parser.add_argument("--max_step", type=float, default=0.03)
    args = parser.parse_args()

    metadata_path = Path(args.record_dir) / args.episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    home = np.asarray(metadata["home_xyz"], dtype=float)
    if home.shape != (3,):
        raise SystemExit(f"Invalid home_xyz in {metadata_path}")

    from franky import Robot

    print("[RETREAT] Arm-only recovery: rise vertically, travel above the table, then return home.")
    print(f"[RETREAT] recorded home={np.array2string(home, precision=4)} travel_z={args.travel_z:.3f}")
    input("Confirm the path above the table is clear and E-stop is reachable, then press Enter... ")

    robot = Robot(args.robot_ip)
    clear_robot_errors(robot)
    configure_collision_behavior(robot)
    robot.relative_dynamics_factor = args.dynamics_factor

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    start = np.asarray(robot.current_pose.end_effector_pose.translation, dtype=float)
    print(f"[RETREAT] current tcp={np.array2string(start, precision=4)}")
    if input("Type RETREAT exactly to move arm home, anything else aborts: ") != "RETREAT":
        print("[RETREAT] aborted without motion.")
        return

    planner = MotionPlanner(
        robot=robot,
        dynamics_factor=args.dynamics_factor,
        max_step=args.max_step,
        travel_z_floor=args.travel_z,
        settle_time=0.30,
        retry_policy="retry_slower",
    )
    try:
        planner.travel_to(home, float(home[2]), "policy-test retreat")
    except BaseException:
        robot.stop()
        raise
    final = np.asarray(robot.current_pose.end_effector_pose.translation, dtype=float)
    print(f"[RETREAT] final tcp={np.array2string(final, precision=4)}")


if __name__ == "__main__":
    main()
