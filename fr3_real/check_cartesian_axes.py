#!/usr/bin/env python3
"""Tiny FR3 Cartesian-axis sanity check.

This intentionally commands only very small relative moves from the current pose.
Use it before table-level scripts to verify that +Z really moves away from the
table and that +X/+Y match the printed base-frame pose convention.
"""

import argparse
import signal
import sys

import numpy as np
from franka_motion import clear_robot_errors, configure_collision_behavior

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--step", type=float, default=0.005,
                    help="Test move size in meters. Default is 5 mm.")
parser.add_argument("--dynamics_factor", type=float, default=0.01)
args = parser.parse_args()

input("[SAFETY] Put the arm high and clear. Press Enter to connect...")

from franky import Affine, CartesianMotion, ReferenceType, Robot, RobotPose  # noqa: E402

robot = Robot(args.robot_ip)
robot.relative_dynamics_factor = args.dynamics_factor


def _sigint_handler(signum, frame):
    print("\n[KILLSWITCH] Ctrl+C received — stopping robot motion now.")
    try:
        robot.stop()
    finally:
        sys.exit(1)


signal.signal(signal.SIGINT, _sigint_handler)

clear_robot_errors(robot)
configure_collision_behavior(robot)


def pose_xyz():
    return np.array(robot.current_pose.end_effector_pose.translation, dtype=float)


def move_delta(delta, label):
    current_pose = robot.current_pose
    current_ee = current_pose.end_effector_pose
    before = np.array(current_ee.translation, dtype=float)
    target = before + delta
    target_pose = RobotPose(
        Affine(target, np.array(current_ee.quaternion, dtype=float)),
        current_pose.elbow_state,
    )
    print(f"[{label}] before=({before[0]:+.4f},{before[1]:+.4f},{before[2]:+.4f}) "
          f"base_frame_delta=({delta[0]:+.4f},{delta[1]:+.4f},{delta[2]:+.4f})")
    input(f"Press Enter to command {label}...")
    robot.move(CartesianMotion(target_pose, ReferenceType.Absolute))
    after = pose_xyz()
    print(f"[{label}] after =({after[0]:+.4f},{after[1]:+.4f},{after[2]:+.4f}) "
          f"observed_delta=({after[0]-before[0]:+.4f},{after[1]-before[1]:+.4f},{after[2]-before[2]:+.4f})\n")


s = args.step
move_delta(np.array([0.0, 0.0, +s]), "+Z")
move_delta(np.array([0.0, 0.0, -s]), "-Z return")
move_delta(np.array([+s, 0.0, 0.0]), "+X")
move_delta(np.array([-s, 0.0, 0.0]), "-X return")
move_delta(np.array([0.0, +s, 0.0]), "+Y")
move_delta(np.array([0.0, -s, 0.0]), "-Y return")

print("=== AXIS CHECK COMPLETE ===")
