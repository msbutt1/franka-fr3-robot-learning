#!/usr/bin/env python3
"""First-contact connection test for the real FR3 — NO MOTION, read-only.

Confirms franky can talk to the robot over FCI before you run anything that moves it.

Prereqs (one-time, on this machine):
    pip install franky-control

On the robot (Franka Desk):
    - joints unlocked (brakes released)
    - FCI activated

Usage:
    python test_connection.py --robot_ip 172.16.0.2
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True,
                    help="IP of the robot's control box (same one you use for the Desk URL).")
args = parser.parse_args()

print(f"[test] importing franky...")
from franky import Robot  # noqa: E402

print(f"[test] connecting to robot at {args.robot_ip} ...")
robot = Robot(args.robot_ip)
print("[test] connected.")

pose = robot.current_pose.end_effector_pose
xyz = pose.translation
print("\n=== READ-ONLY STATE (no motion commanded) ===")
print(f"  end-effector position (base frame, meters): x={xyz[0]:+.4f}  y={xyz[1]:+.4f}  z={xyz[2]:+.4f}")
print(f"  end-effector orientation (quaternion):       {pose.quaternion}")
try:
    print(f"  current joint positions (rad):               {list(robot.current_joint_positions)}")
except Exception as e:  # keep going even if this particular attribute name differs by franky version
    print(f"  (joint positions not read: {e})")

print("\n[test] SUCCESS — franky can talk to the robot. Nothing moved.")
