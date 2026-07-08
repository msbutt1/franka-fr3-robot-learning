#!/usr/bin/env python3
"""Read-only check for the franky Cartesian frame.

Put the robot in hand-guiding mode and place the gripper fingertips/probe at a
known physical point, then run this script. If the reported Z matches the probed
table Z, use --tcp_offset 0. If it is roughly table_z + hand length, your setup
is reporting flange/control-frame height and you need a positive offset.
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--points", type=str, default="probed_points.json")
parser.add_argument("--label", type=str, default="table_z",
                    help="Point label from probed_points.json to compare against.")
args = parser.parse_args()

pts = json.loads(Path(args.points).read_text())
if args.label not in pts:
    raise SystemExit(f"{args.points} does not contain label '{args.label}'")

print("[SAFETY] This is read-only. Put the robot in hand-guiding mode and place the fingertips/probe")
input(f"          on the physical point '{args.label}', then press Enter to read current_pose...")

from franky import Robot  # noqa: E402

robot = Robot(args.robot_ip)
reported = np.array(robot.current_pose.end_effector_pose.translation, dtype=float)
probed = np.array(pts[args.label], dtype=float)
delta = reported - probed

print(f"[probed {args.label}]  x={probed[0]:+.4f} y={probed[1]:+.4f} z={probed[2]:+.4f}")
print(f"[current_pose]      x={reported[0]:+.4f} y={reported[1]:+.4f} z={reported[2]:+.4f}")
print(f"[delta]             x={delta[0]:+.4f} y={delta[1]:+.4f} z={delta[2]:+.4f}")

print("\nInterpretation:")
print("  delta_z near 0.000 -> use --tcp_offset 0.0")
print("  delta_z near 0.10  -> current_pose is flange/control-frame-like; use that delta as --tcp_offset")
