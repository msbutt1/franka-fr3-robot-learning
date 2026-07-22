#!/usr/bin/env python3
# Usage: from fr3_real/, run: python franka-tests/probe_points.py --help
"""Free-drive probe tool for the real FR3 — records labeled base-frame coordinates to files.

Usage (on the machine connected to the robot, with `pip install franky-control`):
    python probe_points.py --robot_ip 172.16.0.2
    python probe_points.py --robot_ip 172.16.0.2 --out_dir ./calibration

Workflow per point:
  1. Put the arm in guide mode (grip the wrist buttons / enable hand-guiding in Desk).
  2. Drag the gripper TIP to the point you want to record (pad corner, table surface, grid cell...).
  3. Type a label (e.g. "pad_center", "table_z", "cell_x044_yn010") and press Enter.
     - press Enter with an empty label to re-read without saving
     - type "q" to finish
Every point is appended immediately (crash-safe) to:
  <out_dir>/probed_points.md    — human-readable table for your notes
  <out_dir>/probed_points.json  — machine-readable, imported by the demo recorder
"""
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, default="172.16.0.2")
parser.add_argument("--out_dir", type=str, default=".")
args = parser.parse_args()

from franky import Robot  # noqa: E402  (import after argparse so --help works without franky)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
md_path = out / "probed_points.md"
json_path = out / "probed_points.json"

# Load existing points so re-runs append instead of clobbering.
points: dict[str, list[float]] = {}
if json_path.exists():
    points = json.loads(json_path.read_text())
    print(f"[probe] loaded {len(points)} existing points from {json_path}")

if not md_path.exists():
    md_path.write_text(
        "# FR3 probed points (robot base frame, meters)\n\n"
        f"robot_ip: {args.robot_ip}\n\n"
        "| label | x | y | z | probed at |\n"
        "|---|---|---|---|---|\n"
    )

robot = Robot(args.robot_ip)
print("[probe] connected. Put the arm in GUIDE MODE and drag the tip to each point.")
print("[probe] label + Enter = save   |   empty Enter = peek   |   q = quit\n")

while True:
    label = input("label ('q' to quit, empty to peek): ").strip()
    if label.lower() == "q":
        break
    p = robot.current_pose.end_effector_pose.translation
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    print(f"    x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")
    if not label:
        continue
    if label in points:
        print(f"    [overwriting previous '{label}': {points[label]}]")
    points[label] = [round(x, 4), round(y, 4), round(z, 4)]
    # append to md + rewrite json atomically after every point (crash-safe)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with md_path.open("a") as f:
        f.write(f"| {label} | {x:+.4f} | {y:+.4f} | {z:+.4f} | {stamp} |\n")
    json_path.write_text(json.dumps(points, indent=2))
    print(f"    saved → {md_path.name}, {json_path.name}  ({len(points)} points total)")

print(f"\n[probe] done. {len(points)} points in {json_path}")
