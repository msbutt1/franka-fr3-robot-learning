#!/usr/bin/env python3
"""STAGE A: grasp mechanics test for the real FR3 — hover, place, grasp, lift, verify,
lower, release. NO basket transport yet (needs basket_rim probed first).

For each grid cell: arm hovers above the spot -> you place the cube (short 4x4cm end
facing between the gripper fingers -- check finger orientation while it's hovering,
BEFORE placing) -> confirm -> arm descends to the cube's vertical center -> grasp() ->
reports success/fail -> lifts a few cm -> checks is_grasped -> lowers back down ->
releases -> retreats to home. Repeat for the next cell.

This validates grasp width/speed/force tuning and cube-orientation placement with
zero risk near the basket (basket transport is Stage B, added once basket_rim is
probed).

Usage:
    python test_grasp_cycle.py --robot_ip 172.16.0.2 --points probed_points.json \\
        --nx 3 --ny 3 --grasp_width 0.04 --cube_height 0.04
"""
import argparse
import itertools
import json
import signal
import sys
from pathlib import Path

import numpy as np
from franka_motion import (
    DEFAULT_FRANKA_HAND_TCP_OFFSET,
    MotionPlanner,
    assert_safe_flange_z,
    clear_robot_errors,
    compute_travel_z_floor,
    configure_collision_behavior,
    flange_z_for_tcp_z,
    reset_dynamics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--points", type=str, default="probed_points.json")
parser.add_argument("--nx", type=int, default=3)
parser.add_argument("--ny", type=int, default=3)
parser.add_argument("--basket_w", type=float, default=0.154)
parser.add_argument("--basket_h", type=float, default=0.134)
parser.add_argument("--basket_margin", type=float, default=0.04)
parser.add_argument("--hover_clearance", type=float, default=0.12,
                    help="Height above table to hover/hold before+after grasp, meters.")
parser.add_argument("--cube_height", type=float, default=0.04,
                    help="Cube height, meters — grasp point is table_z + cube_height/2.")
parser.add_argument("--grasp_lowering", type=float, default=0.0,
                    help="Lower the grasp target below cube_height/2 by this many meters.")
parser.add_argument("--grasp_width", type=float, default=0.04,
                    help="Target gripper closing width, meters (the cube's SHORT face).")
parser.add_argument("--grasp_speed", type=float, default=0.05, help="m/s")
parser.add_argument("--grasp_force", type=float, default=15.0,
                    help="N — start conservative; raise if slipping, lower if the cube is soft/fragile.")
parser.add_argument("--grasp_epsilon", type=float, default=0.01, help="+/- tolerance band, meters.")
parser.add_argument("--open_speed", type=float, default=0.05, help="m/s")
parser.add_argument("--dynamics_factor", type=float, default=0.03,
                    help="Fraction of max dynamics. Lowered from 0.05 after FR3 reflex aborts on long moves.")
parser.add_argument("--lift_check_height", type=float, default=0.05,
                    help="How far to lift after grasp to verify the hold, meters.")
parser.add_argument("--max_step", type=float, default=0.03,
                    help="Maximum Cartesian segment length, meters. Use 0 for one smooth move per leg.")
parser.add_argument("--travel_extra_clearance", type=float, default=0.08,
                    help="Extra Z clearance above the highest hover/home pose for long horizontal travel, meters.")
parser.add_argument("--travel_above_home", action="store_true", default=False,
                    help="Also require the travel plane to be above the startup home Z.")
parser.add_argument("--tcp_offset", type=float, default=DEFAULT_FRANKA_HAND_TCP_OFFSET,
                    help="Meters from commanded flange/control frame down to fingertip TCP along table Z.")
parser.add_argument("--min_flange_z", type=float, default=0.03,
                    help="Abort if any commanded TCP/control-frame Z drops below this value.")
parser.add_argument("--pause_before_hover_descent", action="store_true", default=True,
                    help="Pause above each target before descending to hover.")
parser.add_argument("--no_pause_before_hover_descent", action="store_false", dest="pause_before_hover_descent",
                    help="Do not pause above each target before descending to hover.")
parser.add_argument("--skip_cell", type=int, action="append", default=[],
                    help="Zero-based generated cell index to skip. Can be passed multiple times.")
args = parser.parse_args()

pts = json.loads(Path(args.points).read_text())
required = ["bottom_left", "bottom_right", "top_left", "top_right", "pad_center"]
missing = [k for k in required if k not in pts]
if missing:
    raise SystemExit(f"probed_points.json is missing: {missing}. Probe these labels first.")

BL = np.array(pts["bottom_left"]); BR = np.array(pts["bottom_right"])
TL = np.array(pts["top_left"]); TR = np.array(pts["top_right"]); PAD = np.array(pts["pad_center"])


def bilinear(u: float, v: float) -> np.ndarray:
    near = (1 - v) * BR + v * BL
    far = (1 - v) * TR + v * TL
    return (1 - u) * near + u * far


half_w = args.basket_w / 2 + args.basket_margin
half_h = args.basket_h / 2 + args.basket_margin
cells = []
for i, j in itertools.product(range(args.nx), range(args.ny)):
    u = (i + 0.5) / args.nx
    v = (j + 0.5) / args.ny
    xyz = bilinear(u, v)
    if abs(xyz[0] - PAD[0]) < half_w and abs(xyz[1] - PAD[1]) < half_h:
        continue
    cells.append(xyz)

if args.skip_cell:
    skip = set(args.skip_cell)
    before_skip = len(cells)
    cells = [c for k, c in enumerate(cells) if k not in skip]
    print(f"[grid] skipped cells by request: {sorted(skip)} ({before_skip - len(cells)} matched)")

print(f"[grid] {len(cells)} cells (of {args.nx * args.ny}; excluded/skipped cells are not attempted)")
print(f"[grasp] target_width={args.grasp_width} speed={args.grasp_speed} force={args.grasp_force} "
      f"epsilon=+/-{args.grasp_epsilon} lowering={args.grasp_lowering}")
print("[reminder] orient the cube so its SHORT (4x4cm) face is between the gripper fingers, "
      "8cm length running parallel to the line between the fingertips.\n")

input("[SAFETY] Confirm workspace is clear. Press Enter to connect and begin...")

from franky import Robot, Gripper  # noqa: E402

robot = Robot(args.robot_ip)
robot.relative_dynamics_factor = args.dynamics_factor
gripper = Gripper(args.robot_ip)


def _sigint_handler(signum, frame):
    print("\n[KILLSWITCH] Ctrl+C received — stopping robot motion now.")
    try:
        robot.stop()
        print("[KILLSWITCH] robot.stop() called.")
    except Exception as e:
        print(f"[KILLSWITCH] robot.stop() raised: {e} (use the physical E-stop if the arm keeps moving)")
    sys.exit(1)


signal.signal(signal.SIGINT, _sigint_handler)

clear_robot_errors(robot)
configure_collision_behavior(robot)

print("[gripper] homing (calibrating max width) ...")
gripper.homing()
print(f"[gripper] ready. max_width={gripper.max_width:.4f}")
if args.grasp_width >= gripper.max_width:
    raise SystemExit(f"grasp_width {args.grasp_width} >= gripper max_width {gripper.max_width} — check units/cube size.")

home_xyz = np.array(robot.current_pose.end_effector_pose.translation)
travel_z_floor = compute_travel_z_floor(
    home_xyz,
    [flange_z_for_tcp_z(float(cell[2] + args.hover_clearance), args.tcp_offset) for cell in cells],
    args.travel_extra_clearance,
    include_home_z=args.travel_above_home,
)
print(f"[robot] home pose: x={home_xyz[0]:+.4f} y={home_xyz[1]:+.4f} z={home_xyz[2]:+.4f}")
print(f"[robot] tcp_offset={args.tcp_offset:.4f}  min_flange_z={args.min_flange_z:.4f}")
print(f"[robot] travel_z_floor={travel_z_floor:+.4f} "
      f"(extra clearance {args.travel_extra_clearance:.3f}; travel_above_home={args.travel_above_home})\n")

motion = MotionPlanner(robot, args.dynamics_factor, args.max_step, travel_z_floor)


results = []
for k, cell_xyz in enumerate(cells):
    table_z = float(cell_xyz[2])
    hover_tcp_z = table_z + args.hover_clearance
    grasp_tcp_z = table_z + args.cube_height / 2 - args.grasp_lowering
    hover_flange_z = flange_z_for_tcp_z(hover_tcp_z, args.tcp_offset)
    grasp_flange_z = flange_z_for_tcp_z(grasp_tcp_z, args.tcp_offset)
    assert_safe_flange_z("hover", hover_flange_z, args.min_flange_z)
    assert_safe_flange_z("grasp", grasp_flange_z, args.min_flange_z)

    print(f"=== cell {k+1}/{len(cells)}: x={cell_xyz[0]:+.3f} y={cell_xyz[1]:+.3f} "
          f"table_z={table_z:+.3f} grasp_tcp_z={grasp_tcp_z:+.3f} "
          f"grasp_flange_z={grasp_flange_z:+.3f} ===")
    print(f"  [safety] current_flange_z={motion.current_xyz()[2]:+.4f} "
          f"hover_flange_z={hover_flange_z:+.4f} grasp_flange_z={grasp_flange_z:+.4f}")
    input("  press Enter to hover above this cell...")
    reset_dynamics(robot, args.dynamics_factor)
    if args.pause_before_hover_descent:
        motion.travel_to_with_descent_pause(
            cell_xyz,
            hover_flange_z,
            "to hover",
            prompt=f"  arm is above the target at the travel plane. Press Enter to descend to flange_z={hover_flange_z:+.3f}...",
        )
    else:
        motion.travel_to(cell_xyz, hover_flange_z, "to hover")

    print("  [check] look at the gripper fingers now — note their orientation.")
    ans = input("  place the cube (short face between fingers), then Enter to grasp ('s'=skip cell): ")
    if ans.strip().lower() == "s":
        motion.travel_to(home_xyz, home_xyz[2], "retreat home")
        print("  [skip] retreated home.\n")
        continue

    gripper.open(args.open_speed)
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z]), "descend to grasp")

    grasped_ok = gripper.grasp(args.grasp_width, args.grasp_speed, args.grasp_force,
                               epsilon_inner=args.grasp_epsilon, epsilon_outer=args.grasp_epsilon)
    print(f"  [grasp] grasp() returned: {grasped_ok}   (width now: {gripper.width:.4f})")

    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z + args.lift_check_height]), "lift & verify")
    still_holding = gripper.is_grasped
    print(f"  [verify] after lifting {args.lift_check_height*100:.0f}cm: is_grasped={still_holding}")

    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z]), "lower back down")
    gripper.open(args.open_speed)
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], hover_flange_z]), "retreat to hover")
    motion.travel_to(home_xyz, home_xyz[2], "retreat home")

    ok = bool(grasped_ok and still_holding)
    results.append(ok)
    print(f"  [done] cell result: {'SUCCESS' if ok else 'FAIL'}. back at home.\n")

n = len(results)
s = sum(results)
print(f"=== STAGE A COMPLETE: {s}/{n} grasp cycles succeeded ===" if n else "=== no cells attempted ===")
