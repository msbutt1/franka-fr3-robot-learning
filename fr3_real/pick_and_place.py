#!/usr/bin/env python3
"""STAGE B: full pick-and-place-into-basket cycle for the real FR3.

Per cell: hover -> place cube (short 4x4cm face between the fingers) -> confirm ->
retreat home -> automatically return, descend, grasp(), verify by lifting ->
[only if grasped] transport over the basket rim -> descend into the basket ->
release -> retreat home. If the grasp fails, the cube is never carried toward
the basket -- the arm just retreats home and the cell is marked FAIL.

Every move is computed from a FRESH read of the robot's current pose (not a
running tally), so small execution drift never compounds across the sequence.
Orientation is never commanded -- only translations, so there is no risk from
an unverified rotation.

Usage:
    python pick_and_place.py --robot_ip 172.16.0.2 --points probed_points.json \\
        --nx 3 --ny 3
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
from grid_utils import basket_polygon_from_points, inside_basket_exclusion

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--points", type=str, default="probed_points.json")
parser.add_argument("--cells_json", type=str, default=None,
                    help="Optional JSON exported by filter_grid_tracker.py. If set, use those cells instead of generating a grid.")
parser.add_argument("--nx", type=int, default=3)
parser.add_argument("--ny", type=int, default=3)
parser.add_argument("--basket_w", type=float, default=0.154)
parser.add_argument("--basket_h", type=float, default=0.134)
parser.add_argument("--basket_margin", type=float, default=0.04)
parser.add_argument("--hover_clearance", type=float, default=0.12,
                    help="Height above table to hover over a pick cell, meters.")
parser.add_argument("--cube_height", type=float, default=0.04)
parser.add_argument("--grasp_lowering", type=float, default=0.0,
                    help="Lower the grasp target below cube_height/2 by this many meters.")
parser.add_argument("--grasp_width", type=float, default=0.04)
parser.add_argument("--grasp_speed", type=float, default=0.05)
parser.add_argument("--grasp_force", type=float, default=15.0)
parser.add_argument("--grasp_epsilon", type=float, default=0.01)
parser.add_argument("--open_speed", type=float, default=0.05)
parser.add_argument("--dynamics_factor", type=float, default=0.03)
parser.add_argument("--lift_check_height", type=float, default=0.05,
                    help="How far to lift after grasp to verify the hold, meters.")
parser.add_argument("--transport_clearance", type=float, default=0.05,
                    help="Height ABOVE basket_rim to fly over the wall at, meters.")
parser.add_argument("--release_clearance", type=float, default=0.02,
                    help="Height ABOVE pad_center to descend to before releasing, meters.")
parser.add_argument("--max_step", type=float, default=0.03,
                    help="Maximum Cartesian segment length, meters. Use 0 for one smooth move per leg.")
parser.add_argument("--travel_extra_clearance", type=float, default=0.08,
                    help="Extra Z clearance above the highest hover/transport/home pose for long travel, meters.")
parser.add_argument("--travel_above_home", action="store_true", default=False,
                    help="Also require the travel plane to be above the startup home Z.")
parser.add_argument("--tcp_offset", type=float, default=DEFAULT_FRANKA_HAND_TCP_OFFSET,
                    help="Meters from commanded flange/control frame down to fingertip TCP along table Z.")
parser.add_argument("--min_flange_z", type=float, default=0.03,
                    help="Abort if any commanded TCP/control-frame Z drops below this value.")
parser.add_argument("--pause_before_hover_descent", action="store_true", default=True,
                    help="Pause above each pick target before descending to hover.")
parser.add_argument("--no_pause_before_hover_descent", action="store_false", dest="pause_before_hover_descent",
                    help="Do not pause above each pick target before descending to hover.")
parser.add_argument("--skip_cell", type=int, action="append", default=[],
                    help="Zero-based generated cell index to skip. Can be passed multiple times.")
parser.add_argument("--max_cells", type=int, default=None,
                    help="Keep at most this many cells, sampled evenly across the generated grid.")
parser.add_argument("--skip_selected_cell", type=int, action="append", default=[],
                    help="Zero-based cell index to skip after --max_cells selection.")
parser.add_argument("--start_cell", type=int, default=0,
                    help="Zero-based cell index to start from after filtering.")
parser.add_argument("--end_cell", type=int, default=None,
                    help="Zero-based cell index to stop at after filtering, inclusive.")
args = parser.parse_args()
if args.max_cells is not None and args.max_cells <= 0:
    raise SystemExit("--max_cells must be positive.")

pts = json.loads(Path(args.points).read_text())
required = ["bottom_left", "bottom_right", "top_left", "top_right", "pad_center", "basket_rim"]
missing = [k for k in required if k not in pts]
if missing:
    raise SystemExit(f"probed_points.json is missing: {missing}. Probe these labels first.")

BL = np.array(pts["bottom_left"]); BR = np.array(pts["bottom_right"])
TL = np.array(pts["top_left"]); TR = np.array(pts["top_right"])
PAD = np.array(pts["pad_center"]); RIM = np.array(pts["basket_rim"])
basket_polygon_xy = basket_polygon_from_points(pts)


def bilinear(u: float, v: float) -> np.ndarray:
    near = (1 - v) * BR + v * BL
    far = (1 - v) * TR + v * TL
    return (1 - u) * near + u * far


print("[grid] basket exclusion: "
      + ("probed corner polygon" if basket_polygon_xy is not None else "pad_center rectangle"))
if args.cells_json:
    cell_records = json.loads(Path(args.cells_json).read_text())["cells"]
    cells = [np.array([cell["x"], cell["y"], cell["table_z"]], dtype=float) for cell in cell_records]
    print(f"[grid] loaded {len(cells)} cells from {args.cells_json}")
else:
    cells = []
    for i, j in itertools.product(range(args.nx), range(args.ny)):
        u = (i + 0.5) / args.nx
        v = (j + 0.5) / args.ny
        xyz = bilinear(u, v)
        if inside_basket_exclusion(xyz, PAD, args.basket_margin, args.basket_w, args.basket_h, basket_polygon_xy):
            continue
        cells.append(xyz)

if args.skip_cell:
    skip = set(args.skip_cell)
    before_skip = len(cells)
    cells = [c for k, c in enumerate(cells) if k not in skip]
    print(f"[grid] skipped cells by request: {sorted(skip)} ({before_skip - len(cells)} matched)")

if args.max_cells is not None and len(cells) > args.max_cells:
    keep = np.linspace(0, len(cells) - 1, args.max_cells, dtype=int)
    cells = [cells[k] for k in keep]
    print(f"[grid] limited to {args.max_cells} evenly sampled cells")

if args.skip_selected_cell:
    skip = set(args.skip_selected_cell)
    before_skip = len(cells)
    cells = [c for k, c in enumerate(cells) if k not in skip]
    print(f"[grid] skipped selected cells by request: {sorted(skip)} ({before_skip - len(cells)} matched)")

if args.start_cell or args.end_cell is not None:
    end = len(cells) - 1 if args.end_cell is None else args.end_cell
    cells = cells[args.start_cell:end + 1]
    print(f"[grid] running cell range {args.start_cell}..{end}")

transport_tcp_z = float(RIM[2] + args.transport_clearance)
release_tcp_z = float(PAD[2] + args.release_clearance)
transport_flange_z = flange_z_for_tcp_z(transport_tcp_z, args.tcp_offset)
release_flange_z = flange_z_for_tcp_z(release_tcp_z, args.tcp_offset)
assert_safe_flange_z("basket transport", transport_flange_z, args.min_flange_z)
assert_safe_flange_z("basket release", release_flange_z, args.min_flange_z)

print(f"[grid] {len(cells)} cells (of {args.nx * args.ny}; excluded/skipped cells are not attempted)")
print(f"[basket] pad_center=({PAD[0]:+.3f},{PAD[1]:+.3f},{PAD[2]:+.3f})  rim_z={RIM[2]:+.3f}")
print(f"[basket] transport_tcp_z={transport_tcp_z:+.3f} flange_z={transport_flange_z:+.3f}  "
      f"release_tcp_z={release_tcp_z:+.3f} flange_z={release_flange_z:+.3f}")
print(f"[grasp] target_width={args.grasp_width} speed={args.grasp_speed} force={args.grasp_force} "
      f"epsilon=+/-{args.grasp_epsilon} lowering={args.grasp_lowering}")
print("[reminder] orient the cube so its SHORT (4x4cm) face is between the gripper fingers.\n")

input("[SAFETY] Confirm workspace AND path over the basket are clear. Press Enter to connect...")

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

print("[gripper] homing ...")
gripper.homing()
print(f"[gripper] ready. max_width={gripper.max_width:.4f}")
if args.grasp_width >= gripper.max_width:
    raise SystemExit(f"grasp_width {args.grasp_width} >= max_width {gripper.max_width}")

home_xyz = np.array(robot.current_pose.end_effector_pose.translation)
travel_z_floor = compute_travel_z_floor(
    home_xyz,
    [flange_z_for_tcp_z(float(cell[2] + args.hover_clearance), args.tcp_offset) for cell in cells]
    + [transport_flange_z, release_flange_z],
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
    ans = input("  place the cube, then Enter to retreat home and start automatic pick-place ('s'=skip cell): ")
    if ans.strip().lower() == "s":
        motion.travel_to(home_xyz, home_xyz[2], "retreat home")
        print("  [skip]\n")
        continue

    motion.travel_to(home_xyz, home_xyz[2], "retreat home before auto pick")
    print("  [auto] starting pick-place sequence; no more prompts for this cell.")
    reset_dynamics(robot, args.dynamics_factor)
    gripper.open(args.open_speed)
    motion.travel_to(cell_xyz, hover_flange_z, "auto return to hover")
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z]), "descend to grasp")

    grasped_ok = gripper.grasp(args.grasp_width, args.grasp_speed, args.grasp_force,
                               epsilon_inner=args.grasp_epsilon, epsilon_outer=args.grasp_epsilon)
    print(f"  [grasp] grasp() returned: {grasped_ok}  (width now: {gripper.width:.4f})")

    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z + args.lift_check_height]), "lift & verify")
    still_holding = gripper.is_grasped
    print(f"  [verify] is_grasped={still_holding}")

    if not (grasped_ok and still_holding):
        print("  [fail] grasp did not hold — releasing and retreating without transport.")
        gripper.open(args.open_speed)
        motion.travel_to(home_xyz, home_xyz[2], "retreat home")
        results.append(False)
        print("  [done] FAIL.\n")
        continue

    print("  [auto] grasp confirmed; transporting over basket rim.")
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], transport_flange_z]), "rise to transport height")
    motion.travel_to(PAD, transport_flange_z, "fly to above basket")
    motion.move_in_steps(np.array([PAD[0], PAD[1], release_flange_z]), "descend into basket")

    gripper.open(args.open_speed)
    print("  [release] cube released into basket.")

    motion.move_in_steps(np.array([PAD[0], PAD[1], transport_flange_z]), "rise out of basket")
    motion.travel_to(home_xyz, home_xyz[2], "retreat home")
    results.append(True)
    print("  [done] SUCCESS.\n")

n = len(results)
s = sum(results)
print(f"=== PICK-AND-PLACE COMPLETE: {s}/{n} succeeded ===" if n else "=== no cells attempted ===")
