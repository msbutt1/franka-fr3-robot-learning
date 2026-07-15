#!/usr/bin/env python3
"""Cube-placement indicator loop for the real FR3.

For each grid cell (computed by bilinear interpolation over your 4 probed corners,
excluding the basket footprint), the arm moves to a high travel plane above the
spot so you know where the cube belongs. By default it does NOT descend toward
the table. It then retreats to a safe "home" pose (wherever the arm was when you
started the script).

This script does NOT grasp/move the cube yet — it only points and waits. The
scripted pick-and-place-into-basket motion (needs the Gripper API, which I have
not yet verified against real hardware) is the next script, once you've run:
    python -c "import franky; help(franky.Gripper)"
and shared the output.

SAFETY:
  - Before running: manually guide the arm (hand-guiding) to a clear, comfortable
    "home" pose, well clear of the table/basket. The script reads whatever pose
    the arm is in at startup and treats it as home — it will always return there.
  - Motions use a LOW relative_dynamics_factor (5%) by default.
  - Only pure hover TRANSLATIONS relative to home are commanded — orientation is
    never changed, so there is no risk of an unexpected/uncalibrated rotation.
  - You must explicitly press Enter before the arm moves at all, and before each
    hover move.

Usage:
    python fr3_real/robot/indicate_and_confirm.py --robot_ip 172.16.0.2 \\
        --nx 3 --ny 3 --hover_clearance 0.12
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fr3_real.common.franka_motion import (
    DEFAULT_FRANKA_HAND_TCP_OFFSET,
    MotionPlanner,
    assert_safe_flange_z,
    clear_robot_errors,
    compute_travel_z_floor,
    configure_collision_behavior,
    flange_z_for_tcp_z,
    reset_dynamics,
)
from fr3_real.common.grid_utils import basket_polygon_from_points, inside_basket_exclusion
from fr3_real.paths import DEFAULT_POINTS_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--points", type=str, default=str(DEFAULT_POINTS_PATH),
                    help="Output of fr3_real/robot/probe_points.py — needs bottom_left/bottom_right/"
                         "top_left/top_right/pad_center at minimum.")
parser.add_argument("--cells_json", type=str, default=None,
                    help="Optional JSON exported by fr3_real/grid/filter_grid_tracker.py. If set, use those cells instead of generating a grid.")
parser.add_argument("--nx", type=int, default=3, help="Grid cells along near/far axis.")
parser.add_argument("--ny", type=int, default=3, help="Grid cells along left/right axis.")
parser.add_argument("--basket_w", type=float, default=0.154, help="Basket footprint, meters (long side).")
parser.add_argument("--basket_h", type=float, default=0.134, help="Basket footprint, meters (short side).")
parser.add_argument("--basket_margin", type=float, default=0.04,
                    help="Extra clearance around the basket footprint to exclude, meters.")
parser.add_argument("--hover_clearance", type=float, default=0.12,
                    help="Height above the interpolated table surface to hover, meters.")
parser.add_argument("--dynamics_factor", type=float, default=0.03,
                    help="Fraction of max vel/accel/jerk to use (franky's 'start slow' setting).")
parser.add_argument("--max_step", type=float, default=0.03,
                    help="Maximum Cartesian segment length, meters. Use 0 for one smooth move per leg.")
parser.add_argument("--travel_extra_clearance", type=float, default=0.08,
                    help="Extra Z clearance above the highest hover/home pose for long travel, meters.")
parser.add_argument("--travel_above_home", action="store_true", default=False,
                    help="Also require the travel plane to be above the startup home Z.")
parser.add_argument("--tcp_offset", type=float, default=DEFAULT_FRANKA_HAND_TCP_OFFSET,
                    help="Meters from commanded flange/control frame down to fingertip TCP along table Z.")
parser.add_argument("--min_flange_z", type=float, default=0.10,
                    help="Abort if any commanded flange/control-frame Z drops below this value.")
parser.add_argument("--pause_before_hover_descent", action="store_true", default=True,
                    help="Pause above each target before descending to hover.")
parser.add_argument("--no_pause_before_hover_descent", action="store_false", dest="pause_before_hover_descent",
                    help="Do not pause above each target before descending to hover.")
parser.add_argument("--allow_hover_descent", action="store_true",
                    help="Actually descend from the travel plane to table_z + hover_clearance.")
parser.add_argument("--auto_retreat_after_hover_descent", action="store_true",
                    help="After an allowed hover descent, try to return home automatically.")
parser.add_argument("--settle_time", type=float, default=0.10,
                    help="Seconds to wait after each Cartesian move before sending the next command.")
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
required = ["bottom_left", "bottom_right", "top_left", "top_right", "pad_center"]
missing = [k for k in required if k not in pts]
if missing:
    raise SystemExit(f"probed_points.json is missing: {missing}. Probe these labels first.")

BL = np.array(pts["bottom_left"])   # near, left
BR = np.array(pts["bottom_right"])  # near, right
TL = np.array(pts["top_left"])      # far, left
TR = np.array(pts["top_right"])     # far, right
PAD = np.array(pts["pad_center"])
basket_polygon_xy = basket_polygon_from_points(pts)


def bilinear(u: float, v: float) -> np.ndarray:
    """u: 0(near)->1(far).  v: 0(right)->1(left).  Returns interpolated (x, y, z)."""
    near = (1 - v) * BR + v * BL
    far = (1 - v) * TR + v * TL
    return (1 - u) * near + u * far


# Build the grid, excluding cells inside the basket footprint (+margin).
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
            continue  # inside the basket footprint — skip
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

print(f"[grid] {len(cells)} placement cells (of {args.nx * args.ny}; "
      f"{args.nx * args.ny - len(cells)} excluded/skipped)")
for k, c in enumerate(cells):
    print(f"  cell {k:02d}: x={c[0]:+.3f} y={c[1]:+.3f} table_z={c[2]:+.3f}")

input("\n[SAFETY] Confirm the workspace is clear and you're ready for the arm to move. Press Enter...")

from franky import Robot  # noqa: E402

robot = Robot(args.robot_ip)
robot.relative_dynamics_factor = args.dynamics_factor
print(f"[robot] connected. relative_dynamics_factor={args.dynamics_factor}")
clear_robot_errors(robot)
configure_collision_behavior(robot)

home_xyz = np.array(robot.current_pose.end_effector_pose.translation)
travel_z_floor = compute_travel_z_floor(
    home_xyz,
    [flange_z_for_tcp_z(float(cell[2] + args.hover_clearance), args.tcp_offset) for cell in cells],
    args.travel_extra_clearance,
    include_home_z=args.travel_above_home,
)
print(f"[robot] home pose recorded: x={home_xyz[0]:+.4f} y={home_xyz[1]:+.4f} z={home_xyz[2]:+.4f}")
print(f"[robot] tcp_offset={args.tcp_offset:.4f}  min_flange_z={args.min_flange_z:.4f}")
print(f"[robot] travel_z_floor={travel_z_floor:+.4f} "
      f"(extra clearance {args.travel_extra_clearance:.3f}; travel_above_home={args.travel_above_home})")
print("[robot] this pose will always be returned to between cells.\n")

motion = MotionPlanner(robot, args.dynamics_factor, args.max_step, travel_z_floor, args.settle_time)

for k, cell_xyz in enumerate(cells):
    hover_tcp_z = float(cell_xyz[2] + args.hover_clearance)
    hover_flange_z = flange_z_for_tcp_z(hover_tcp_z, args.tcp_offset)
    assert_safe_flange_z("hover", hover_flange_z, args.min_flange_z)

    print(f"=== cell {k+1}/{len(cells)}: x={cell_xyz[0]:+.3f} y={cell_xyz[1]:+.3f} "
          f"table_z={cell_xyz[2]:+.3f} hover_tcp_z={hover_tcp_z:+.3f} "
          f"hover_flange_z={hover_flange_z:+.3f} ===")
    input("  press Enter to move horizontally above this cell...")
    reset_dynamics(robot, args.dynamics_factor)
    if not args.allow_hover_descent:
        target = motion.travel_above(np.array([cell_xyz[0], cell_xyz[1], hover_flange_z]), "to above cell")
        print(f"  [above] stopped at x={target[0]:+.3f} y={target[1]:+.3f} z={target[2]:+.3f}; no descent commanded.")
    elif args.pause_before_hover_descent:
        motion.travel_to_with_descent_pause(
            cell_xyz,
            hover_flange_z,
            "to hover",
            prompt=f"  arm is above the target at the travel plane. Press Enter to descend to flange_z={hover_flange_z:+.3f}...",
        )
    else:
        motion.travel_to(cell_xyz, hover_flange_z, "to hover")

    if args.allow_hover_descent and not args.auto_retreat_after_hover_descent:
        print("  [hold] hover descent complete; automatic retreat is disabled because this low pose may be singular.")
        print("         Use Desk/hand-guiding to move back to a comfortable home pose before continuing.")
        break

    input("  press Enter to retreat home...")
    motion.travel_to(home_xyz, home_xyz[2], "retreat home")
    print("  [ok] arm back at home.\n")

print("=== ALL CELLS DONE — this pass only indicated positions, nothing was recorded/grasped yet. ===")
