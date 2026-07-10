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
import atexit
import itertools
import json
import signal
import sys
import time
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
from create_grid_tracker import worksheet_xml, write_xlsx
from filter_grid_tracker import read_tracker
from realsense_recorder import RealSenseEpisodeRecorder

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, required=True)
parser.add_argument("--points", type=str, default="probed_points.json")
parser.add_argument("--cells_json", type=str, default=None,
                    help="Optional JSON exported by filter_grid_tracker.py. If set, use those cells instead of generating a grid.")
parser.add_argument("--status_tracker", type=str, default=None,
                    help="Optional tracker .xlsx to update PASS/FAIL/SKIP by printed_cell during execution.")
parser.add_argument("--record_dir", type=str, default=None,
                    help="If set, record RealSense RGB episodes for the cell-pick-to-basket segment.")
parser.add_argument("--camera_serial", type=str, action="append", default=None,
                    help="RealSense serial to record. Repeat for multiple cameras. Defaults to all connected cameras.")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_fps", type=int, default=60)
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
parser.add_argument("--recycle_from_basket", action="store_true",
                    help="Start with cube in basket, place it on each cell, retreat home, then pick it back into basket.")
parser.add_argument("--cell_drop_clearance", type=float, default=0.04,
                    help="In recycle mode, release cube this far above the cell grasp height instead of setting it down.")
parser.add_argument("--post_cell_pause", type=float, default=0.0,
                    help="Seconds to pause after each completed/skipped/failed cell before continuing.")
parser.add_argument("--settle_time", type=float, default=0.10,
                    help="Seconds to wait after each Cartesian move before sending the next command.")
parser.add_argument("--motion_retry_policy", choices=["retry_slower", "same_speed", "fail_fast"], default="fail_fast",
                    help="For dataset consistency, fail_fast avoids slower retries and segmented fallback.")
parser.add_argument("--allow_segment_fallback", action="store_true", default=False,
                    help="Allow fallback from one smooth move to segmented motion after a rejected command.")
parser.add_argument("--episode_retries", type=int, default=1,
                    help="Recorded-segment retries per cell before marking FAIL.")
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
parser.add_argument("--min_grasp_above_table", type=float, default=0.006,
                    help="Abort if grasp target is closer than this to the local table surface, meters.")
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
parser.add_argument("--printed_cell", type=int, action="append", default=[],
                    help="Run only matching tracker printed_cell values from --cells_json. Can be repeated.")
parser.add_argument("--start_cell", type=int, default=0,
                    help="Zero-based cell index to start from after filtering.")
parser.add_argument("--end_cell", type=int, default=None,
                    help="Zero-based cell index to stop at after filtering, inclusive.")
parser.add_argument("--resume_file", type=Path, default=Path("next_cell_to_record.txt"),
                    help="Text file updated with the next cell to record.")
parser.add_argument("--resume_from_file", action="store_true",
                    help="Start from --resume_file's next_printed_cell when using --cells_json.")
args = parser.parse_args()
if args.max_cells is not None and args.max_cells <= 0:
    raise SystemExit("--max_cells must be positive.")

CURRENT_SOURCE = None
recorder = None


class MotionExecutionFailure(RuntimeError):
    pass


def source_label(source: dict | None, selected_index: int) -> str:
    if source is not None and "printed_cell" in source:
        return f"printed_cell_{int(source['printed_cell']):03d}"
    return f"selected_index_{selected_index:03d}"


def read_resume_printed_cell(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        value = data.get("next_printed_cell")
    except json.JSONDecodeError:
        value = text
    if value in (None, "", "DONE"):
        return None
    return int(value)


def write_resume_file(source: dict | None, run_index: int | None, status: str) -> None:
    payload = {
        "status": status,
        "run_index": run_index,
        "next_printed_cell": None,
        "next_selected_index": None,
        "next_generated_index": None,
        "updated_time_ns": time.time_ns(),
    }
    if source is not None:
        payload["next_printed_cell"] = int(source["printed_cell"])
        payload["next_selected_index"] = int(source["selected_index"])
        payload["next_generated_index"] = int(source["generated_index"])
    args.resume_file.write_text(json.dumps(payload, indent=2) + "\n")


def update_status_tracker(source: dict | None, status: str, note: str = "") -> None:
    if not args.status_tracker or source is None:
        return
    tracker_path = Path(args.status_tracker)
    printed_cell = int(source["printed_cell"])
    rows = read_tracker(tracker_path)
    updated = False
    for row in rows:
        if int(row["printed_cell"]) == printed_cell:
            row["status"] = status
            if note:
                old_note = str(row.get("notes") or "")
                row["notes"] = note if not old_note else f"{old_note}; {note}"
            updated = True
            break
    if not updated:
        print(f"  [tracker] printed_cell={printed_cell} not found in {tracker_path}; status not updated.")
        return
    sheet = worksheet_xml(rows, args.hover_clearance, args.cube_height, args.grasp_lowering)
    write_xlsx(tracker_path, sheet)
    print(f"  [tracker] marked printed_cell={printed_cell} as {status} in {tracker_path}")


def mark_current_fail(note: str) -> None:
    if recorder is not None and recorder.active:
        update_status_tracker(CURRENT_SOURCE, "FAIL", note)
        recorder.stop_episode(False, {"failure_note": note})


def _excepthook(exc_type, exc, tb):
    if exc_type is not KeyboardInterrupt:
        mark_current_fail(f"exception: {exc_type.__name__}")
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _excepthook

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
    cell_records = None
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
    if cell_records is not None:
        cell_records = [c for k, c in enumerate(cell_records) if k not in skip]
    print(f"[grid] skipped cells by request: {sorted(skip)} ({before_skip - len(cells)} matched)")

if args.max_cells is not None and len(cells) > args.max_cells:
    keep = np.linspace(0, len(cells) - 1, args.max_cells, dtype=int)
    cells = [cells[k] for k in keep]
    if cell_records is not None:
        cell_records = [cell_records[k] for k in keep]
    print(f"[grid] limited to {args.max_cells} evenly sampled cells")

if args.skip_selected_cell:
    skip = set(args.skip_selected_cell)
    before_skip = len(cells)
    cells = [c for k, c in enumerate(cells) if k not in skip]
    if cell_records is not None:
        cell_records = [c for k, c in enumerate(cell_records) if k not in skip]
    print(f"[grid] skipped selected cells by request: {sorted(skip)} ({before_skip - len(cells)} matched)")

if args.printed_cell:
    if cell_records is None:
        raise SystemExit("--printed_cell requires --cells_json.")
    wanted = set(args.printed_cell)
    matched = [(cell, record) for cell, record in zip(cells, cell_records) if int(record["printed_cell"]) in wanted]
    cells = [cell for cell, _ in matched]
    cell_records = [record for _, record in matched]
    print(f"[grid] selected tracker printed_cell values: {sorted(wanted)} ({len(cells)} matched)")

if args.resume_from_file:
    if cell_records is None:
        raise SystemExit("--resume_from_file requires --cells_json.")
    resume_printed_cell = read_resume_printed_cell(args.resume_file)
    if resume_printed_cell is not None:
        start_idx = next(
            (idx for idx, record in enumerate(cell_records) if int(record["printed_cell"]) == resume_printed_cell),
            None,
        )
        if start_idx is None:
            greater_idx = next(
                (idx for idx, record in enumerate(cell_records) if int(record["printed_cell"]) > resume_printed_cell),
                None,
            )
            if greater_idx is None:
                print(f"[resume] next_printed_cell={resume_printed_cell} is past the end of the current cells JSON.")
                cells = []
                cell_records = []
            else:
                print(f"[resume] next_printed_cell={resume_printed_cell} was removed; resuming at next available "
                      f"printed_cell={cell_records[greater_idx]['printed_cell']}.")
                cells = cells[greater_idx:]
                cell_records = cell_records[greater_idx:]
        else:
            print(f"[resume] starting from resume file printed_cell={resume_printed_cell} at current JSON index {start_idx}.")
            cells = cells[start_idx:]
            cell_records = cell_records[start_idx:]
    else:
        print(f"[resume] no usable next_printed_cell in {args.resume_file}; starting from selected cells.")

if args.start_cell or args.end_cell is not None:
    end = len(cells) - 1 if args.end_cell is None else args.end_cell
    cells = cells[args.start_cell:end + 1]
    if cell_records is not None:
        cell_records = cell_records[args.start_cell:end + 1]
    print(f"[grid] running cell range {args.start_cell}..{end}")

if not cells:
    print("[grid] no cells selected; exiting before robot connection.")
    raise SystemExit(0)

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
print(f"[mode] {'basket recycle' if args.recycle_from_basket else 'manual cell placement'}")
if args.recycle_from_basket:
    print(f"[recycle] cell_drop_clearance={args.cell_drop_clearance:.3f}")
print("[reminder] orient the cube so its SHORT (4x4cm) face is between the gripper fingers.\n")

if args.record_dir:
    print(f"[record] preparing RealSense recorder in {args.record_dir}")
    recorder = RealSenseEpisodeRecorder(
        args.record_dir,
        serials=args.camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    atexit.register(recorder.close)
    print(f"[record] cameras: {', '.join(recorder.camera_names)}")

input("[SAFETY] Confirm workspace AND path over the basket are clear. Press Enter to connect...")

from franky import Robot, Gripper  # noqa: E402

robot = Robot(args.robot_ip)
robot.relative_dynamics_factor = args.dynamics_factor
gripper = Gripper(args.robot_ip)


def _sigint_handler(signum, frame):
    print("\n[KILLSWITCH] Ctrl+C received — stopping robot motion now.")
    mark_current_fail("interrupted")
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

if recorder is not None:
    recorder.attach_robot(robot, gripper, sample_hz=args.camera_fps)

motion = MotionPlanner(
    robot,
    args.dynamics_factor,
    args.max_step,
    travel_z_floor,
    args.settle_time,
    action_callback=(recorder.log_action if recorder is not None else None),
    retry_policy=args.motion_retry_policy,
    allow_segment_fallback=args.allow_segment_fallback,
)


def open_and_stop_after_motion_failure(context: str, exc: Exception) -> None:
    print(f"  [fail] {context} failed: {exc}")
    print("  [fail] opening gripper and handing failure back to the episode controller.")
    try:
        gripper.open(args.open_speed)
    finally:
        raise MotionExecutionFailure(context) from exc


def pick_from_pose(xyz: np.ndarray, hover_z: float, grasp_z: float, label: str) -> bool:
    motion.travel_to(xyz, hover_z, f"{label}: to hover")
    motion.move_in_steps(np.array([xyz[0], xyz[1], grasp_z]), f"{label}: descend to grasp")
    grasped = gripper.grasp(
        args.grasp_width,
        args.grasp_speed,
        args.grasp_force,
        epsilon_inner=args.grasp_epsilon,
        epsilon_outer=args.grasp_epsilon,
    )
    print(f"  [grasp] {label}: grasp() returned: {grasped}  (width now: {gripper.width:.4f})")
    try:
        motion.move_in_steps(np.array([xyz[0], xyz[1], grasp_z + args.lift_check_height]), f"{label}: lift & verify")
    except Exception as e:
        open_and_stop_after_motion_failure(f"{label}: lift from grasp pose", e)
    holding = gripper.is_grasped
    print(f"  [verify] {label}: is_grasped={holding}")
    return bool(grasped and holding)


def place_at_pose(xyz: np.ndarray, hover_z: float, place_z: float, label: str) -> None:
    motion.travel_to(xyz, hover_z, f"{label}: to hover")
    motion.move_in_steps(np.array([xyz[0], xyz[1], place_z]), f"{label}: descend to place")
    gripper.open(args.open_speed)
    print(f"  [release] {label}: cube released.")
    motion.move_in_steps(np.array([xyz[0], xyz[1], hover_z]), f"{label}: retreat to hover")


def start_cell_recording(source: dict | None, selected_index: int, cell_xyz: np.ndarray, table_z: float) -> None:
    if recorder is None:
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    episode_name = f"{stamp}_{source_label(source, selected_index)}"
    metadata = {
        "robot_ip": args.robot_ip,
        "mode": "basket_recycle" if args.recycle_from_basket else "manual_cell_placement",
        "selected_index": selected_index,
        "cell_xyz": [float(cell_xyz[0]), float(cell_xyz[1]), float(cell_xyz[2])],
        "table_z": float(table_z),
        "home_xyz": [float(home_xyz[0]), float(home_xyz[1]), float(home_xyz[2])],
        "hover_clearance": args.hover_clearance,
        "cube_height": args.cube_height,
        "grasp_lowering": args.grasp_lowering,
        "cell_drop_clearance": args.cell_drop_clearance,
    }
    if source is not None:
        metadata["source"] = dict(source)
    recorder.start_episode(episode_name, metadata)
    recorder.add_event("cube_on_cell_home_reached")


def stop_cell_recording(success: bool, note: str) -> None:
    if recorder is None or not recorder.active:
        return
    try:
        recorder.add_event("episode_stop_requested", {"success": success, "note": note})
        recorder.stop_episode(success, {"note": note})
    except Exception as e:
        print(f"  [record-fail] could not stop/write recording cleanly: {e}")
        try:
            recorder.active = False
        except Exception:
            pass
        update_status_tracker(CURRENT_SOURCE, "RECORDING_FAIL", f"{note}; recorder error: {type(e).__name__}")


def recover_for_episode_retry(context: str, cube_may_be_held: bool) -> bool:
    print(f"  [retry] recorded attempt failed during {context}; discarding this episode.")
    stop_cell_recording(False, context)
    try:
        if robot.has_errors:
            robot.recover_from_errors()
        reset_dynamics(robot, args.dynamics_factor)
        if cube_may_be_held and gripper.is_grasped:
            print("  [retry] cube appears grasped; returning it to the basket before retrying.")
            current = motion.current_xyz()
            motion.move_in_steps(np.array([current[0], current[1], transport_flange_z]), "retry recovery: rise")
            place_at_pose(PAD, transport_flange_z, release_flange_z, "retry recovery: place in basket")
            motion.travel_to(home_xyz, home_xyz[2], "retry recovery: home")
            print("  [retry] placing cube back on the cell for a clean retry.")
            gripper.open(args.open_speed)
            basket_picked = pick_from_pose(PAD, transport_flange_z, basket_grasp_flange_z, "retry basket pickup")
            if not basket_picked:
                return False
            place_at_pose(cell_xyz, hover_flange_z, cell_drop_flange_z, "retry drop on cell")
            motion.travel_to(home_xyz, home_xyz[2], "retry home after cell placement")
        else:
            print("  [retry] cube should still be on the cell; returning home and retrying from there.")
            gripper.open(args.open_speed)
            motion.travel_to(home_xyz, home_xyz[2], "retry recovery: home")
        return True
    except Exception as e:
        print(f"  [fail] automatic retry recovery failed: {e}")
        stop_cell_recording(False, f"recovery failed: {type(e).__name__}")
        try:
            gripper.open(args.open_speed)
        finally:
            return False


def stop_after_setup_failure(source: dict | None, run_index: int, context: str, exc: Exception) -> None:
    print(f"  [setup-fail] {context}: {exc}")
    print("  [setup-fail] no recorded episode was active; marking SETUP_FAIL and stopping for hand-guiding.")
    update_status_tracker(source, "SETUP_FAIL", context)
    advance_resume(run_index, "advanced_after_setup_fail")
    if recorder is not None and recorder.active:
        stop_cell_recording(False, f"setup failure: {context}")
    try:
        if robot.has_errors:
            robot.recover_from_errors()
        gripper.open(args.open_speed)
    finally:
        raise SystemExit("[robot] stopped after setup failure. Hand-guide home, then restart from this cell.")


if args.recycle_from_basket:
    input("[SETUP] Put the cube in the basket at pad_center, with the same orientation as normal grasping. Press Enter...")


def post_cell_pause() -> None:
    if args.post_cell_pause > 0:
        print(f"  [pause] waiting {args.post_cell_pause:.1f}s before next cell...")
        time.sleep(args.post_cell_pause)


def advance_resume(run_index: int, status: str) -> None:
    next_index = run_index + 1
    next_source = cell_records[next_index] if cell_records is not None and next_index < len(cell_records) else None
    write_resume_file(next_source, next_index if next_source is not None else None, status)


results = []
for k, cell_xyz in enumerate(cells):
    source = cell_records[k] if cell_records is not None else None
    CURRENT_SOURCE = source
    write_resume_file(source, k, "current")
    table_z = float(cell_xyz[2])
    hover_tcp_z = table_z + args.hover_clearance
    grasp_tcp_z = table_z + args.cube_height / 2 - args.grasp_lowering
    basket_grasp_tcp_z = float(PAD[2] + args.cube_height / 2 - args.grasp_lowering)
    basket_grasp_flange_z = flange_z_for_tcp_z(basket_grasp_tcp_z, args.tcp_offset)
    hover_flange_z = flange_z_for_tcp_z(hover_tcp_z, args.tcp_offset)
    grasp_flange_z = flange_z_for_tcp_z(grasp_tcp_z, args.tcp_offset)
    cell_drop_flange_z = min(hover_flange_z, grasp_flange_z + args.cell_drop_clearance)
    assert_safe_flange_z("hover", hover_flange_z, args.min_flange_z)
    if grasp_tcp_z - table_z < args.min_grasp_above_table:
        raise SystemExit(
            f"[SAFETY] Refusing grasp: target is only {grasp_tcp_z - table_z:.4f} m above "
            f"local table_z={table_z:.4f}; min_grasp_above_table={args.min_grasp_above_table:.4f}."
        )
    if args.recycle_from_basket and basket_grasp_tcp_z - PAD[2] < args.min_grasp_above_table:
        raise SystemExit(
            f"[SAFETY] Refusing basket grasp: target is only {basket_grasp_tcp_z - PAD[2]:.4f} m above "
            f"pad_center_z={PAD[2]:.4f}; min_grasp_above_table={args.min_grasp_above_table:.4f}."
        )

    print(f"=== cell {k+1}/{len(cells)}: x={cell_xyz[0]:+.3f} y={cell_xyz[1]:+.3f} "
          f"table_z={table_z:+.3f} grasp_tcp_z={grasp_tcp_z:+.3f} "
          f"grasp_flange_z={grasp_flange_z:+.3f} ===")
    if source is not None:
        print(f"  [source] tracker printed_cell={source['printed_cell']} "
              f"selected_index={source['selected_index']} generated_index={source['generated_index']}")
    print(f"  [safety] current_flange_z={motion.current_xyz()[2]:+.4f} "
          f"hover_flange_z={hover_flange_z:+.4f} grasp_flange_z={grasp_flange_z:+.4f}")

    if args.recycle_from_basket:
        print("  [auto] recycling cube: basket -> cell -> home -> cell -> basket.")
        try:
            reset_dynamics(robot, args.dynamics_factor)
            gripper.open(args.open_speed)
            basket_picked = pick_from_pose(PAD, transport_flange_z, basket_grasp_flange_z, "basket pickup")
            if not basket_picked:
                print("  [setup-fail] could not pick cube from basket; stopping before moving to cell.")
                gripper.open(args.open_speed)
                motion.travel_to(home_xyz, home_xyz[2], "retreat home")
                results.append(False)
                update_status_tracker(source, "SETUP_FAIL", "basket pickup failed")
                advance_resume(k, "advanced_after_setup_fail")
                post_cell_pause()
                continue

            print(f"  [auto] dropping cube on cell from z={cell_drop_flange_z:+.4f} "
                  f"(grasp_z={grasp_flange_z:+.4f}, clearance={args.cell_drop_clearance:.3f}).")
            place_at_pose(cell_xyz, hover_flange_z, cell_drop_flange_z, "drop on cell")
            motion.travel_to(home_xyz, home_xyz[2], "retreat home after cell placement")
            print("  [auto] cube placed on cell; returning from home to pick it into basket.")
        except Exception as e:
            stop_after_setup_failure(source, k, "basket-to-cell setup failed", e)

        cell_success = False
        retry_note = ""
        for attempt in range(1, args.episode_retries + 2):
            if attempt > 1:
                print(f"  [retry] starting clean recorded attempt {attempt}/{args.episode_retries + 1} at same speed.")
            cube_may_be_held = False
            start_cell_recording(source, k, cell_xyz, table_z)
            try:
                gripper.open(args.open_speed)
                cell_picked = pick_from_pose(cell_xyz, hover_flange_z, grasp_flange_z, "cell pickup")
                cube_may_be_held = bool(cell_picked)
                if not cell_picked:
                    retry_note = "cell pickup failed"
                    stop_cell_recording(False, retry_note)
                    gripper.open(args.open_speed)
                    motion.travel_to(home_xyz, home_xyz[2], "retreat home")
                    if attempt <= args.episode_retries:
                        continue
                    break

                print("  [auto] cell grasp confirmed; transporting over basket rim.")
                motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], transport_flange_z]), "rise to transport height")
                place_at_pose(PAD, transport_flange_z, release_flange_z, "place in basket")
                motion.travel_to(home_xyz, home_xyz[2], "retreat home")
                stop_cell_recording(True, "auto basket recycle success")
                cell_success = True
                break
            except Exception as e:
                retry_note = f"recorded attempt exception: {type(e).__name__}"
                if attempt > args.episode_retries:
                    stop_cell_recording(False, retry_note)
                    raise
                recovered = recover_for_episode_retry(retry_note, cube_may_be_held)
                if not recovered:
                    retry_note = "automatic retry recovery failed"
                    break

        results.append(cell_success)
        if cell_success:
            print("  [done] SUCCESS.\n")
            update_status_tracker(source, "PASS", "auto basket recycle success")
            advance_resume(k, "advanced_after_success")
        else:
            print("  [done] FAIL.\n")
            update_status_tracker(source, "FAIL", retry_note or "recorded attempt failed")
            advance_resume(k, "advanced_after_fail")
        post_cell_pause()
        continue

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
        update_status_tracker(source, "SKIP", "operator skipped")
        advance_resume(k, "advanced_after_skip")
        post_cell_pause()
        continue

    motion.travel_to(home_xyz, home_xyz[2], "retreat home before auto pick")
    print("  [auto] starting pick-place sequence; no more prompts for this cell.")
    start_cell_recording(source, k, cell_xyz, table_z)
    reset_dynamics(robot, args.dynamics_factor)
    gripper.open(args.open_speed)
    motion.travel_to(cell_xyz, hover_flange_z, "auto return to hover")
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z]), "descend to grasp")

    grasped_ok = gripper.grasp(args.grasp_width, args.grasp_speed, args.grasp_force,
                               epsilon_inner=args.grasp_epsilon, epsilon_outer=args.grasp_epsilon)
    print(f"  [grasp] grasp() returned: {grasped_ok}  (width now: {gripper.width:.4f})")

    try:
        motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], grasp_flange_z + args.lift_check_height]), "lift & verify")
        still_holding = gripper.is_grasped
    except Exception as e:
        print(f"  [fail] lift from grasp pose failed: {e}")
        print("  [fail] opening gripper. Mark this tracker cell as FAIL/SKIP and hand-guide back to home.")
        try:
            gripper.open(args.open_speed)
        finally:
            stop_cell_recording(False, "lift from grasp pose failed")
            raise SystemExit("[robot] stopped after lift failure to avoid commanding more motion from a singular pose.")
    print(f"  [verify] is_grasped={still_holding}")

    if not (grasped_ok and still_holding):
        print("  [fail] grasp did not hold — releasing and retreating without transport.")
        gripper.open(args.open_speed)
        motion.travel_to(home_xyz, home_xyz[2], "retreat home")
        stop_cell_recording(False, "grasp did not hold")
        results.append(False)
        print("  [done] FAIL.\n")
        update_status_tracker(source, "FAIL", "grasp did not hold")
        advance_resume(k, "advanced_after_fail")
        post_cell_pause()
        continue

    print("  [auto] grasp confirmed; transporting over basket rim.")
    motion.move_in_steps(np.array([cell_xyz[0], cell_xyz[1], transport_flange_z]), "rise to transport height")
    motion.travel_to(PAD, transport_flange_z, "fly to above basket")
    motion.move_in_steps(np.array([PAD[0], PAD[1], release_flange_z]), "descend into basket")

    gripper.open(args.open_speed)
    print("  [release] cube released into basket.")

    motion.move_in_steps(np.array([PAD[0], PAD[1], transport_flange_z]), "rise out of basket")
    motion.travel_to(home_xyz, home_xyz[2], "retreat home")
    stop_cell_recording(True, "manual placement success")
    results.append(True)
    print("  [done] SUCCESS.\n")
    update_status_tracker(source, "PASS", "manual placement success")
    advance_resume(k, "advanced_after_success")
    post_cell_pause()

n = len(results)
s = sum(results)
write_resume_file(None, None, "done")
print(f"=== PICK-AND-PLACE COMPLETE: {s}/{n} succeeded ===" if n else "=== no cells attempted ===")
