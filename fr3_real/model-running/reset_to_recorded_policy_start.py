#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/reset_to_recorded_policy_start.py --help
"""Slowly reset FR3 to the exact initial joint state of one recorded episode."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import RECORDINGS_DIR

JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def first_state(path: Path) -> dict:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"No robot state in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--record_dir", default=str(RECORDINGS_DIR / "droid_raw_full_v3"))
    parser.add_argument("--episode", required=True, help="Episode directory name, not its full path.")
    parser.add_argument("--dynamics_factor", type=float, default=0.05)
    parser.add_argument("--joint_margin", type=float, default=0.12)
    parser.add_argument("--min_x", type=float, default=0.18)
    parser.add_argument("--max_x", type=float, default=0.75)
    parser.add_argument("--min_y", type=float, default=-0.50)
    parser.add_argument("--max_y", type=float, default=0.45)
    parser.add_argument("--min_z", type=float, default=0.10)
    parser.add_argument("--max_z", type=float, default=0.65)
    args = parser.parse_args()

    episode_dir = Path(args.record_dir) / args.episode
    metadata_path = episode_dir / "metadata.json"
    state_path = episode_dir / "robot_state.jsonl"
    if not metadata_path.exists() or not state_path.exists():
        raise SystemExit(f"Missing metadata.json or robot_state.jsonl in {episode_dir}")
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("success", False):
        raise SystemExit(f"{episode_dir.name} was not a successful episode")
    target_q = np.asarray(first_state(state_path)["q"], dtype=float)
    if target_q.shape != (7,):
        raise SystemExit(f"Expected 7 target joints, got {target_q.shape}")

    from franky import Frame, JointMotion, JointState, Robot

    print("[RESET] One slow arm-only move to a recorded policy start state.")
    print("[RESET] This script never commands the gripper.")
    print(f"[RESET] episode={episode_dir.name}")
    print(f"[RESET] recorded cell={metadata.get('source', {}).get('printed_cell', 'unknown')}")
    input("Keep the workspace clear and E-stop reachable, then press Enter to connect...")

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    current = robot.state
    current_q = np.asarray(current.q, dtype=float)
    lower = JOINT_MIN + args.joint_margin
    upper = JOINT_MAX - args.joint_margin
    if np.any(target_q < lower) or np.any(target_q > upper):
        raise RuntimeError("Recorded target violates guarded joint limits")

    # Frame.Joint7 is the reliable base-frame FK exposed by this Franky build.
    tcp_current = np.asarray(current.O_T_EE.matrix, dtype=float)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, current).matrix, dtype=float)
    joint7_to_tcp = np.linalg.solve(joint7_current, tcp_current)
    samples = np.linspace(current_q, target_q, num=41)
    path_xyz = np.asarray([
        (
            np.asarray(
                robot.model.pose(Frame.Joint7, q, current.F_T_EE, current.EE_T_K).matrix,
                dtype=float,
            )
            @ joint7_to_tcp
        )[:3, 3]
        for q in samples
    ])
    xyz_min = np.asarray([args.min_x, args.min_y, args.min_z])
    xyz_max = np.asarray([args.max_x, args.max_y, args.max_z])
    if np.any(path_xyz < xyz_min) or np.any(path_xyz > xyz_max):
        raise RuntimeError(
            "Predicted reset path leaves guarded Cartesian bounds: "
            f"min={path_xyz.min(axis=0)}, max={path_xyz.max(axis=0)}"
        )

    print(f"[CHECK] current q={np.array2string(current_q, precision=4)}")
    print(f"[CHECK] target  q={np.array2string(target_q, precision=4)}")
    print(f"[CHECK] q delta ={np.array2string(target_q - current_q, precision=4)}")
    print(
        f"[CHECK] predicted TCP path start={np.array2string(path_xyz[0], precision=4)} "
        f"end={np.array2string(path_xyz[-1], precision=4)}"
    )
    if input("Type MOVE exactly to reset to this recorded start, anything else aborts: ") != "MOVE":
        print("[RESET] aborted without motion.")
        return

    try:
        robot.move(JointMotion(JointState(target_q)))
    except BaseException:
        robot.stop()
        raise
    end = robot.state
    print(f"[DONE] final q={np.array2string(np.asarray(end.q, dtype=float), precision=4)}")
    print(f"[DONE] final tcp={np.array2string(np.asarray(end.O_T_EE.translation, dtype=float), precision=4)}")
    print("[DONE] recorded-start reset complete; no gripper command was issued.")


if __name__ == "__main__":
    main()
