#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/manual_fr3_policy_grasp.py --help
"""Perform one stationary FR3 gripper action after arm streaming stops.

Run only after the streaming arm controller has stopped at a policy-requested
closure or release. This tool never commands arm motion. ``SKIP`` deliberately
performs no motion so the separate recover-only flow can return the cube to the
basket.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--width", type=float, default=0.04)
    parser.add_argument("--speed", type=float, default=0.02)
    parser.add_argument("--force", type=float, default=15.0)
    parser.add_argument("--open-speed", type=float, default=0.05)
    parser.add_argument("--release", action="store_true", help="Prompt for OPEN rather than CLOSE.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute the requested gripper action without an interactive prompt.",
    )
    parser.add_argument(
        "--require-held",
        action="store_true",
        help="Before release, require the gripper to report a held object.",
    )
    args = parser.parse_args()
    if args.width <= 0 or args.speed <= 0 or args.force <= 0:
        raise SystemExit("--width, --speed, and --force must be positive")

    from franky import Gripper

    print("[GRIPPER] Arm-only streaming must already be stopped. This tool never moves the arm.")
    expected = "OPEN" if args.release else "CLOSE"
    decision = expected if args.yes else input(
        f"Type {expected} to command the gripper, or SKIP for no command: "
    ).strip().upper()
    if decision == "SKIP":
        print("[GRIPPER] skipped; no command issued.")
        return
    if decision != expected:
        print("[GRIPPER] aborted; no command issued.")
        return
    gripper = Gripper(args.robot_ip)
    if args.release:
        if args.require_held and not gripper.is_grasped:
            raise SystemExit("[GRIPPER] release refused: no held object was detected")
        held_before_open = bool(gripper.is_grasped)
        gripper.open(args.open_speed)
        width = float(gripper.state.width)
        print(
            f"[GRIPPER] open() sent held_before_open={held_before_open} width={width:.4f}"
        )
        if width < 0.07:
            raise SystemExit("[GRIPPER] release failed: gripper did not reach an open width")
        return
    accepted = gripper.grasp(args.width, args.speed, args.force, 0.01, 0.01)
    grasped = bool(gripper.is_grasped)
    print(
        f"[GRIPPER] grasp()={accepted} is_grasped={grasped} "
        f"width={gripper.state.width:.4f}"
    )
    if not accepted or not grasped:
        raise SystemExit("[GRIPPER] grasp failed verification")


if __name__ == "__main__":
    main()
