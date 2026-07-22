#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/commission_fr3_gripper_grasp.py --help
"""Commission one stationary FR3 gripper grasp without moving the arm."""

from __future__ import annotations

import argparse
import signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--width", type=float, default=0.04)
    parser.add_argument("--speed", type=float, default=0.02)
    parser.add_argument("--force", type=float, default=15.0)
    parser.add_argument("--epsilon_inner", type=float, default=0.01)
    parser.add_argument("--epsilon_outer", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 <= args.width <= 0.08 or args.speed <= 0 or args.force <= 0:
        raise SystemExit("Invalid gripper width, speed, or force")

    from franky import Gripper

    print("[GRIPPER] One stationary grasp only. This script never commands the arm.")
    print("[GRIPPER] Run only when the cube is visibly centered between the fingers.")
    input("Keep E-stop reachable, then press Enter to connect...")
    gripper = Gripper(args.robot_ip)

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling gripper.stop() now.")
        gripper.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    initial_width = float(gripper.state.width)
    print(f"[CHECK] current width={initial_width:.4f} m max_width={gripper.max_width:.4f} m")
    print(
        f"[CHECK] grasp target width={args.width:.4f} m speed={args.speed:.3f} m/s "
        f"force={args.force:.1f} N"
    )
    if input("Type GRASP exactly to close the stationary gripper, anything else aborts: ") != "GRASP":
        print("[GRIPPER] aborted without command.")
        return

    try:
        accepted = gripper.grasp(
            args.width,
            args.speed,
            args.force,
            args.epsilon_inner,
            args.epsilon_outer,
        )
    except BaseException:
        gripper.stop()
        raise
    final_width = float(gripper.state.width)
    print(f"[DONE] grasp() returned={accepted} width={final_width:.4f} m is_grasped={gripper.is_grasped}")
    print("[DONE] Arm was not moved. Do not lift until this result is reviewed.")


if __name__ == "__main__":
    main()
