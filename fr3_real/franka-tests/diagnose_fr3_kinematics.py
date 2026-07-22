#!/usr/bin/env python3
# Usage: from fr3_real/, run: python franka-tests/diagnose_fr3_kinematics.py --help
"""Read-only Franky kinematics probe for commissioning safety checks."""

from __future__ import annotations

import argparse

import numpy as np


def array(value: object) -> np.ndarray:
    """Convert Franky vector-like objects into a flat float array."""
    return np.asarray(value, dtype=float).reshape(-1)


def translation(value: object) -> np.ndarray:
    return array(value.translation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="172.16.0.2")
    args = parser.parse_args()

    from franky import Frame, Robot

    print("[KIN] Read-only diagnostic: this script never commands robot motion.")
    input("Confirm the workspace is clear, then press Enter to connect read-only...")

    robot = Robot(args.robot_ip)
    state = robot.state
    model = robot.model

    print(f"[KIN] q={np.array2string(array(state.q), precision=6)}")
    print(f"[KIN] O_T_EE translation={np.array2string(translation(state.O_T_EE), precision=6)}")
    print(f"[KIN] F_T_EE translation={np.array2string(translation(state.F_T_EE), precision=6)}")
    print(f"[KIN] EE_T_K translation={np.array2string(translation(state.EE_T_K), precision=6)}")

    frames = [
        Frame.Joint1,
        Frame.Joint2,
        Frame.Joint3,
        Frame.Joint4,
        Frame.Joint5,
        Frame.Joint6,
        Frame.Joint7,
        Frame.Flange,
        Frame.EndEffector,
        Frame.Stiffness,
    ]
    for frame in frames:
        try:
            pose_from_state = model.pose(frame, state)
            pose_from_q = model.pose(frame, state.q, state.F_T_EE, state.EE_T_K)
            print(
                f"[KIN] {frame.name:>11} state={np.array2string(translation(pose_from_state), precision=6)} "
                f"q={np.array2string(translation(pose_from_q), precision=6)}"
            )
        except Exception as exc:
            print(f"[KIN] {frame!s:>11} ERROR {type(exc).__name__}: {exc}")

    for name, fn in (
        ("zero_jacobian", model.zero_jacobian),
        ("body_jacobian", model.body_jacobian),
    ):
        try:
            jacobian = np.asarray(fn(Frame.EndEffector, state), dtype=float)
            print(f"[KIN] {name} EndEffector shape={jacobian.shape}")
            print(np.array2string(jacobian, precision=6, suppress_small=True))
        except Exception as exc:
            print(f"[KIN] {name} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
