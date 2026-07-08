"""Shared FR3 motion helpers aligned with libfranka example patterns."""

from __future__ import annotations

import numpy as np
from franky import Affine, CartesianMotion, ReferenceType, RobotPose


LOWER_TORQUE_THRESHOLDS = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
UPPER_TORQUE_THRESHOLDS = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
LOWER_FORCE_THRESHOLDS = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
UPPER_FORCE_THRESHOLDS = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
# probe_points.py records robot.current_pose.end_effector_pose.translation while
# the physical tip is placed on the table. The safest default is therefore no
# extra offset: command the same frame that was probed. Override only after a
# live touch test proves the reported pose is a flange frame.
DEFAULT_FRANKA_HAND_TCP_OFFSET = 0.0


def clear_robot_errors(robot) -> None:
    if robot.has_errors:
        print("[robot] robot reports existing error/reflex state — clearing before proceeding...")
        recovered = robot.recover_from_errors()
        print(f"[robot] recover_from_errors() -> {recovered}")
        if not recovered or robot.has_errors:
            raise SystemExit("[robot] could not clear error state — check Desk before retrying.")
    else:
        print("[robot] no existing errors.")


def configure_collision_behavior(robot) -> None:
    """Apply the libfranka example thresholds when the binding exposes them."""
    if not hasattr(robot, "set_collision_behavior"):
        print("[robot] set_collision_behavior() not exposed by this Python binding; using controller defaults.")
        return

    robot.set_collision_behavior(
        LOWER_TORQUE_THRESHOLDS,
        UPPER_TORQUE_THRESHOLDS,
        LOWER_FORCE_THRESHOLDS,
        UPPER_FORCE_THRESHOLDS,
    )
    print("[robot] collision behavior set from libfranka example defaults.")


class MotionPlanner:
    def __init__(
        self,
        robot,
        dynamics_factor: float,
        max_step: float,
        travel_z_floor: float,
    ) -> None:
        self.robot = robot
        self.dynamics_factor = dynamics_factor
        self.max_step = max_step
        self.travel_z_floor = travel_z_floor

    def current_xyz(self) -> np.ndarray:
        return np.array(self.robot.current_pose.end_effector_pose.translation, dtype=float)

    def _absolute_pose_at(self, target_xyz) -> RobotPose:
        current_pose = self.robot.current_pose
        current_ee = current_pose.end_effector_pose
        target_ee = Affine(np.array(target_xyz, dtype=float), np.array(current_ee.quaternion, dtype=float))
        return RobotPose(target_ee, current_pose.elbow_state)

    def safe_move_to(self, target_xyz, label: str = "") -> None:
        target_xyz = np.array(target_xyz, dtype=float)
        for attempt in (1, 2):
            current = self.current_xyz()
            delta = target_xyz - current
            try:
                self.robot.move(CartesianMotion(self._absolute_pose_at(target_xyz), ReferenceType.Absolute))
                return
            except Exception as e:
                if "reflex" not in str(e).lower() or attempt == 2:
                    raise
                print(f"    [reflex] {label or 'move'} aborted mid-motion — recovering, retrying at half speed...")
                print(f"             current=({current[0]:+.3f},{current[1]:+.3f},{current[2]:+.3f}) "
                      f"target=({target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}) "
                      f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})")
                self.robot.recover_from_errors()
                self.robot.relative_dynamics_factor = self.dynamics_factor / 2

    def move_in_steps(self, target_xyz, label: str = "", max_step: float | None = None) -> None:
        target_xyz = np.array(target_xyz, dtype=float)
        if max_step is None:
            max_step = self.max_step
        if max_step <= 0:
            self.safe_move_to(target_xyz, label)
            return
        current = self.current_xyz()
        dist = float(np.linalg.norm(target_xyz - current))
        n_steps = max(1, int(np.ceil(dist / max_step)))
        for i in range(1, n_steps + 1):
            waypoint = current + (target_xyz - current) * (i / n_steps)
            self.safe_move_to(waypoint, f"{label} [{i}/{n_steps}]")

    def travel_to(self, xy_target, end_z: float, label: str = "", max_step: float | None = None) -> None:
        if max_step is None:
            max_step = self.max_step
        xy_target = np.array(xy_target, dtype=float)
        current = self.current_xyz()
        travel_z = max(float(end_z), self.travel_z_floor)
        if current[2] > travel_z:
            self.move_in_steps(np.array([xy_target[0], xy_target[1], current[2]]), f"{label}: horizontal", max_step)
            self.move_in_steps(np.array([xy_target[0], xy_target[1], travel_z]), f"{label}: descend to travel", max_step)
        else:
            self.move_in_steps(np.array([current[0], current[1], travel_z]), f"{label}: rise", max_step)
            self.move_in_steps(np.array([xy_target[0], xy_target[1], travel_z]), f"{label}: horizontal", max_step)
        self.move_in_steps(np.array([xy_target[0], xy_target[1], end_z]), f"{label}: descend", max_step)

    def travel_to_with_descent_pause(
        self,
        xy_target,
        end_z: float,
        label: str = "",
        prompt: str = "Press Enter to descend...",
        max_step: float | None = None,
    ) -> None:
        if max_step is None:
            max_step = self.max_step
        xy_target = np.array(xy_target, dtype=float)
        current = self.current_xyz()
        travel_z = max(float(end_z), self.travel_z_floor)
        if current[2] > travel_z:
            self.move_in_steps(np.array([xy_target[0], xy_target[1], current[2]]), f"{label}: horizontal", max_step)
            print(f"    [pause] above target at current z: x={xy_target[0]:+.3f} y={xy_target[1]:+.3f} z={current[2]:+.3f}")
        else:
            self.move_in_steps(np.array([current[0], current[1], travel_z]), f"{label}: rise", max_step)
            self.move_in_steps(np.array([xy_target[0], xy_target[1], travel_z]), f"{label}: horizontal", max_step)
            print(f"    [pause] at travel plane over target: x={xy_target[0]:+.3f} y={xy_target[1]:+.3f} z={travel_z:+.3f}")
        input(prompt)
        self.move_in_steps(np.array([xy_target[0], xy_target[1], end_z]), f"{label}: descend", max_step)

    def travel_above(self, xy_target, label: str = "", max_step: float | None = None) -> np.ndarray:
        if max_step is None:
            max_step = self.max_step
        xy_target = np.array(xy_target, dtype=float)
        current = self.current_xyz()
        target = np.array([xy_target[0], xy_target[1], self.travel_z_floor])
        self.move_in_steps(np.array([current[0], current[1], self.travel_z_floor]), f"{label}: rise", max_step)
        self.move_in_steps(target, f"{label}: horizontal", max_step)
        return target


def compute_travel_z_floor(home_xyz, z_candidates, extra_clearance: float, include_home_z: bool = True) -> float:
    max_candidate_z = max(float(z) for z in z_candidates)
    if include_home_z:
        return max(float(home_xyz[2]), max_candidate_z) + extra_clearance
    return max_candidate_z + extra_clearance


def flange_z_for_tcp_z(tcp_z: float, tcp_offset: float) -> float:
    return float(tcp_z) + float(tcp_offset)


def assert_safe_flange_z(label: str, flange_z: float, min_flange_z: float) -> None:
    if float(flange_z) < float(min_flange_z):
        raise SystemExit(
            f"[SAFETY] Refusing {label}: target flange/control-frame z={flange_z:.4f} "
            f"is below min_flange_z={min_flange_z:.4f}."
        )


def reset_dynamics(robot, dynamics_factor: float) -> None:
    robot.relative_dynamics_factor = dynamics_factor
