"""Minimal multi-RealSense RGB recorder for FR3 pick-and-place episodes."""
# Usage: import this shared module from an FR3 script; it is not a standalone command.

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np


class RealSenseEpisodeRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        serials: list[str] | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        import cv2  # noqa: PLC0415
        import pyrealsense2 as rs  # noqa: PLC0415

        self.cv2 = cv2
        self.rs = rs
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.serials = serials or self._discover_serials()
        if not self.serials:
            raise RuntimeError("No Intel RealSense cameras found.")

        self.pipelines = []
        self.camera_names = []
        self.camera_frame_counts = {}
        self.camera_errors = {}
        for idx, serial in enumerate(self.serials):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline.start(config)
            self.pipelines.append(pipeline)
            self.camera_names.append(f"camera_{idx}_{serial}")
            self.camera_frame_counts[f"camera_{idx}_{serial}"] = 0
            self.camera_errors[f"camera_{idx}_{serial}"] = []

        self.active = False
        self.threads: list[threading.Thread] = []
        self.stop_event = threading.Event()
        self.episode_dir: Path | None = None
        self.writers = {}
        self.timestamp_files = {}
        self.metadata = {}
        self.robot = None
        self.gripper = None
        self.state_sample_hz = 30.0
        self.state_thread: threading.Thread | None = None
        self.gripper_state_thread: threading.Thread | None = None
        self.state_stop_event = threading.Event()
        self.state_samples = []
        self.action_records = []
        self.gripper_command_width: float | None = None
        self.latest_gripper_state: dict = {}

    def _discover_serials(self) -> list[str]:
        context = self.rs.context()
        return [device.get_info(self.rs.camera_info.serial_number) for device in context.devices]

    def attach_robot(self, robot, gripper=None, sample_hz: float = 30.0) -> None:
        self.robot = robot
        self.gripper = gripper
        self.state_sample_hz = sample_hz
        if gripper is not None:
            gripper_state = gripper.state
            self.gripper_command_width = float(gripper_state.width)
            self.latest_gripper_state = {
                "gripper_state_time_ns": time.time_ns(),
                "gripper_width": float(gripper_state.width),
                "gripper_is_grasped": bool(gripper_state.is_grasped),
            }
        sample = self._read_robot_state_sample()
        print(
            f"[record] robot-state preflight ready: "
            f"q={len(sample['q'])} dq_d={len(sample['dq_d'])} "
            f"gripper_width={sample.get('gripper_width', float('nan')):.4f}"
        )

    def set_gripper_command(self, width: float, command: str) -> None:
        self.gripper_command_width = float(width)
        self.log_action(command, {"target_width": self.gripper_command_width})

    def start_episode(self, episode_name: str, metadata: dict) -> Path:
        if self.active:
            raise RuntimeError("Recorder episode already active.")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in episode_name)
        self.episode_dir = self.output_dir / safe_name
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.camera_frame_counts = {name: 0 for name in self.camera_names}
        self.camera_errors = {name: [] for name in self.camera_names}
        self.metadata = {
            **metadata,
            "episode_name": safe_name,
            "record_start_time_ns": time.time_ns(),
            "cameras": self.camera_names,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
        (self.episode_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

        fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
        self.writers = {}
        self.timestamp_files = {}
        for name in self.camera_names:
            self.writers[name] = self.cv2.VideoWriter(
                str(self.episode_dir / f"{name}_rgb.mp4"),
                fourcc,
                self.fps,
                (self.width, self.height),
            )
            if not self.writers[name].isOpened():
                raise RuntimeError(f"OpenCV could not open video writer for {name}.")
            self.timestamp_files[name] = (self.episode_dir / f"{name}_timestamps.jsonl").open("w")

        self.stop_event.clear()
        self.state_stop_event.clear()
        self.state_samples = []
        self.action_records = []
        self.active = True
        self.threads = []
        for name, pipeline in zip(self.camera_names, self.pipelines):
            thread = threading.Thread(target=self._record_camera_loop, args=(name, pipeline), daemon=True)
            thread.start()
            self.threads.append(thread)
        if self.robot is not None:
            self.state_thread = threading.Thread(target=self._sample_robot_state_loop, daemon=True)
            self.state_thread.start()
        if self.gripper is not None:
            self.gripper_state_thread = threading.Thread(target=self._sample_gripper_state_loop, daemon=True)
            self.gripper_state_thread.start()
        print(f"[record] started episode: {self.episode_dir}")
        return self.episode_dir

    def add_event(self, event: str, payload: dict | None = None) -> None:
        if not self.episode_dir:
            return
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        record = {"time_ns": time.time_ns(), "event": event, "payload": payload or {}}
        with (self.episode_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log_action(self, action: str, payload: dict | None = None) -> None:
        if not self.active:
            return
        record = {"time_ns": time.time_ns(), "action": action, "payload": payload or {}}
        self.action_records.append(record)

    @staticmethod
    def _as_float_list(value) -> list[float]:
        if hasattr(value, "matrix"):
            value = value.matrix
        elif hasattr(value, "linear") and hasattr(value, "angular"):
            value = np.concatenate(
                [np.asarray(value.linear, dtype=float), np.asarray(value.angular, dtype=float)]
            )
        return np.asarray(value, dtype=float).reshape(-1).tolist()

    def _read_robot_state_sample(self) -> dict:
        state = self.robot.state
        pose = state.O_T_EE
        robot_time = getattr(state, "time", 0.0)
        if hasattr(robot_time, "to_sec"):
            robot_time = robot_time.to_sec()
        sample = {
            "time_ns": time.time_ns(),
            "robot_time": float(robot_time),
            "q": self._as_float_list(state.q),
            "dq": self._as_float_list(state.dq),
            "q_d": self._as_float_list(state.q_d),
            "dq_d": self._as_float_list(state.dq_d),
            "ddq_d": self._as_float_list(state.ddq_d),
            "tau_J": self._as_float_list(state.tau_J),
            "tau_J_d": self._as_float_list(state.tau_J_d),
            "tau_ext_hat_filtered": self._as_float_list(state.tau_ext_hat_filtered),
            "ee_translation": self._as_float_list(pose.translation),
            "ee_quaternion": self._as_float_list(pose.quaternion),
            "O_T_EE": self._as_float_list(state.O_T_EE),
            "O_T_EE_d": self._as_float_list(state.O_T_EE_d),
            "O_dP_EE_d": self._as_float_list(state.O_dP_EE_d),
            "O_F_ext_hat_K": self._as_float_list(state.O_F_ext_hat_K),
            "control_command_success_rate": float(state.control_command_success_rate),
        }
        if self.gripper is not None:
            sample.update(self.latest_gripper_state)
            sample["gripper_command_width"] = self.gripper_command_width
        return sample

    def _sample_robot_state_loop(self) -> None:
        period = 1.0 / max(1.0, float(self.state_sample_hz))
        while not self.state_stop_event.is_set():
            try:
                self.state_samples.append(self._read_robot_state_sample())
            except Exception as e:
                self.action_records.append(
                    {"time_ns": time.time_ns(), "action": "state_sample_error", "payload": {"error": str(e)}}
                )
            time.sleep(period)

    def _sample_gripper_state_loop(self) -> None:
        period = 1.0 / min(15.0, max(1.0, float(self.state_sample_hz)))
        while not self.state_stop_event.is_set():
            try:
                state = self.gripper.state
                self.latest_gripper_state = {
                    "gripper_state_time_ns": time.time_ns(),
                    "gripper_width": float(state.width),
                    "gripper_is_grasped": bool(state.is_grasped),
                }
            except Exception as e:
                self.action_records.append(
                    {"time_ns": time.time_ns(), "action": "gripper_state_sample_error", "payload": {"error": str(e)}}
                )
            time.sleep(period)

    def _record_camera_loop(self, name: str, pipeline) -> None:
        while not self.stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    time.sleep(0.001)
                    continue
                frame = np.asanyarray(color.get_data())
                if frame.shape[:2] != (self.height, self.width):
                    frame = self.cv2.resize(frame, (self.width, self.height))
                self.writers[name].write(frame)
                self.camera_frame_counts[name] += 1
                self.timestamp_files[name].write(
                    json.dumps(
                        {
                            "time_ns": time.time_ns(),
                            "frame_number": int(color.get_frame_number()),
                            "sensor_timestamp_ms": float(color.get_timestamp()),
                        }
                    )
                    + "\n"
                )
                if self.camera_frame_counts[name] % max(1, self.fps) == 0:
                    self.timestamp_files[name].flush()
            except Exception as e:
                self.camera_errors[name].append({"time_ns": time.time_ns(), "error": str(e)})
                time.sleep(0.01)
            time.sleep(0.001)

    def stop_episode(self, success: bool, metadata: dict | None = None) -> None:
        if not self.active:
            return
        self.stop_event.set()
        self.state_stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3.0)
        if self.state_thread:
            self.state_thread.join(timeout=3.0)
        if self.gripper_state_thread:
            self.gripper_state_thread.join(timeout=3.0)
        for writer in self.writers.values():
            writer.release()
        for f in self.timestamp_files.values():
            f.close()
        self._write_dataset_files()
        stop_metadata = {
            "record_stop_time_ns": time.time_ns(),
            "success": success,
            "camera_frame_counts": dict(self.camera_frame_counts),
            "camera_errors": dict(self.camera_errors),
            **(metadata or {}),
        }
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.episode_dir / "metadata.json"
        if metadata_path.exists():
            current = json.loads(metadata_path.read_text())
        else:
            current = dict(self.metadata)
        with metadata_path.open("w") as f:
            current.update(stop_metadata)
            f.write(json.dumps(current, indent=2) + "\n")
        print(f"[record] stopped episode: {self.episode_dir} success={success}")
        self.active = False
        self.episode_dir = None
        self.writers = {}
        self.timestamp_files = {}
        self.threads = []
        self.state_thread = None
        self.gripper_state_thread = None

    def _write_dataset_files(self) -> None:
        if self.episode_dir is None:
            return
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        with (self.episode_dir / "robot_state.jsonl").open("w") as f:
            for sample in self.state_samples:
                f.write(json.dumps(sample) + "\n")
        with (self.episode_dir / "actions.jsonl").open("w") as f:
            for action in self.action_records:
                f.write(json.dumps(action) + "\n")
        arrays = {}
        if self.state_samples:
            keys = [
                "time_ns",
                "robot_time",
                "q",
                "dq",
                "q_d",
                "dq_d",
                "ddq_d",
                "tau_J",
                "tau_J_d",
                "tau_ext_hat_filtered",
                "ee_translation",
                "ee_quaternion",
                "O_T_EE",
                "O_T_EE_d",
                "O_dP_EE_d",
                "O_F_ext_hat_K",
                "control_command_success_rate",
            ]
            for key in keys:
                arrays[f"state_{key}"] = np.asarray([sample[key] for sample in self.state_samples])
            if "gripper_width" in self.state_samples[0]:
                arrays["state_gripper_width"] = np.asarray([sample["gripper_width"] for sample in self.state_samples])
                arrays["state_gripper_state_time_ns"] = np.asarray(
                    [sample["gripper_state_time_ns"] for sample in self.state_samples], dtype=np.int64
                )
                arrays["state_gripper_command_width"] = np.asarray(
                    [sample["gripper_command_width"] for sample in self.state_samples], dtype=float
                )
                arrays["state_gripper_is_grasped"] = np.asarray(
                    [sample["gripper_is_grasped"] for sample in self.state_samples], dtype=bool
                )
        if self.action_records:
            arrays["action_time_ns"] = np.asarray([record["time_ns"] for record in self.action_records])
            arrays["action_name"] = np.asarray([record["action"] for record in self.action_records], dtype=str)
            targets = []
            for record in self.action_records:
                target = record.get("payload", {}).get("target_xyz")
                targets.append([np.nan, np.nan, np.nan] if target is None else target)
            arrays["action_target_xyz"] = np.asarray(targets, dtype=float)
        if arrays:
            np.savez_compressed(self.episode_dir / "episode_arrays.npz", **arrays)

    def close(self) -> None:
        try:
            if self.active:
                self.stop_episode(False, {"closed_while_active": True})
        except Exception as e:
            print(f"[record] close warning: could not stop active episode cleanly: {e}")
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"[record] close warning: could not stop pipeline cleanly: {e}")
