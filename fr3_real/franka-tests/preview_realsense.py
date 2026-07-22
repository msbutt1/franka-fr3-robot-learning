#!/usr/bin/env python3
# Usage: from fr3_real/, run: python franka-tests/preview_realsense.py --help
"""Live RGB preview for one or more Intel RealSense cameras."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def discover_serials() -> list[str]:
    context = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in context.devices]


parser = argparse.ArgumentParser()
parser.add_argument("--camera_serial", type=str, action="append", default=None)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

serials = args.camera_serial or discover_serials()
if not serials:
    raise SystemExit("No RealSense cameras found.")

pipelines = []
names = []
try:
    for idx, serial in enumerate(serials):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipeline.start(config)
        pipelines.append(pipeline)
        names.append(f"camera_{idx}_{serial}")

    print("[preview] cameras: " + ", ".join(names))
    print("[preview] press q or Esc in the preview window to quit.")
    time.sleep(0.5)

    while True:
        frames_for_display = []
        for name, pipeline in zip(names, pipelines, strict=True):
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            cv2.putText(
                frame,
                name,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            frames_for_display.append(frame)

        if not frames_for_display:
            continue
        preview = frames_for_display[0] if len(frames_for_display) == 1 else np.hstack(frames_for_display)
        cv2.imshow("RealSense RGB Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
finally:
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
