#!/usr/bin/env python3
"""Camera-only RealSense recording sanity check."""

import argparse
import time

from realsense_recorder import RealSenseEpisodeRecorder


parser = argparse.ArgumentParser()
parser.add_argument("--record_dir", type=str, default="recordings/camera_test")
parser.add_argument("--camera_serial", type=str, action="append", default=None)
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_fps", type=int, default=30)
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()

recorder = RealSenseEpisodeRecorder(
    args.record_dir,
    serials=args.camera_serial,
    width=args.camera_width,
    height=args.camera_height,
    fps=args.camera_fps,
)
print(f"[record] cameras: {', '.join(recorder.camera_names)}")
recorder.start_episode(
    time.strftime("%Y%m%d_%H%M%S_camera_test"),
    {"mode": "camera_only_test", "duration_s": args.seconds},
)
time.sleep(args.seconds)
recorder.stop_episode(True, {"note": "camera-only test complete"})
recorder.close()
print("[record] done")
