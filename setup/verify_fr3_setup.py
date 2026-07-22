#!/usr/bin/env python3
"""Report whether an FR3 workstation is ready for Python, cameras, and streaming.

This check does not connect to the robot, move the arm, command the gripper, or
open cameras. It only inspects local imports, files, executables, and network
configuration supplied by the operator.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import socket
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FR3_ROOT = REPO_ROOT / "fr3_real"


def check(label: str, passed: bool, detail: str) -> bool:
    status = "OK" if passed else "MISSING"
    print(f"[{status:7}] {label}: {detail}")
    return passed


def import_check(module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
    except Exception as error:  # Report native-library/import errors verbatim.
        return False, str(error)
    return True, "imported"


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true", help="Skip host and project checks.")
    parser.add_argument("--robot-ip", default="172.16.0.2", help="Printed only; never contacted.")
    parser.add_argument("--server-host", default="10.6.38.133")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--check-policy-server", action="store_true", help="Attempt a TCP connection only.")
    args = parser.parse_args()

    print(f"[INFO   ] python: {sys.executable} ({sys.version.split()[0]})")
    passed = True
    for module in (
        "numpy",
        "cv2",
        "pyrealsense2",
        "franky",
        "openpi_client",
        "pyarrow.parquet",
    ):
        ok, detail = import_check(module)
        passed &= check(f"python:{module}", ok, detail)

    if args.python_only:
        raise SystemExit(0 if passed else 1)

    passed &= check("repository", (FR3_ROOT / "repo_paths.py").is_file(), str(FR3_ROOT))
    passed &= check(
        "calibration",
        (FR3_ROOT / "configs/probed_points.json").is_file(),
        str(FR3_ROOT / "configs/probed_points.json"),
    )
    passed &= check(
        "streamer source",
        (FR3_ROOT / "model-running/streaming/CMakeLists.txt").is_file(),
        str(FR3_ROOT / "model-running/streaming"),
    )
    passed &= check("cmake", shutil.which("cmake") is not None, shutil.which("cmake") or "not on PATH")
    passed &= check("C++ compiler", shutil.which("g++") is not None, shutil.which("g++") or "not on PATH")
    print(f"[INFO   ] robot target: {args.robot_ip} (not contacted)")

    if args.check_policy_server:
        ok = port_open(args.server_host, args.server_port)
        passed &= check(
            "policy server",
            ok,
            f"{args.server_host}:{args.server_port}",
        )
    else:
        print("[INFO   ] policy server: not checked; add --check-policy-server for a TCP-only check")

    if not passed:
        print("[NEXT   ] Install/fix the MISSING items before live robot operation.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
