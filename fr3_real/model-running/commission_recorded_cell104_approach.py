#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/commission_recorded_cell104_approach.py --help
"""Commission the live executor with the full recorded cell-104 approach."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import CONFIGS_DIR, EVAL_LOGS_DIR

RUN_DIR = Path(__file__).resolve().parent
EPISODE = "20260713_110105_printed_cell_104"
MANIFEST = str(CONFIGS_DIR / "fr3_training_cell_smoke_v1.json")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("[COMMISSION] $", " ".join(command), flush=True)
    return subprocess.run(command, cwd=RUN_DIR, text=True)


def robot_phase_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable,
        "run_fr3_policy_eval_trial.py",
        "--robot_ip",
        args.robot_ip,
        "--manifest",
        MANIFEST,
        "--eval_id",
        "C104",
        "--dynamics_factor",
        str(args.dynamics_factor),
        "--min_z",
        str(args.min_z),
        "--min_flange_z",
        str(args.min_z),
        "--run",
        mode,
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--dynamics-factor", type=float, default=0.085)
    parser.add_argument("--min-z", type=float, default=0.020)
    parser.add_argument("--streamer-cpu", default="2")
    parser.add_argument(
        "--validator-cpu",
        default="3",
        help="Helper core used by the reference guard and UDP threads.",
    )
    parser.add_argument("--feeder-cpus", default="4-15")
    parser.add_argument(
        "--controller",
        choices=("current", "reference"),
        default="current",
        help="Streamer implementation to commission with recorded actions.",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Refusing robot motion without --run")

    print(
        f"[COMMISSION] controller={args.controller}. Recorded actions only: "
        "no cameras, policy inference, or policy gripper command."
    )
    setup = run(robot_phase_command(args, "--setup-only"))
    if setup.returncode:
        raise SystemExit("Cell-104 setup failed; no recorded replay was started")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = EVAL_LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    streamer_log = log_dir / f"streamer_{args.controller}_recorded_C104_{timestamp}.log"
    replay_log = log_dir / f"recorded_{args.controller}_C104_through_grasp_{timestamp}.jsonl"
    streamer_binary = {
        "current": "./streaming/build/fr3_joint_velocity_streamer",
        "reference": "./streaming/build/fr3_reference_joint_velocity_streamer",
    }[args.controller]
    if not (RUN_DIR / streamer_binary).is_file():
        raise SystemExit(
            f"Missing {streamer_binary}; run cmake --build streaming/build -j2 first"
        )
    streamer_cpu_set = (
        f"{args.streamer_cpu},{args.validator_cpu}"
        if args.controller == "reference"
        else args.streamer_cpu
    )
    streamer_command = [
        "taskset",
        "-c",
        streamer_cpu_set,
        streamer_binary,
        args.robot_ip,
        "--command-port",
        "51000",
        "--state-port",
        "51001",
        "--velocity-caps",
        "0.41",
        "0.47",
        "0.025",
        "0.60",
        "0.14",
        "0.46",
        "0.39",
        "--acceleration-caps",
        "1.75",
        "3.17",
        "0.34",
        "2.57",
        "0.85",
        "3.19",
        "1.77",
        "--min-z",
        str(args.min_z),
        "--guard-margin",
        "0.005",
        "0.005",
        "0.002",
    ]
    replay_command = [
        "taskset",
        "-c",
        args.feeder_cpus,
        sys.executable,
        "streaming/replay_recorded_velocity_chunks.py",
        "--episode",
        EPISODE,
        "--through-grasp",
        "--replan-at-step",
        "8",
        "--output",
        str(replay_log),
    ]

    replay_result: subprocess.CompletedProcess[str] | None = None
    streamer_returncode: int | None = None
    try:
        print(f"[COMMISSION] starting streamer; log={streamer_log}")
        with streamer_log.open("w", encoding="utf-8") as handle:
            streamer = subprocess.Popen(
                streamer_command,
                cwd=ROOT,
                text=True,
                stdin=subprocess.PIPE,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                time.sleep(0.5)
                if streamer.poll() is not None:
                    raise RuntimeError(f"Streamer exited before START; inspect {streamer_log}")
                assert streamer.stdin is not None
                streamer.stdin.write("START\n")
                streamer.stdin.flush()
                time.sleep(0.5)
                if streamer.poll() is not None:
                    raise RuntimeError(f"Streamer rejected START; inspect {streamer_log}")
                replay_result = run(replay_command)
                try:
                    streamer_returncode = streamer.wait(timeout=6.0)
                except subprocess.TimeoutExpired:
                    streamer.terminate()
                    streamer_returncode = streamer.wait(timeout=3.0)
            finally:
                if streamer.stdin is not None and not streamer.stdin.closed:
                    streamer.stdin.close()
                if streamer.poll() is None:
                    streamer.terminate()
                    streamer.wait(timeout=3.0)
    finally:
        print("[COMMISSION] recovering the cube from cell 104 to the basket.")
        recovery = run(robot_phase_command(args, "--recover-only"))

    if replay_result is None or replay_result.returncode:
        raise SystemExit(f"Recorded replay failed; inspect {streamer_log} and {replay_log}")
    if streamer_returncode != 0:
        raise SystemExit(f"Streamer exit={streamer_returncode}; inspect {streamer_log}")
    if recovery.returncode:
        raise SystemExit("Replay passed, but automatic cube recovery failed")
    print(
        f"[COMMISSION] PASS controller={args.controller}: full recorded approach "
        "reached the grasp boundary. "
        f"streamer_log={streamer_log} replay_log={replay_log}"
    )


if __name__ == "__main__":
    main()
