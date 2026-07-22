#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/run_streaming_validation_trial.py --help
"""Orchestrate one timestamped receding-horizon FR3 validation trial.

The operator starts the pinned streamer in a second terminal when prompted.
This script owns setup, policy feeder phases, gripper approval, result logging,
and automatic recovery when no grasp was executed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grid_utils import basket_polygon_from_points, point_in_polygon_xy
from repo_paths import CONFIGS_DIR, EVAL_LOGS_DIR


RUN_DIR = Path(__file__).resolve().parent


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("[TRIAL] $", " ".join(command))
    completed = subprocess.run(command, cwd=RUN_DIR, text=True, capture_output=capture)
    if capture:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, default=CONFIGS_DIR / "fr3_spatial_validation_v3.json")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--server-host", default="10.6.38.133")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--dynamics-factor", type=float, default=0.085)
    parser.add_argument("--points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument(
        "--min-z",
        type=float,
        default=0.020,
        help="Calibrated control-frame floor shared by setup, recovery, and the streamer.",
    )
    parser.add_argument("--close-threshold", type=float, default=0.45)
    parser.add_argument(
        "--manual-target",
        action="store_true",
        help="Skip scripted basket setup; use a manually placed cube and arm at policy start.",
    )
    parser.add_argument(
        "--manual-live",
        action="store_true",
        help="Reset the arm to policy start and use an arbitrary manually placed cube.",
    )
    parser.add_argument("--release-threshold", type=float, default=0.10)
    parser.add_argument("--replan-at-step", type=int, default=8)
    parser.add_argument("--max-steady-latency", type=float, default=0.75)
    parser.add_argument(
        "--guard-margin",
        type=float,
        nargs=3,
        default=[0.005, 0.005, 0.002],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--auto-start-streamer",
        action="store_true",
        help="Launch the pinned local streamer for each arm phase instead of prompting for a second terminal.",
    )
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
        help="Local joint-velocity controller implementation.",
    )
    parser.add_argument(
        "--auto-gripper",
        action="store_true",
        help=(
            "Automatically execute policy-triggered CLOSE/OPEN. Grasp, held-object, "
            "release TCP, and open-width checks remain mandatory."
        ),
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run a commissioning repeat without replacing or adding an evaluation result.",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Refusing robot motion without --run")

    manifest = str(args.manifest)
    common = [
        sys.executable,
        "run_fr3_policy_eval_trial.py",
        "--robot_ip",
        args.robot_ip,
        "--server_host",
        args.server_host,
        "--server_port",
        str(args.server_port),
        "--manifest",
        manifest,
        "--points",
        str(args.points),
        "--eval_id",
        args.eval_id,
        "--dynamics_factor",
        str(args.dynamics_factor),
        "--min_z",
        str(args.min_z),
        "--min_flange_z",
        str(args.min_z),
        "--run",
    ]

    print(
        f"[TRIAL] checkpoint={args.checkpoint} eval={args.eval_id} "
        f"trial={args.trial} controller={args.controller}"
    )
    if args.manual_target and args.manual_live:
        raise SystemExit("Use only one of --manual-target or --manual-live")
    setup_flags = ["--setup-only"]
    if args.manual_target:
        setup_flags.append("--manual-target")
    if args.manual_live:
        setup_flags.append("--manual-live")
    setup = run(common + setup_flags)
    if setup.returncode:
        raise SystemExit("Setup failed; no policy rollout was started")

    warmup = run(
        [
            sys.executable,
            "streaming/warm_policy_server.py",
            "--robot-ip",
            args.robot_ip,
            "--server-host",
            args.server_host,
            "--server-port",
            str(args.server_port),
            "--checkpoint-label",
            args.checkpoint,
            "--iterations",
            "3",
            "--max-steady-latency",
            str(args.max_steady_latency),
        ]
    )
    if warmup.returncode:
        _finish(
            args,
            manifest,
            "infrastructure",
            "policy server failed pre-FCI warmup or latency validation",
            recover=not args.manual_live,
            invalid=True,
        )
        raise SystemExit(3)

    streamer_binary = {
        "current": "./streaming/build/fr3_joint_velocity_streamer",
        "reference": "./streaming/build/fr3_reference_joint_velocity_streamer",
    }[args.controller]
    streamer_hint = (
        f"In another terminal run: taskset -c 2 {streamer_binary} "
        "172.16.0.2 --command-port 51000 --state-port 51001 --velocity-caps "
        "0.41 0.47 0.025 0.60 0.14 0.46 0.39 --acceleration-caps "
        f"1.75 3.17 0.34 2.57 0.85 3.19 1.77 --min-z {args.min_z:.3f}; "
        "type START there, then press Enter here. "
        "Ctrl+C in the feeder or streamer stops arm motion."
    )
    with _streamer(args, streamer_hint, "approach") as approach_log:
        feeder = _run_feeder(args, holding=False)
    if _streamer_fault(approach_log):
        _finish(
            args,
            manifest,
            "infrastructure",
            "local streamer or libfranka fault during approach; manual recovery required",
            recover=False,
            invalid=True,
        )
        raise SystemExit(2)
    close_marker = "policy requested gripper closure" in feeder.stdout
    if feeder.returncode or not close_marker:
        if _is_infrastructure_invalid(feeder):
            _finish(
                args,
                manifest,
                "infrastructure",
                "policy inference exceeded the streaming plan budget; trial is invalid",
                recover=False,
                invalid=True,
            )
            raise SystemExit(3)
        stage = "approach"
        safety_stop = _is_safety_stop(feeder)
        notes = (
            "policy rollout reached a safety guard before grasp"
            if safety_stop
            else "streamer stopped before an approved grasp"
        )
        _finish(args, manifest, stage, notes, recover=not safety_stop and not args.manual_live)
        if safety_stop:
            print("[TRIAL] Safety stop: scripted recovery is disabled; inspect and recover manually.")
            raise SystemExit(2)
        return

    grasp_command = [sys.executable, "manual_fr3_policy_grasp.py", "--robot-ip", args.robot_ip]
    if args.auto_gripper:
        grasp_command.append("--yes")
    grasp = run(grasp_command)
    grasp_verified = grasp.returncode == 0
    if not args.auto_gripper:
        grasp_verified = grasp_verified and input(
            "[TRIAL] Type HELD only if the grasp visibly verified; otherwise FAIL: "
        ).strip().upper() == "HELD"
    if not grasp_verified:
        _finish(args, manifest, "grasp", "grasp not verified", recover=not args.manual_live)
        return

    with _streamer(args, streamer_hint, "transport") as transport_log:
        transport = _run_feeder(args, holding=True)
    if _streamer_fault(transport_log):
        _finish(
            args,
            manifest,
            "infrastructure",
            "local streamer or libfranka fault while holding; manual recovery required",
            recover=False,
            invalid=True,
        )
        raise SystemExit(2)
    release_marker = "policy requested gripper release" in transport.stdout
    if transport.returncode or not release_marker:
        if _is_infrastructure_invalid(transport):
            _finish(
                args,
                manifest,
                "infrastructure",
                "policy inference exceeded the streaming plan budget; trial is invalid",
                recover=False,
                invalid=True,
            )
            raise SystemExit(3)
        _finish(args, manifest, "transport", "streamer stopped before release request", recover=False)
        if _is_safety_stop(transport):
            print("[TRIAL] Safety stop while holding: leave the arm stopped and recover manually.")
        raise SystemExit(2)

    release_command = [
        sys.executable,
        "manual_fr3_policy_grasp.py",
        "--robot-ip",
        args.robot_ip,
        "--release",
    ]
    notes = "streaming validation"
    if args.auto_gripper:
        release_tcp = _transition_tcp(transport, "release")
        if release_tcp is None or not _release_tcp_inside_basket(args.points, release_tcp):
            _finish(
                args,
                manifest,
                "transport",
                "automatic release refused: release TCP was not verified inside the basket",
                recover=False,
            )
            print("[TRIAL] Cube may still be held; recover manually.")
            raise SystemExit(2)
        print(f"[TRIAL] automatic release TCP verified inside basket: {release_tcp}")
        release_command += ["--yes", "--require-held"]
        notes = "automated gripper; held object and in-basket release TCP verified"
    release = run(release_command)
    passed = release.returncode == 0
    if not args.auto_gripper:
        passed = passed and input(
            "[TRIAL] Type PASS only if the cube was released into the basket; otherwise FAIL: "
        ).strip().upper() == "PASS"
    _finish(args, manifest, "release", notes, recover=False, passed=passed)
    if passed:
        reset = _post_release_reset(args, manifest)
        if reset.returncode:
            print(
                "[TRIAL] PASS remains recorded at release, but post-release turnover failed. "
                "Inspect the stopped arm before another trial."
            )
            raise SystemExit(2)
        print("[TRIAL] post-release turnover complete; arm returned to policy start.")


def _run_feeder(args: argparse.Namespace, *, holding: bool) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "streaming/stream_policy_chunks.py",
        "--server-host",
        args.server_host,
        "--server-port",
        str(args.server_port),
        "--execute-steps",
        "16",
        "--replan-at-step",
        str(args.replan_at_step),
        "--run-id",
        f"{args.checkpoint}_{args.controller}_{args.eval_id}_trial{args.trial}_"
        f"{'transport' if holding else 'approach'}_{time.time_ns()}",
    ]
    if holding:
        command += [
            "--holding",
            "--gripper-closedness",
            "0.50",
            "--stop-on-release-threshold",
            str(args.release_threshold),
        ]
    else:
        command += ["--stop-on-gripper-threshold", str(args.close_threshold)]
    if args.auto_start_streamer and args.feeder_cpus:
        command = ["taskset", "-c", args.feeder_cpus, "nice", "-n", "10"] + command
    return run(command, capture=True)


def _post_release_reset(
    args: argparse.Namespace, manifest: str
) -> subprocess.CompletedProcess[str]:
    print(
        "[TRIAL] Release result finalized. Running guarded scripted turnover; "
        "this motion is not part of the policy score."
    )
    return run(
        [
            sys.executable,
            "run_fr3_policy_eval_trial.py",
            "--robot_ip",
            args.robot_ip,
            "--manifest",
            manifest,
            "--points",
            str(args.points),
            "--eval_id",
            args.eval_id,
            "--dynamics_factor",
            str(args.dynamics_factor),
            "--min_z",
            str(args.min_z),
            "--min_flange_z",
            str(args.min_z),
            "--run",
            "--reset-only",
            "--assume-confirmed",
        ]
    )


@contextlib.contextmanager
def _streamer(args: argparse.Namespace, hint: str, phase: str):
    if not args.auto_start_streamer:
        input(f"[TRIAL] {hint}")
        yield None
        return

    log_dir = EVAL_LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"streamer_{args.checkpoint}_{args.controller}_{args.eval_id}_{args.trial}_{phase}.log"
    )
    streamer_binary = {
        "current": "./streaming/build/fr3_joint_velocity_streamer",
        "reference": "./streaming/build/fr3_reference_joint_velocity_streamer",
    }[args.controller]
    streamer_cpu_set = (
        f"{args.streamer_cpu},{args.validator_cpu}"
        if args.controller == "reference"
        else args.streamer_cpu
    )
    command = [
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
        *(str(value) for value in args.guard_margin),
    ]
    print(f"[TRIAL] starting pinned {phase} streamer; log={log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            time.sleep(0.4)
            if process.poll() is not None:
                raise RuntimeError(f"Streamer exited before start; inspect {log_path}")
            assert process.stdin is not None
            process.stdin.write("START\n")
            process.stdin.flush()
            time.sleep(0.4)
            if process.poll() is not None:
                raise RuntimeError(f"Streamer rejected START; inspect {log_path}")
            yield log_path
        finally:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print(f"[TRIAL] {phase} streamer exit={process.returncode}; log={log_path}")


def _is_infrastructure_invalid(feeder: subprocess.CompletedProcess[str]) -> bool:
    return feeder.returncode == 3 or "[FEEDER] INVALID" in feeder.stdout


def _is_safety_stop(feeder: subprocess.CompletedProcess[str]) -> bool:
    return any(
        token in feeder.stdout
        for token in ("workspace_guard", "joint_guard", "predicted_guard")
    )


def _streamer_fault(log_path: Path | None) -> bool:
    if log_path is None or not log_path.exists():
        return False
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    return "[STREAM] libfranka error:" in contents or "[STREAM] error:" in contents


def _transition_tcp(
    feeder: subprocess.CompletedProcess[str], transition_kind: str
) -> list[float] | None:
    match = re.search(r"telemetry=(\S+)", feeder.stdout)
    if match is None:
        return None
    path = Path(match.group(1))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    transition = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row.get("event") == "gripper_transition_reached"
            and row.get("transition_kind") == transition_kind
        ):
            transition = row
    if transition is None:
        return None
    tcp = transition.get("actual_tcp")
    if not isinstance(tcp, list) or len(tcp) != 3:
        return None
    return [float(value) for value in tcp]


def _release_tcp_inside_basket(points_path: Path, tcp: list[float]) -> bool:
    points = json.loads(points_path.read_text())
    polygon = basket_polygon_from_points(points)
    if polygon is None or not point_in_polygon_xy(np.asarray(tcp[:2]), polygon):
        return False
    pad_z = float(points["pad_center"][2])
    rim_z = float(points["basket_rim"][2])
    return pad_z - 0.01 <= tcp[2] <= rim_z + 0.10


def _finish(
    args: argparse.Namespace,
    manifest: str,
    stage: str,
    notes: str,
    *,
    recover: bool,
    passed: bool = False,
    invalid: bool = False,
) -> None:
    notes = f"controller={args.controller}; {notes}"
    outcome = "INVALID" if invalid else ("PASS" if passed else "FAIL")
    record_command = [
        sys.executable,
        "record_fr3_eval_result.py",
        "--manifest",
        manifest,
        "--eval-id",
        args.eval_id,
        "--trial",
        str(args.trial),
        "--checkpoint",
        args.checkpoint,
        "--outcome",
        outcome,
        "--stage",
        stage,
        "--notes",
        notes,
    ]
    if args.no_record:
        print(f"[TRIAL] no-record commissioning result={outcome} stage={stage} notes={notes}")
    else:
        results_path = ROOT / "eval_logs/fr3_policy_eval_results.jsonl"
        if results_path.exists():
            existing = [
                json.loads(line) for line in results_path.read_text().splitlines() if line
            ]
            replace_invalid = any(
                row.get("manifest") == manifest
                and row.get("eval_id") == args.eval_id
                and row.get("trial") == args.trial
                and row.get("checkpoint") == args.checkpoint
                and row.get("outcome") == "INVALID"
                for row in existing
            )
            if replace_invalid:
                record_command.append("--replace")
        run(record_command)
    if recover:
        print("[TRIAL] No verified grasp. Recovering the cube from the cell to the basket.")
        recovery = run(
            [
                sys.executable,
                "run_fr3_policy_eval_trial.py",
                "--robot_ip",
                args.robot_ip,
                "--manifest",
                manifest,
                "--points",
                str(args.points),
                "--eval_id",
                args.eval_id,
                "--dynamics_factor",
                str(args.dynamics_factor),
                "--min_z",
                str(args.min_z),
                "--min_flange_z",
                str(args.min_z),
                "--run",
                "--recover-only",
            ]
        )
        if recovery.returncode:
            print(
                "[TRIAL] Scripted recovery failed; stopping before another trial. "
                "Inspect the arm and cube manually."
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
