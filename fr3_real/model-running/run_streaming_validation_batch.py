#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/run_streaming_validation_batch.py --help
"""Run one checkpoint across validation positions with automatic local streaming.

The policy server must already be serving the requested checkpoint. Each cell
uses either visual gripper confirmation or the per-trial runner's guarded
automatic gripper checks. Approach failures are logged and recovered
automatically when recovery is safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import CONFIGS_DIR, EVAL_LOGS_DIR

RUN_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_LOGS_DIR / "fr3_policy_eval_results.jsonl"


def _resolved_manifest(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else CONFIGS_DIR / candidate).resolve()


def _completed_result(
    manifest: Path, checkpoint: str, eval_id: str, trial: int
) -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    expected_manifest = _resolved_manifest(manifest)
    for line in RESULTS_PATH.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if (
            _resolved_manifest(row.get("manifest", "")) == expected_manifest
            and row.get("checkpoint") == checkpoint
            and row.get("eval_id") == eval_id
            and row.get("trial") == trial
            and row.get("outcome") in {"PASS", "FAIL"}
        ):
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--eval-ids", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=CONFIGS_DIR / "fr3_spatial_validation_v3.json")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--server-host", default="10.6.38.133")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--dynamics-factor", type=float, default=0.085)
    parser.add_argument("--min-z", type=float, default=0.020)
    parser.add_argument("--streamer-cpu", default="2")
    parser.add_argument("--validator-cpu", default="3")
    parser.add_argument("--feeder-cpus", default="4-15")
    parser.add_argument(
        "--controller", choices=("current", "reference"), default="current"
    )
    parser.add_argument(
        "--auto-gripper",
        action="store_true",
        help=(
            "Automatically close after a verified policy transition and release only "
            "while an object is held and the release TCP is inside the basket."
        ),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Refusing robot motion without --run")

    print(
        f"[BATCH] checkpoint={args.checkpoint} trial={args.trial} "
        f"controller={args.controller} cells={','.join(args.eval_ids)}. "
        "The policy server must already serve this checkpoint."
    )
    if input("[BATCH] Workspace clear, cube in basket, E-stop reachable. Type START to begin: ") != "START":
        raise SystemExit("Cancelled")

    failures = 0
    for index, eval_id in enumerate(args.eval_ids, start=1):
        completed_result = _completed_result(
            args.manifest, args.checkpoint, eval_id, args.trial
        )
        if completed_result is not None:
            print(
                f"[BATCH] [{index}/{len(args.eval_ids)}] skipping {eval_id}: "
                f"already recorded {completed_result['outcome']} at "
                f"{completed_result['stage']}"
            )
            continue
        print(f"[BATCH] [{index}/{len(args.eval_ids)}] starting {eval_id}")
        command = [
            sys.executable,
            "run_streaming_validation_trial.py",
            "--eval-id",
            eval_id,
            "--trial",
            str(args.trial),
            "--checkpoint",
            args.checkpoint,
            "--manifest",
            str(args.manifest),
            "--robot-ip",
            args.robot_ip,
            "--server-host",
            args.server_host,
            "--server-port",
            str(args.server_port),
            "--dynamics-factor",
            str(args.dynamics_factor),
            "--min-z",
            str(args.min_z),
            "--streamer-cpu",
            args.streamer_cpu,
            "--validator-cpu",
            args.validator_cpu,
            "--feeder-cpus",
            args.feeder_cpus,
            "--controller",
            args.controller,
            "--auto-start-streamer",
            "--run",
        ]
        if args.auto_gripper:
            command.append("--auto-gripper")
        print("[BATCH] $", " ".join(command))
        completed = subprocess.run(command, cwd=RUN_DIR)
        if completed.returncode:
            failures += 1
            print(f"[BATCH] {eval_id} runner error exit={completed.returncode}; stopping batch.")
            break
        if _completed_result(args.manifest, args.checkpoint, eval_id, args.trial) is None:
            failures += 1
            print(
                f"[BATCH] {eval_id} exited without a valid PASS/FAIL ledger entry; "
                "stopping batch."
            )
            break

    if failures:
        raise SystemExit(1)
    print("[BATCH] completed requested validation positions.")


if __name__ == "__main__":
    main()
