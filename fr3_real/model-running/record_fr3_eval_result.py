#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/record_fr3_eval_result.py --help
"""Append one manually observed FR3 evaluation outcome to a JSONL ledger."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import EVAL_LOGS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--outcome", choices=("PASS", "FAIL", "INVALID"), required=True)
    parser.add_argument(
        "--stage",
        choices=("setup", "approach", "grasp", "transport", "release", "infrastructure"),
        required=True,
        help="For PASS use release; for FAIL record the first failed stage.",
    )
    parser.add_argument("--checkpoint", default="7999")
    parser.add_argument("--notes", default="")
    parser.add_argument("--results", type=Path, default=EVAL_LOGS_DIR / "fr3_policy_eval_results.jsonl")
    parser.add_argument("--replace", action="store_true", help="Replace an existing result for this checkpoint and trial.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    positions = {position["eval_id"]: position for position in manifest.get("positions", [])}
    if args.eval_id not in positions:
        raise SystemExit(f"{args.eval_id!r} is not present in {args.manifest}")
    trials = int(positions[args.eval_id].get("trials", manifest.get("trials_per_position", 1)))
    if not 1 <= args.trial <= trials:
        raise SystemExit(f"--trial must be in [1, {trials}]")
    if args.outcome == "PASS" and args.stage != "release":
        raise SystemExit("PASS requires --stage release")
    if args.outcome == "INVALID" and args.stage != "infrastructure":
        raise SystemExit("INVALID requires --stage infrastructure")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.results.exists():
        existing = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    duplicate = [
        row for row in existing
        if row.get("manifest") == str(args.manifest)
        and row.get("eval_id") == args.eval_id
        and row.get("trial") == args.trial
        and row.get("checkpoint") == args.checkpoint
    ]
    if duplicate:
        if not args.replace:
            raise SystemExit("Refusing to overwrite an existing result for this checkpoint and trial")
        existing = [row for row in existing if row not in duplicate]

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "role": positions[args.eval_id].get("role"),
        "eval_id": args.eval_id,
        "trial": args.trial,
        "checkpoint": args.checkpoint,
        "outcome": args.outcome,
        "stage": args.stage,
        "target": {
            key: positions[args.eval_id][key]
            for key in ("x", "y", "table_z", "source_cells")
        },
        "notes": args.notes,
    }
    with args.results.open("w") as handle:
        for row in existing:
            handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps(record) + "\n")
    print(f"[EVAL] recorded {args.outcome} {args.eval_id} trial {args.trial} at {args.stage}")


if __name__ == "__main__":
    main()
