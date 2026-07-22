#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/create_interstitial_eval_manifest.py --help
"""Create a fixed, spatially held-out FR3 evaluation manifest.

Each evaluation position is the midpoint of two adjacent recorded cells, so it
is inside the demonstrated workspace but is not itself a training location.
The manifest is generated once and then treated as test-only: do not use its
outcomes to choose a checkpoint or alter the training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grid_utils import basket_polygon_from_points, inside_basket_exclusion
from repo_paths import CONFIGS_DIR


def midpoint_candidate(a: dict, b: dict, axis: str) -> dict:
    return {
        "x": (float(a["x"]) + float(b["x"])) / 2.0,
        "y": (float(a["y"]) + float(b["y"])) / 2.0,
        "table_z": (float(a["table_z"]) + float(b["table_z"])) / 2.0,
        "axis": axis,
        "source_cells": [int(a["printed_cell"]), int(b["printed_cell"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells_json", type=Path, default=CONFIGS_DIR / "all_working_cells.json")
    parser.add_argument("--probed_points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument("--output", type=Path, default=CONFIGS_DIR / "fr3_interstitial_eval_v1.json")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--trials_per_position", type=int, default=3)
    parser.add_argument(
        "--role",
        default="final_test_only",
        choices=("validation_only", "final_test_only"),
        help="Purpose recorded in every position; validation positions may select checkpoints.",
    )
    parser.add_argument(
        "--exclude_manifest",
        type=Path,
        help="Existing manifest whose positions must not be reused (for validation/test separation).",
    )
    parser.add_argument(
        "--exclude_radius",
        type=float,
        default=0.04,
        help="Minimum XY separation from --exclude_manifest positions in metres.",
    )
    parser.add_argument("--basket_margin", type=float, default=0.04)
    parser.add_argument("--basket_w", type=float, default=0.18)
    parser.add_argument("--basket_h", type=float, default=0.18)
    args = parser.parse_args()

    if args.positions < 1 or args.trials_per_position < 1:
        raise SystemExit("--positions and --trials_per_position must be positive")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {args.output}")

    cells = json.loads(args.cells_json.read_text())["cells"]
    cells = [cell for cell in cells if cell.get("status") == "PASS"]
    if len(cells) < 2:
        raise SystemExit("Need at least two PASS cells")
    by_grid = {(int(cell["grid_i"]), int(cell["grid_j"])): cell for cell in cells}

    points = json.loads(args.probed_points.read_text())
    pad_center = np.asarray(points["pad_center"], dtype=float)
    basket_polygon = basket_polygon_from_points(points)
    excluded_xy = np.empty((0, 2), dtype=float)
    if args.exclude_manifest is not None:
        excluded = json.loads(args.exclude_manifest.read_text()).get("positions", [])
        excluded_xy = np.asarray([[position["x"], position["y"]] for position in excluded], dtype=float)
        if excluded_xy.size == 0:
            excluded_xy = np.empty((0, 2), dtype=float)
    candidates: list[dict] = []
    for (grid_i, grid_j), cell in sorted(by_grid.items()):
        for delta, axis in (((1, 0), "x_midpoint"), ((0, 1), "y_midpoint")):
            neighbor = by_grid.get((grid_i + delta[0], grid_j + delta[1]))
            if neighbor is None:
                continue
            candidate = midpoint_candidate(cell, neighbor, axis)
            xyz = np.asarray([candidate["x"], candidate["y"], candidate["table_z"]], dtype=float)
            if not inside_basket_exclusion(
                xyz, pad_center, args.basket_margin, args.basket_w, args.basket_h, basket_polygon
            ):
                if len(excluded_xy) and np.min(np.linalg.norm(excluded_xy - xyz[:2], axis=1)) < args.exclude_radius:
                    continue
                candidates.append(candidate)
    if len(candidates) < args.positions:
        raise SystemExit(f"Only {len(candidates)} safe interstitial candidates for {args.positions} positions")

    # Deterministic farthest-point sampling spreads the test positions across the
    # workspace rather than concentrating them in an easy local region.
    xy = np.asarray([[candidate["x"], candidate["y"]] for candidate in candidates], dtype=float)
    center = xy.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(xy - center, axis=1)))]
    while len(selected) < args.positions:
        distances = np.min(
            np.linalg.norm(xy[:, None, :] - xy[np.asarray(selected)][None, :, :], axis=2), axis=1
        )
        distances[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distances)))

    positions = []
    for index, candidate_index in enumerate(selected, start=1):
        candidate = dict(candidates[candidate_index])
        candidate.update(
            {
                "eval_id": f"I{index:02d}",
                "trials": args.trials_per_position,
                "role": args.role,
                "notes": (
                    "May be used for checkpoint selection, but never for retraining decisions."
                    if args.role == "validation_only"
                    else "Do not use these outcomes for checkpoint selection or retraining decisions."
                ),
            }
        )
        positions.append(candidate)

    manifest = {
        "description": "FR3 spatial-interpolation final test set.",
        "training_cells_source": str(args.cells_json),
        "total_positions": len(positions),
        "trials_per_position": args.trials_per_position,
        "total_trials": len(positions) * args.trials_per_position,
        "success_target": (len(positions) * args.trials_per_position + 1) // 2,
        "selection": "adjacent-cell midpoints, then deterministic farthest-point sampling",
        "excluded_manifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "positions": positions,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[EVAL] wrote {args.output}: {len(positions)} held-out positions, "
        f"{manifest['total_trials']} total trials, success target >= {manifest['success_target']}"
    )
    for position in positions:
        print(
            f"  {position['eval_id']} x={position['x']:+.4f} y={position['y']:+.4f} "
            f"z={position['table_z']:+.4f} from cells={position['source_cells']}"
        )


if __name__ == "__main__":
    main()
