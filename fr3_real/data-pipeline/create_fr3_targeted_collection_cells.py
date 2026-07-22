#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/create_fr3_targeted_collection_cells.py --help
"""Create a deterministic, held-out-safe FR3 targeted-data collection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grid_utils import basket_polygon_from_points, inside_basket_exclusion
from repo_paths import CONFIGS_DIR


def load_positions(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("positions", [])


def load_cells(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("cells", [])


def nearest_table_z(xy: np.ndarray, source_cells: list[dict]) -> float:
    source_xy = np.asarray([[cell["x"], cell["y"]] for cell in source_cells], dtype=float)
    index = int(np.argmin(np.linalg.norm(source_xy - xy, axis=1)))
    return float(source_cells[index]["table_z"])


def is_valid(
    xy: np.ndarray,
    *,
    source_cells: list[dict],
    protected_xy: np.ndarray,
    protected_radius: float,
    avoided_xy: np.ndarray,
    avoided_radius: float,
    pad_center: np.ndarray,
    basket_polygon: np.ndarray | None,
    basket_margin: float,
) -> bool:
    source_xy = np.asarray([[cell["x"], cell["y"]] for cell in source_cells], dtype=float)
    lower = source_xy.min(axis=0) + 0.015
    upper = source_xy.max(axis=0) - 0.015
    if np.any(xy < lower) or np.any(xy > upper):
        return False
    if len(protected_xy) and np.min(np.linalg.norm(protected_xy - xy, axis=1)) < protected_radius:
        return False
    if len(avoided_xy) and np.min(np.linalg.norm(avoided_xy - xy, axis=1)) < avoided_radius:
        return False
    xyz = np.asarray([xy[0], xy[1], nearest_table_z(xy, source_cells)])
    return not inside_basket_exclusion(
        xyz,
        pad_center,
        basket_margin,
        basket_w=0.18,
        basket_h=0.18,
        basket_polygon_xy=basket_polygon,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-json", type=Path, default=CONFIGS_DIR / "all_working_cells.json")
    parser.add_argument("--points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument("--validation-manifest", type=Path, default=CONFIGS_DIR / "fr3_spatial_validation_v3.json")
    parser.add_argument("--final-manifest", type=Path, default=CONFIGS_DIR / "fr3_interstitial_eval_v1.json")
    parser.add_argument("--avoid-cells-json", type=Path, action="append", default=[],
                        help="Existing collection manifests whose XY positions must not be reused.")
    parser.add_argument("--focus-eval-id", default="I01")
    parser.add_argument("--focus-count", type=int, default=32)
    parser.add_argument("--total-count", type=int, default=80)
    parser.add_argument("--protected-radius", type=float, default=0.03)
    parser.add_argument("--focus-min-radius", type=float, default=0.04)
    parser.add_argument("--focus-max-radius", type=float, default=0.07)
    parser.add_argument("--jitter-min", type=float, default=0.01)
    parser.add_argument("--jitter-max", type=float, default=0.03)
    parser.add_argument("--min-cell-separation", type=float, default=0.012)
    parser.add_argument("--avoid-radius", type=float, default=0.02,
                        help="Minimum XY separation from positions in --avoid-cells-json, meters.")
    parser.add_argument("--printed-cell-start", type=int, default=1001)
    parser.add_argument("--notes", default="targeted_v2")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, default=CONFIGS_DIR / "fr3_targeted_v2_cells.json")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    if not 0 < args.protected_radius < args.focus_min_radius < args.focus_max_radius:
        raise SystemExit("Require 0 < protected radius < focus min radius < focus max radius")
    if not 0 < args.jitter_min <= args.jitter_max:
        raise SystemExit("Require 0 < jitter min <= jitter max")
    if not 0 <= args.focus_count <= args.total_count:
        raise SystemExit("Require 0 <= focus count <= total count")
    if args.avoid_radius < 0:
        raise SystemExit("--avoid-radius must be non-negative")

    source_cells = [
        cell for cell in json.loads(args.cells_json.read_text())["cells"] if cell.get("status") == "PASS"
    ]
    validation = load_positions(args.validation_manifest)
    final = load_positions(args.final_manifest)
    protected = validation + final
    protected_xy = np.asarray([[item["x"], item["y"]] for item in protected], dtype=float)
    avoided = [cell for path in args.avoid_cells_json for cell in load_cells(path)]
    avoided_xy = np.asarray([[item["x"], item["y"]] for item in avoided], dtype=float)
    focus = next((item for item in validation if item.get("eval_id") == args.focus_eval_id), None)
    if focus is None:
        raise SystemExit(f"No validation position {args.focus_eval_id!r} in {args.validation_manifest}")
    focus_xy = np.asarray([focus["x"], focus["y"]], dtype=float)

    points = json.loads(args.points.read_text())
    pad_center = np.asarray(points["pad_center"], dtype=float)
    basket_polygon = basket_polygon_from_points(points)
    rng = np.random.default_rng(args.seed)
    candidates: list[dict] = []

    def add_candidate(xy: np.ndarray, region: str, reference_xy: np.ndarray | None = None) -> bool:
        if not is_valid(
            xy,
            source_cells=source_cells,
            protected_xy=protected_xy,
            protected_radius=args.protected_radius,
            avoided_xy=avoided_xy,
            avoided_radius=args.avoid_radius,
            pad_center=pad_center,
            basket_polygon=basket_polygon,
            basket_margin=0.04,
        ):
            return False
        for existing in candidates:
            if np.linalg.norm(xy - np.asarray([existing["x"], existing["y"]])) < args.min_cell_separation:
                return False
        cell_id = args.printed_cell_start + len(candidates)
        record = {
            "printed_cell": cell_id,
            "selected_index": cell_id - 1,
            "generated_index": cell_id - 1,
            "status": "PLANNED",
            "notes": args.notes,
            "collection_region": region,
            "x": float(xy[0]),
            "y": float(xy[1]),
            "table_z": nearest_table_z(xy, source_cells),
        }
        if reference_xy is not None:
            record["reference_xy"] = [float(reference_xy[0]), float(reference_xy[1])]
        candidates.append(record)
        return True

    if args.focus_count:
        attempts = 0
        while len(candidates) < args.focus_count and attempts < 20_000:
            attempts += 1
            angle = rng.uniform(0.0, 2.0 * np.pi)
            radius = rng.uniform(args.focus_min_radius, args.focus_max_radius)
            add_candidate(focus_xy + radius * np.asarray([np.cos(angle), np.sin(angle)]), "I01_annulus", focus_xy)
        if len(candidates) != args.focus_count:
            raise SystemExit(f"Could only generate {len(candidates)}/{args.focus_count} focus candidates")

    safe_anchors = [
        cell
        for cell in source_cells
        if is_valid(
            np.asarray([cell["x"], cell["y"]], dtype=float),
            source_cells=source_cells,
                protected_xy=protected_xy,
                protected_radius=args.protected_radius,
                avoided_xy=avoided_xy,
                avoided_radius=args.avoid_radius,
            pad_center=pad_center,
            basket_polygon=basket_polygon,
            basket_margin=0.04,
        )
    ]
    if not safe_anchors:
        raise SystemExit("No safe source cells remain after protected-set exclusion")
    attempts = 0
    while len(candidates) < args.total_count and attempts < 40_000:
        attempts += 1
        anchor = safe_anchors[int(rng.integers(len(safe_anchors)))]
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(args.jitter_min, args.jitter_max)
        anchor_xy = np.asarray([anchor["x"], anchor["y"]], dtype=float)
        add_candidate(anchor_xy + radius * np.asarray([np.cos(angle), np.sin(angle)]), "workspace_jitter", anchor_xy)
    if len(candidates) != args.total_count:
        raise SystemExit(f"Could only generate {len(candidates)}/{args.total_count} candidates")

    output = {
        "description": "Targeted FR3 collection positions; not evaluation positions.",
        "seed": args.seed,
        "protected_radius_m": args.protected_radius,
        "focus_eval_id": args.focus_eval_id,
        "focus_count": args.focus_count,
        "total_count": args.total_count,
        "avoid_radius_m": args.avoid_radius,
        "avoided_manifests": [str(path) for path in args.avoid_cells_json],
        "cells": candidates,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[targeted] wrote {args.output}: {len(candidates)} planned positions")
    for region in ("I01_annulus", "workspace_jitter"):
        print(f"[targeted] {region}={sum(cell['collection_region'] == region for cell in candidates)}")


if __name__ == "__main__":
    main()
