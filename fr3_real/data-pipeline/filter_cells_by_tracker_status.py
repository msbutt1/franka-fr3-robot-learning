#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/filter_cells_by_tracker_status.py --help
"""Filter a cells JSON by statuses in a tracker spreadsheet."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filter_grid_tracker import read_tracker
from repo_paths import CONFIGS_DIR


parser = argparse.ArgumentParser()
parser.add_argument("--cells_json", type=Path, default=CONFIGS_DIR / "all_working_cells.json")
parser.add_argument("--tracker", type=Path, default=CONFIGS_DIR / "fr3_all_working_cell_tracker.xlsx")
parser.add_argument("--keep_status", type=str, action="append", default=["PASS"])
parser.add_argument("--output", type=Path, default=None,
                    help="Defaults to overwriting --cells_json.")
args = parser.parse_args()

data = json.loads(args.cells_json.read_text())
rows = read_tracker(args.tracker)
keep_statuses = {status.strip().upper() for status in args.keep_status}
keep_ids = {
    int(row["printed_cell"])
    for row in rows
    if str(row.get("status", "")).strip().upper() in keep_statuses
}

old_cells = data["cells"]
new_cells = [cell for cell in old_cells if int(cell.get("printed_cell", -1)) in keep_ids]
removed = [cell for cell in old_cells if int(cell.get("printed_cell", -1)) not in keep_ids]

data["cells"] = new_cells
output = args.output or args.cells_json
output.write_text(json.dumps(data, indent=2) + "\n")

print(f"tracker: {args.tracker}")
print(f"input:   {args.cells_json}")
print(f"output:  {output}")
print(f"kept statuses: {sorted(keep_statuses)}")
print(f"old count: {len(old_cells)}")
print(f"new count: {len(new_cells)}")
print("removed printed_cell values:", [cell.get("printed_cell") for cell in removed])
