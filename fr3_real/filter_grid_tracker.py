#!/usr/bin/env python3
"""Filter a grid tracker workbook and export runnable working cells.

Typical use:
  1. Mark tracker rows as PASS, FAIL, or SKIP in the status column.
  2. Run this script to remove FAIL/SKIP rows.
  3. Pass the produced JSON to pick_and_place.py with --cells_json.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from create_grid_tracker import worksheet_xml, write_xlsx


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_RE = re.compile(r"([A-Z]+)([0-9]+)")


def col_to_index(name: str) -> int:
    index = 0
    for ch in name:
        index = index * 26 + (ord(ch) - 64)
    return index - 1


def cell_col(ref: str) -> int:
    match = CELL_RE.fullmatch(ref)
    if not match:
        raise ValueError(f"bad cell ref: {ref}")
    return col_to_index(match.group(1))


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("x:si", NS):
        values.append("".join(text.text or "" for text in item.findall(".//x:t", NS)))
    return values


def cell_value(cell: ET.Element, strings: list[str]):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//x:t", NS))
    value = cell.find("x:v", NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return strings[int(value.text)]
    text = value.text
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def read_tracker(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values = {}
        for cell in row.findall("x:c", NS):
            values[cell_col(cell.attrib["r"])] = cell_value(cell, strings)
        rows.append(values)
    if not rows:
        return []

    headers = {col: str(value) for col, value in rows[0].items()}
    cells = []
    for row in rows[1:]:
        item = {header: row.get(col, "") for col, header in headers.items()}
        if not item.get("x_m") or not item.get("y_m") or not item.get("table_z_m"):
            continue
        cells.append(
            {
                "printed_cell": int(item["printed_cell"]),
                "selected_index": int(item["selected_index"]),
                "generated_index": int(item["generated_index"]),
                "grid_i": int(item["grid_i"]),
                "grid_j": int(item["grid_j"]),
                "status": str(item.get("status") or "UNTESTED").strip().upper(),
                "notes": str(item.get("notes") or ""),
                "x": float(item["x_m"]),
                "y": float(item["y_m"]),
                "table_z": float(item["table_z_m"]),
            }
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="fr3_100_cell_tracker.xlsx")
    parser.add_argument("--out_xlsx", default="fr3_working_cell_tracker.xlsx")
    parser.add_argument("--out_json", default="working_cells.json")
    parser.add_argument("--drop_status", action="append", default=["FAIL", "SKIP"],
                        help="Status to remove. Defaults to FAIL and SKIP. Can be repeated.")
    parser.add_argument("--keep_status", action="append", default=None,
                        help="If provided, keep only these statuses instead of using --drop_status.")
    parser.add_argument("--hover_clearance", type=float, default=0.12)
    parser.add_argument("--cube_height", type=float, default=0.04)
    parser.add_argument("--grasp_lowering", type=float, default=0.0)
    args = parser.parse_args()

    cells = read_tracker(Path(args.tracker))
    if args.keep_status:
        keep = {status.upper() for status in args.keep_status}
        filtered = [cell for cell in cells if cell["status"] in keep]
    else:
        drop = {status.upper() for status in args.drop_status}
        filtered = [cell for cell in cells if cell["status"] not in drop]

    sheet = worksheet_xml(filtered, args.hover_clearance, args.cube_height, args.grasp_lowering)
    write_xlsx(Path(args.out_xlsx), sheet)
    Path(args.out_json).write_text(json.dumps({"cells": filtered}, indent=2) + "\n")
    print(f"read {len(cells)} cells")
    print(f"wrote {len(filtered)} working cells to {args.out_xlsx}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
