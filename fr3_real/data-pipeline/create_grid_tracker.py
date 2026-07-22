#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/create_grid_tracker.py --help
"""Create an Excel tracker for dense FR3 grid validation.

The grid logic intentionally matches the real-robot scripts:
  - bilinear interpolation from the four probed table corners
  - basket footprint exclusion
  - optional even down-selection with --max_cells

The output .xlsx uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grid_utils import basket_polygon_from_points, inside_basket_exclusion
from repo_paths import CONFIGS_DIR


def col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def inline_cell(row: int, col: int, value, style: int | None = None) -> str:
    ref = f"{col_name(col)}{row}"
    style_attr = f' s="{style}"' if style is not None else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def row_xml(row: int, values, style: int | None = None) -> str:
    cells = "".join(inline_cell(row, col, value, style) for col, value in enumerate(values))
    return f'<row r="{row}">{cells}</row>'


def build_cells(points, nx: int, ny: int, max_cells: int | None, basket_w: float, basket_h: float, basket_margin: float):
    bl = np.array(points["bottom_left"], dtype=float)
    br = np.array(points["bottom_right"], dtype=float)
    tl = np.array(points["top_left"], dtype=float)
    tr = np.array(points["top_right"], dtype=float)
    pad = np.array(points["pad_center"], dtype=float)
    basket_polygon_xy = basket_polygon_from_points(points)

    cells = []
    generated_index = 0
    for i, j in itertools.product(range(nx), range(ny)):
        u = (i + 0.5) / nx
        v = (j + 0.5) / ny
        near = (1 - v) * br + v * bl
        far = (1 - v) * tr + v * tl
        xyz = (1 - u) * near + u * far
        excluded = inside_basket_exclusion(xyz, pad, basket_margin, basket_w, basket_h, basket_polygon_xy)
        if not excluded:
            cells.append(
                {
                    "generated_index": generated_index,
                    "grid_i": i,
                    "grid_j": j,
                    "u": u,
                    "v": v,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "table_z": float(xyz[2]),
                }
            )
        generated_index += 1

    if max_cells is not None and len(cells) > max_cells:
        keep = np.linspace(0, len(cells) - 1, max_cells, dtype=int)
        cells = [cells[k] for k in keep]

    for selected_index, cell in enumerate(cells):
        cell["selected_index"] = selected_index
        cell["printed_cell"] = selected_index + 1
    return cells


def workbook_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Cells" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="3">
    <dxf><fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor rgb="FFC6EFCE"/></patternFill></fill><font><color rgb="FF006100"/></font></dxf>
    <dxf><fill><patternFill patternType="solid"><fgColor rgb="FFFFC7CE"/><bgColor rgb="FFFFC7CE"/></patternFill></fill><font><color rgb="FF9C0006"/></font></dxf>
    <dxf><fill><patternFill patternType="solid"><fgColor rgb="FFFFEB9C"/><bgColor rgb="FFFFEB9C"/></patternFill></fill><font><color rgb="FF9C6500"/></font></dxf>
  </dxfs>
</styleSheet>
"""


def worksheet_xml(cells, hover_clearance: float, cube_height: float, grasp_lowering: float) -> str:
    headers = [
        "printed_cell",
        "selected_index",
        "generated_index",
        "grid_i",
        "grid_j",
        "status",
        "notes",
        "x_m",
        "y_m",
        "table_z_m",
        "hover_tcp_z_m",
        "grasp_tcp_z_m",
    ]
    rows = [row_xml(1, headers, style=1)]
    for row_idx, cell in enumerate(cells, start=2):
        table_z = cell["table_z"]
        values = [
            cell["printed_cell"],
            cell["selected_index"],
            cell["generated_index"],
            cell["grid_i"],
            cell["grid_j"],
            cell.get("status", "UNTESTED"),
            cell.get("notes", ""),
            round(cell["x"], 4),
            round(cell["y"], 4),
            round(table_z, 4),
            round(table_z + hover_clearance, 4),
            round(table_z + cube_height / 2 - grasp_lowering, 4),
        ]
        rows.append(row_xml(row_idx, values))

    last = len(cells) + 1
    cols = [
        '<col min="1" max="5" width="14" customWidth="1"/>',
        '<col min="6" max="6" width="14" customWidth="1"/>',
        '<col min="7" max="7" width="32" customWidth="1"/>',
        '<col min="8" max="12" width="14" customWidth="1"/>',
    ]
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>{''.join(cols)}</cols>
  <sheetData>{''.join(rows)}</sheetData>
  <autoFilter ref="A1:L{last}"/>
  <dataValidations count="1">
    <dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="F2:F{last}">
      <formula1>"UNTESTED,PASS,FAIL,SKIP"</formula1>
    </dataValidation>
  </dataValidations>
  <conditionalFormatting sqref="F2:F{last}">
    <cfRule type="cellIs" priority="1" operator="equal" dxfId="0"><formula>"PASS"</formula></cfRule>
    <cfRule type="cellIs" priority="2" operator="equal" dxfId="1"><formula>"FAIL"</formula></cfRule>
    <cfRule type="cellIs" priority="3" operator="equal" dxfId="2"><formula>"SKIP"</formula></cfRule>
  </conditionalFormatting>
</worksheet>
"""


def write_xlsx(path: Path, sheet_xml: str) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", default=str(CONFIGS_DIR / "probed_points.json"))
    parser.add_argument("--out", default="fr3_100_cell_tracker.xlsx")
    parser.add_argument("--nx", type=int, default=11)
    parser.add_argument("--ny", type=int, default=11)
    parser.add_argument("--max_cells", type=int, default=100)
    parser.add_argument("--basket_w", type=float, default=0.154)
    parser.add_argument("--basket_h", type=float, default=0.134)
    parser.add_argument("--basket_margin", type=float, default=0.04)
    parser.add_argument("--hover_clearance", type=float, default=0.12)
    parser.add_argument("--cube_height", type=float, default=0.04)
    parser.add_argument("--grasp_lowering", type=float, default=0.0)
    args = parser.parse_args()

    points = json.loads(Path(args.points).read_text())
    cells = build_cells(points, args.nx, args.ny, args.max_cells, args.basket_w, args.basket_h, args.basket_margin)
    sheet = worksheet_xml(cells, args.hover_clearance, args.cube_height, args.grasp_lowering)
    out = Path(args.out)
    write_xlsx(out, sheet)
    print(f"wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
