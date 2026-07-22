#!/usr/bin/env python3
# Usage: from fr3_real/, run: python setup/verify_fr3_setup.py --help
"""Compatibility launcher for the repository FR3 setup verifier."""

from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "setup" / "verify_fr3_setup.py"),
    run_name="__main__",
)
