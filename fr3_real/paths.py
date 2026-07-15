"""Canonical paths for the FR3 real-robot workspace."""

from pathlib import Path

FR3_REAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FR3_REAL_ROOT.parent

CONFIG_DIR = FR3_REAL_ROOT / "config"
DATA_DIR = FR3_REAL_ROOT / "data"
CALIBRATION_DIR = DATA_DIR / "calibration"
CELLS_DIR = DATA_DIR / "cells"
TRACKERS_DIR = DATA_DIR / "trackers"
STATE_DIR = DATA_DIR / "state"
RECORDINGS_DIR = FR3_REAL_ROOT / "recordings"
RAW_RECORDINGS_DIR = REPO_ROOT / "fr3_raw_recordings"

DEFAULT_POINTS_PATH = CALIBRATION_DIR / "probed_points.json"
DEFAULT_PROBE_OUTPUT_DIR = CALIBRATION_DIR
DEFAULT_CAMERA_TEST_RECORD_DIR = RECORDINGS_DIR / "camera_test"

DEFAULT_GRID_TRACKER_PATH = TRACKERS_DIR / "fr3_100_cell_tracker.xlsx"
DEFAULT_WORKING_TRACKER_PATH = TRACKERS_DIR / "fr3_working_cell_tracker.xlsx"
DEFAULT_ALL_WORKING_TRACKER_PATH = TRACKERS_DIR / "fr3_all_working_cell_tracker.xlsx"

DEFAULT_WORKING_CELLS_PATH = CELLS_DIR / "working_cells.json"
DEFAULT_ALL_WORKING_CELLS_PATH = CELLS_DIR / "all_working_cells.json"

DEFAULT_RESUME_PATH = STATE_DIR / "next_cell_to_record.txt"
