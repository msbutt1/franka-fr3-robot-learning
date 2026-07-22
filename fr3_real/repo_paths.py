"""Canonical filesystem locations for the FR3 project.
# Usage: import this shared module from an FR3 script; it is not a standalone command.

Scripts may be launched from the repository root, a category directory, or an
arbitrary shell working directory. Keep project-owned defaults here so their
location never depends on the process working directory.
"""

from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FR3_ROOT / "configs"
RECORDINGS_DIR = FR3_ROOT / "recordings"
EVAL_LOGS_DIR = FR3_ROOT / "eval_logs"
SHADOW_LOGS_DIR = FR3_ROOT / "shadow_logs"


def config_path(name: str) -> Path:
    """Return a tracked configuration file under the project config directory."""
    return CONFIGS_DIR / name
