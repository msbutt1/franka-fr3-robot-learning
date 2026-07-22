#!/usr/bin/env python3
# Usage: from fr3_real/, run: python training/prepare_pi05_fr3_real_droid_full.py --help
"""Add a conservative full pi0.5-DROID fine-tuning config to OpenPI.

Run once from the OpenPI repository root on Nibi. The generated config keeps
the pi05-DROID checkpoint, action representation, and normalization assets,
but points at the physical FR3 v3 LeRobot dataset and uses a shorter low-LR
run suitable for the collected real-robot demonstrations.
"""

from __future__ import annotations

import re
from pathlib import Path


CONFIG_PATH = Path("src/openpi/training/config.py")
BASE_NAME = "pi05_droid_finetune"
NEW_NAME = "pi05_fr3_real_droid_full"
REPO_ID = "local/fr3_real_pick_place_droid_v3"


def extract_train_config(text: str, config_name: str) -> tuple[int, int, str]:
    marker = f'name="{config_name}"'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find config {config_name!r}")

    block_start = text.rfind("TrainConfig(", 0, start)
    if block_start < 0:
        raise RuntimeError(f"Could not find the TrainConfig for {config_name!r}")

    depth = 0
    for index, char in enumerate(text[block_start:]):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                block_end = block_start + index + 1
                return block_start, block_end, text[block_start:block_end]
    raise RuntimeError(f"Could not bracket config {config_name!r}")


def replace_required(block: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, block, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one config match for {pattern!r}, found {count}")
    return updated


def set_optional_numeric_field(block: str, name: str, value: str, anchor: str) -> str:
    pattern = rf"{name}=[\d_]+"
    updated, count = re.subn(pattern, f"{name}={value}", block, count=1)
    if count == 1:
        return updated
    if count > 1:
        raise RuntimeError(f"Expected at most one {name!r} field, found {count}")
    if anchor not in block:
        raise RuntimeError(f"Could not insert inherited {name!r} override")
    return block.replace(anchor, f"        {name}={value},\n" + anchor, 1)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit("Run this from the OpenPI repository root.")

    text = CONFIG_PATH.read_text()
    if f'name="{NEW_NAME}"' in text:
        print(f"[skip] config {NEW_NAME!r} already exists")
        return

    _, insert_at, block = extract_train_config(text, BASE_NAME)
    block = block.replace(f'name="{BASE_NAME}"', f'name="{NEW_NAME}"', 1)
    block = replace_required(
        block,
        r'repo_id="[^"]+"',
        f'repo_id="{REPO_ID}"',
    )
    block = replace_required(block, r"num_train_steps=[\d_]+", "num_train_steps=8_000")
    block = replace_required(block, r"batch_size=[\d_]+", "batch_size=32")
    anchor = "        num_train_steps=8_000,"
    block = set_optional_numeric_field(block, "save_interval", "1_000", anchor)
    block = set_optional_numeric_field(block, "keep_period", "2_000", anchor)

    schedule = '''        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1.0e-5,
            decay_steps=8_000,
            decay_lr=1.0e-6,
        ),
'''
    if anchor not in block:
        raise RuntimeError("Could not find schedule insertion point")
    block = block.replace(anchor, schedule + anchor, 1)

    CONFIG_PATH.write_text(text[:insert_at] + ",\n    " + block + text[insert_at:])
    print(f"[add ] config {NEW_NAME!r}")
    print("      full parameter fine-tuning, batch=32, steps=8000, peak_lr=1e-5")


if __name__ == "__main__":
    main()
