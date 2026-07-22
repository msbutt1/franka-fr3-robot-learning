#!/usr/bin/env python3
# Usage: from fr3_real/, run: python training/prepare_pi05_fr3_real_droid_lora_r16.py --help
"""Add a physical-FR3 pi0.5-DROID rank-16 LoRA config to an OpenPI checkout.

Run once from the OpenPI root on Nibi. It leaves all Polaris simulator configs
untouched and is idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


CONFIG_PATH = Path("src/openpi/training/config.py")
GEMMA_PATH = Path("src/openpi/models/gemma.py")
BASE_NAME = "pi05_droid_finetune"
NEW_NAME = "pi05_fr3_real_droid_lora_r16"
REPO_ID = "local/fr3_real_pick_place_droid"


def extract_train_config(text: str, config_name: str) -> tuple[int, int, str]:
    marker = f'name="{config_name}"'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find config {config_name!r}")
    block_start = text.rfind("TrainConfig(", 0, start)
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


def ensure_rank16_variants() -> None:
    text = GEMMA_PATH.read_text()
    if "gemma_2b_lora_r16" in text and "gemma_300m_lora_r16" in text:
        print("[skip] rank-16 Gemma variants already exist")
        return
    anchor = '    raise ValueError(f"Unknown variant: {variant}")'
    if anchor not in text:
        raise RuntimeError("Could not find Gemma variant insertion point")
    block = '''    if variant == "gemma_2b_lora_r16":
        return Config(
            width=2048, depth=18, mlp_dim=16_384, num_heads=8,
            num_kv_heads=1, head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=32.0)},
        )
    if variant == "gemma_300m_lora_r16":
        return Config(
            width=1024, depth=18, mlp_dim=4096, num_heads=8,
            num_kv_heads=1, head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=32.0)},
        )
'''
    GEMMA_PATH.write_text(text.replace(anchor, block + anchor, 1))
    print("[add ] rank-16 Gemma variants")


def set_optional_numeric_field(block: str, name: str, value: str, anchor_pattern: str) -> str:
    pattern = rf"{name}=[\d_]+"
    updated, count = re.subn(pattern, f"{name}={value}", block, count=1)
    if count == 1:
        return updated
    if count > 1:
        raise RuntimeError(f"Expected at most one {name!r} field, found {count}")
    anchor_match = re.search(anchor_pattern, block)
    if anchor_match is None:
        raise RuntimeError(f"Could not insert inherited {name!r} override")
    anchor = anchor_match.group(0)
    return block.replace(anchor, f"        {name}={value},\n" + anchor, 1)


def ensure_checkpoint_fields(block: str) -> str:
    anchor_pattern = r"        num_train_steps=[\d_]+,"
    block = set_optional_numeric_field(block, "save_interval", "1_000", anchor_pattern)
    return set_optional_numeric_field(block, "keep_period", "2_000", anchor_pattern)


def main() -> None:
    if not CONFIG_PATH.exists() or not GEMMA_PATH.exists():
        raise SystemExit("Run this from the OpenPI repository root.")
    ensure_rank16_variants()
    text = CONFIG_PATH.read_text()
    if f'name="{NEW_NAME}"' in text:
        block_start, block_end, block = extract_train_config(text, NEW_NAME)
        updated = ensure_checkpoint_fields(block)
        if updated == block:
            print(f"[skip] config {NEW_NAME!r} already exists and is current")
        else:
            CONFIG_PATH.write_text(text[:block_start] + updated + text[block_end:])
            print(f"[fix ] added explicit checkpoint fields to {NEW_NAME!r}")
        return

    _, insert_at, block = extract_train_config(text, BASE_NAME)
    block = block.replace(f'name="{BASE_NAME}"', f'name="{NEW_NAME}"', 1)
    block = block.replace('repo_id="your_hf_username/my_droid_dataset"', f'repo_id="{REPO_ID}"', 1)
    block = block.replace(
        "action_horizon=16,",
        'action_horizon=16,\n            paligemma_variant="gemma_2b_lora_r16",\n            action_expert_variant="gemma_300m_lora_r16",',
        1,
    )
    block = re.sub(r"num_train_steps=[\d_]+", "num_train_steps=12_000", block, count=1)
    block = re.sub(r"batch_size=[\d_]+", "batch_size=32", block, count=1)
    block = ensure_checkpoint_fields(block)
    insertion = '''        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora_r16",
            action_expert_variant="gemma_300m_lora_r16",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=12_000, decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=0.01),
        ema_decay=0.999,
'''
    block = block.replace("        num_train_steps=12_000,", insertion + "        num_train_steps=12_000,", 1)
    CONFIG_PATH.write_text(text[:insert_at] + ",\n    " + block + text[insert_at:])
    print(f"[add ] config {NEW_NAME!r}")


if __name__ == "__main__":
    main()
