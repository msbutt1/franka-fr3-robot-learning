#!/usr/bin/env python3
# Usage: from fr3_real/, run: python training/prepare_pi05_fr3_cartesian_h50_checkpoint.py --help
"""Add the exact inference config for the legacy FR3 Cartesian-delta checkpoint.

Run from the OpenPI repository root. This is deliberately isolated from the
joint-velocity DROID configs used by the current physical policy.
"""

from __future__ import annotations

from pathlib import Path


GEMMA_PATH = Path("src/openpi/models/gemma.py")
CONFIG_PATH = Path("src/openpi/training/config.py")
CONFIG_NAME = "pi05_fr3_cartesian_delta_h50_mixed_lora"


def add_gemma_variants() -> None:
    text = GEMMA_PATH.read_text()
    if "gemma_2b_lora_r16_a16" in text and "gemma_300m_lora_r32_a32" in text:
        print("[skip] exact mixed-LoRA Gemma variants already exist")
        return

    anchor = '    raise ValueError(f"Unknown variant: {variant}")'
    if anchor not in text:
        raise RuntimeError(f"Could not find Gemma variant anchor in {GEMMA_PATH}")

    block = '''    if variant == "gemma_2b_lora_r16_a16":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora_r32_a32":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
'''
    GEMMA_PATH.write_text(text.replace(anchor, block + anchor, 1))
    print("[add ] exact mixed-LoRA Gemma variants")


def extract_train_config(text: str, name: str) -> tuple[int, int]:
    marker = f'name="{name}"'
    marker_at = text.find(marker)
    if marker_at < 0:
        raise RuntimeError(f"Could not find base config {name!r}")
    start = text.rfind("TrainConfig(", 0, marker_at)
    if start < 0:
        raise RuntimeError(f"Could not find TrainConfig start for {name!r}")
    depth = 0
    for offset, char in enumerate(text[start:]):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, start + offset + 1
    raise RuntimeError(f"Could not bracket TrainConfig {name!r}")


def add_train_config() -> None:
    text = CONFIG_PATH.read_text()
    if f'name="{CONFIG_NAME}"' in text:
        print(f"[skip] config {CONFIG_NAME!r} already exists")
        return

    _, insert_at = extract_train_config(text, "pi05_droid_finetune")
    block = f''',
    TrainConfig(
        # Inference-only reconstruction of pi05_fr3_checkpoint_29999.
        # It consumes Cartesian TCP pose and emits 60 Hz Cartesian deltas.
        name="{CONFIG_NAME}",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora_r16_a16",
            action_expert_variant="gemma_300m_lora_r32_a32",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/fr3_lerobot_v1",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="training_dataset"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        num_train_steps=30_000,
        batch_size=1,
    )'''
    CONFIG_PATH.write_text(text[:insert_at] + block + text[insert_at:])
    print(f"[add ] config {CONFIG_NAME!r}")


def main() -> None:
    if not GEMMA_PATH.exists() or not CONFIG_PATH.exists():
        raise SystemExit("Run this script from the OpenPI repository root.")
    add_gemma_variants()
    add_train_config()


if __name__ == "__main__":
    main()
