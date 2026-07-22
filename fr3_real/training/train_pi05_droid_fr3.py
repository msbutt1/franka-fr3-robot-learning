#!/usr/bin/env python3
# Usage: from fr3_real/, run: python training/train_pi05_droid_fr3.py --help
"""Launch OpenPI pi0.5-DROID fine-tuning on an FR3 LeRobot dataset.

Run this from the OpenPI VM environment. The script deliberately imports the
OpenPI checkout supplied by --openpi_root so it uses the exact LeRobot revision
pinned by OpenPI, rather than an unrelated `lerobot==0.1.0` installation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi_root", type=Path, required=True)
    parser.add_argument("--dataset_repo", required=True, help="Local/Hugging Face LeRobot repo id.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--checkpoint_base_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=20_000)
    parser.add_argument("--save_interval", type=int, default=1_000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fsdp_devices", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mode", choices=["full", "lora"], default="full")
    args = parser.parse_args()

    openpi_root = args.openpi_root.expanduser().resolve()
    if not (openpi_root / "scripts" / "train.py").exists():
        raise SystemExit(f"--openpi_root is not an OpenPI checkout: {openpi_root}")
    sys.path.insert(0, str(openpi_root))

    from scripts.train import main as train_main
    from openpi.models import pi0_config
    from openpi.training import weight_loaders
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotDROIDDataConfig
    from openpi.training.config import TrainConfig

    if args.mode == "full":
        model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16)
        freeze_filter = model.get_freeze_filter()
        ema_decay = 0.999
    else:
        # OpenPI's stock variants use rank 16 for the 2B PaliGemma adapter and
        # rank 32 for the 300M action-expert adapter. It does not expose an
        # all-modules rank-16 option without modifying OpenPI source.
        model = pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze_filter = model.get_freeze_filter()
        ema_decay = None
        print("[train] LoRA mode: 2B adapter rank=16, action-expert adapter rank=32.")

    config = TrainConfig(
        name="fr3_pi05_droid_lora" if args.mode == "lora" else "fr3_pi05_droid_full",
        exp_name=args.exp_name,
        model=model,
        data=LeRobotDROIDDataConfig(
            repo_id=args.dataset_repo,
            base_config=DataConfig(prompt_from_task=True),
            # Keep original DROID normalization statistics: required when
            # fine-tuning the pi05_droid expert on DROID-compatible controls.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        checkpoint_base_dir=str(args.checkpoint_base_dir.expanduser().resolve()),
        wandb_enabled=args.wandb,
    )
    train_main(config)


if __name__ == "__main__":
    main()
