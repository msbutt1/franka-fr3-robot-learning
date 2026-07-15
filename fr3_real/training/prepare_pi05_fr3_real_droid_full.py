#!/usr/bin/env python3
"""Patch an existing OpenPI checkout with the FR3 pi0.5-DROID full fine-tune config.

This script intentionally does not clone or vendor OpenPI. Run it on the server
against the OpenPI checkout you will train from.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import textwrap


CLASS_START = "# BEGIN FR3_PI05_DROID_FULL_DATA_CONFIG"
CLASS_END = "# END FR3_PI05_DROID_FULL_DATA_CONFIG"
CONFIG_START = "# BEGIN FR3_PI05_DROID_FULL_TRAIN_CONFIG"
CONFIG_END = "# END FR3_PI05_DROID_FULL_TRAIN_CONFIG"


CLASS_BLOCK = textwrap.dedent(
    f"""
    {CLASS_START}
    @dataclasses.dataclass(frozen=True)
    class FR3LeRobotDROIDDataConfig(DataConfigFactory):
        \"\"\"Two-camera FR3 LeRobot dataset using pi0.5-DROID joint velocity actions.\"\"\"

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {{
                            "observation/exterior_image_1_left": "exterior_image_1_left",
                            "observation/wrist_image_left": "wrist_image_left",
                            "observation/joint_position": "joint_position",
                            "observation/gripper_position": "gripper_position",
                            "actions": "actions",
                            "prompt": "prompt",
                        }}
                    )
                ]
            )
            # The FR3 converter writes 7D joint velocity actions + 1D gripper
            # position, matching the pi0.5-DROID pretraining action space.
            data_transforms = _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
                outputs=[droid_policy.DroidOutputs()],
            )
            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )


    {CLASS_END}
    """
).strip()


CONFIG_TEMPLATE = textwrap.dedent(
    """
        {config_start}
        TrainConfig(
            name={config_name},
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
            ),
            data=FR3LeRobotDROIDDataConfig(
                repo_id={repo_id},
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(
                    assets_dir={assets_dir},
                    asset_id={asset_id},
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader({checkpoint}),
            num_train_steps={num_train_steps},
            batch_size={batch_size},
            num_workers={num_workers},
            save_interval={save_interval},
            keep_period={keep_period},
        ),
        {config_end}
    """
)


def remove_marked_block(text: str, start: str, end: str) -> str:
    pattern = re.compile(rf"\n?{re.escape(start)}.*?{re.escape(end)}\n?", re.DOTALL)
    return pattern.sub("\n", text)


def insert_before(text: str, needle: str, block: str, description: str) -> str:
    index = text.find(needle)
    if index == -1:
        raise SystemExit(f"Could not find insertion point for {description}: {needle!r}")
    return text[:index] + block.rstrip() + "\n\n" + text[index:]


def config_path_from_openpi(openpi_dir: Path) -> Path:
    path = openpi_dir / "src" / "openpi" / "training" / "config.py"
    if not path.exists():
        raise SystemExit(f"OpenPI config.py not found: {path}")
    return path


def build_config_block(args: argparse.Namespace) -> str:
    return CONFIG_TEMPLATE.format(
        config_start=CONFIG_START,
        config_end=CONFIG_END,
        config_name=repr(args.config_name),
        repo_id=repr(args.repo_id),
        assets_dir=repr(args.assets_dir),
        asset_id=repr(args.asset_id),
        checkpoint=repr(args.checkpoint),
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
    ).rstrip()


def patch_config(config_path: Path, args: argparse.Namespace) -> str:
    original = config_path.read_text()
    text = remove_marked_block(original, CLASS_START, CLASS_END)
    text = remove_marked_block(text, CONFIG_START, CONFIG_END)

    if "class FR3LeRobotDROIDDataConfig" in text:
        raise SystemExit("FR3LeRobotDROIDDataConfig already exists outside the managed FR3 block.")
    name_pattern = re.compile(rf"name\s*=\s*['\"]{re.escape(args.config_name)}['\"]")
    if name_pattern.search(text):
        raise SystemExit(f"Config {args.config_name!r} already exists outside the managed FR3 block.")

    class_needle = "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:"
    text = insert_before(text, class_needle, CLASS_BLOCK, "FR3 data config class")

    config_needle = "    #\n    # ALOHA Sim configs."
    text = insert_before(text, config_needle, build_config_block(args), "FR3 train config")

    ast.parse(text, filename=str(config_path))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openpi-dir",
        type=Path,
        default=Path(os.environ.get("OPENPI_DIR", ".")).expanduser(),
        help="Existing OpenPI checkout to patch.",
    )
    parser.add_argument("--config-name", default=os.environ.get("FR3_OPENPI_CONFIG", "pi05_fr3_real_droid_full"))
    parser.add_argument("--repo-id", default=os.environ.get("FR3_LEROBOT_REPO_ID", "local/fr3_real_pick_place_droid"))
    parser.add_argument("--num-train-steps", type=int, default=int(os.environ.get("FR3_NUM_TRAIN_STEPS", "20000")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FR3_BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("FR3_NUM_WORKERS", "0")))
    parser.add_argument("--save-interval", type=int, default=int(os.environ.get("FR3_SAVE_INTERVAL", "1000")))
    parser.add_argument("--keep-period", type=int, default=int(os.environ.get("FR3_KEEP_PERIOD", "5000")))
    parser.add_argument(
        "--assets-dir",
        default=os.environ.get("FR3_ASSETS_DIR", "gs://openpi-assets/checkpoints/pi05_droid/assets"),
    )
    parser.add_argument("--asset-id", default=os.environ.get("FR3_ASSET_ID", "droid"))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FR3_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_droid/params"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size > 32:
        raise SystemExit(f"Refusing batch_size={args.batch_size}; requested maximum is 32.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")

    config_path = config_path_from_openpi(args.openpi_dir.resolve())
    patched = patch_config(config_path, args)
    if args.dry_run:
        print(f"[dry-run] {config_path} parses after patch")
        return

    config_path.write_text(patched)
    print(f"[ok] patched {config_path}")
    print(f"[ok] config={args.config_name} repo_id={args.repo_id} batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
