#!/usr/bin/env python3
# Usage: from fr3_real/, run: python training/train_fr3_phase_filtered.py --help
"""Launch an OpenPI full fine-tune with phase-aware LeRobot start indices."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import SupportsIndex

import numpy as np


class IndexMappedDataset:
    """Map a weighted list of starts onto an unchanged temporal dataset."""

    def __init__(self, dataset, indices: np.ndarray):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, index: SupportsIndex):
        return self._dataset[int(self._indices[index.__index__()])]

    def __len__(self) -> int:
        return len(self._indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--config", default="pi05_fr3_real_droid_full")
    parser.add_argument("--dataset-repo", default="local/fr3_real_pick_place_droid_v3")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        help="Persistent directory holding OpenPI checkpoints. Defaults to the config value.",
    )
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument("--decay-lr", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest alignment and data loading, then exit before model initialization.",
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        help="Directory containing norm_stats.json. Omit to retain pretrained DROID statistics.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    openpi_root = args.openpi_root.expanduser().resolve()
    manifest_path = args.sample_manifest.expanduser().resolve()
    if not (openpi_root / "scripts" / "train.py").exists():
        raise SystemExit(f"Not an OpenPI checkout: {openpi_root}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing sampling manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != args.dataset_repo:
        raise SystemExit(
            f"Manifest repo_id={manifest.get('repo_id')!r} does not match "
            f"--dataset-repo={args.dataset_repo!r}"
        )
    indices = np.asarray(manifest["sample_indices"], dtype=np.int64)
    if len(indices) == 0 or np.min(indices) < 0:
        raise SystemExit("Sampling manifest has no valid indices")

    sys.path.insert(0, str(openpi_root))
    os.chdir(openpi_root)

    from scripts import train
    from openpi.training import config as training_config
    from openpi.training import data_loader
    from openpi.training import optimizer

    original_create_torch_dataset = data_loader.create_torch_dataset

    def create_indexed_torch_dataset(data_config, action_horizon, model_config):
        dataset = original_create_torch_dataset(data_config, action_horizon, model_config)
        expected = int(manifest["dataset_total_frames"])
        if len(dataset) != expected:
            raise RuntimeError(
                f"Manifest/dataset mismatch: manifest={expected} frames, dataset={len(dataset)}. "
                "Regenerate the manifest from the exact raw data used for this conversion."
            )
        if int(np.max(indices)) >= len(dataset):
            raise RuntimeError("Sampling manifest contains an out-of-range dataset index")
        print(
            f"[sampling] base_frames={len(dataset)} weighted_starts={len(indices)} "
            f"unique_starts={manifest['kept_unique_count']} dropped_idle={manifest['dropped_idle_count']}"
        )
        return IndexMappedDataset(dataset, indices)

    data_loader.create_torch_dataset = create_indexed_torch_dataset

    base = training_config.get_config(args.config)
    if args.norm_stats_dir is None:
        model_data = dataclasses.replace(base.data, repo_id=args.dataset_repo)
        norm_label = "pretrained_droid"
    else:
        norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
        if not (norm_stats_dir / "norm_stats.json").is_file():
            raise SystemExit(f"Missing {norm_stats_dir / 'norm_stats.json'}")
        model_data = dataclasses.replace(
            base.data,
            repo_id=args.dataset_repo,
            assets=training_config.AssetsConfig(
                assets_dir=str(norm_stats_dir.parent),
                asset_id=norm_stats_dir.name,
            ),
        )
        norm_label = str(norm_stats_dir)
    config = dataclasses.replace(
        base,
        exp_name=args.exp_name,
        checkpoint_base_dir=(
            str(args.checkpoint_base_dir.expanduser().resolve())
            if args.checkpoint_base_dir is not None
            else base.checkpoint_base_dir
        ),
        data=model_data,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.num_train_steps,
            decay_lr=args.decay_lr,
        ),
        ema_decay=args.ema_decay,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        num_workers=args.num_workers,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        f"[train] config={config.name} dataset={args.dataset_repo} exp={args.exp_name} "
        f"steps={args.num_train_steps} ema={args.ema_decay} norm={norm_label}"
    )
    if args.validate_only:
        resolved_data = config.data.create(config.assets_dirs, config.model)
        dataset = data_loader.create_torch_dataset(resolved_data, config.model.action_horizon, config.model)
        print(f"[validate] aligned weighted dataset has {len(dataset)} starts; no training was started")
        return
    train.main(config)


if __name__ == "__main__":
    main()
