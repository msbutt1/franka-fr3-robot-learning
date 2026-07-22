#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/compute_fr3_custom_norm_stats.py --help
"""Compute unfiltered custom normalization statistics for a merged FR3 dataset."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--config", default="pi05_fr3_real_droid_full")
    parser.add_argument("--dataset-repo", default="local/fr3_real_pick_place_droid_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    openpi_root = args.openpi_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"{output} already exists; use a new output path for a reproducible ablation")
    if not (openpi_root / "scripts" / "compute_norm_stats.py").exists():
        raise SystemExit(f"Not an OpenPI checkout: {openpi_root}")

    sys.path.insert(0, str(openpi_root))
    os.chdir(openpi_root)

    from scripts import compute_norm_stats
    from openpi.shared import normalize
    from openpi.training import config as training_config

    base = training_config.get_config(args.config)
    data_factory = dataclasses.replace(
        base.data,
        repo_id=args.dataset_repo,
        assets=training_config.AssetsConfig(),
    )
    data_config = data_factory.create(base.assets_dirs, base.model)
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config,
        base.model.action_horizon,
        args.batch_size,
        base.model,
        args.num_workers,
        args.max_frames,
    )
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch_index, batch in enumerate(loader, start=1):
        for key in running:
            values = np.asarray(batch[key])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values found in {key} at batch {batch_index}")
            running[key].update(values)
        if batch_index % 25 == 0 or batch_index == num_batches:
            print(f"[norm] batch={batch_index}/{num_batches}", flush=True)

    stats = {key: value.get_statistics() for key, value in running.items()}
    for key, value in stats.items():
        span = np.asarray(value.q99) - np.asarray(value.q01)
        print(f"[norm] {key}.q01={np.array2string(np.asarray(value.q01), precision=6)}")
        print(f"[norm] {key}.q99={np.array2string(np.asarray(value.q99), precision=6)}")
        print(f"[norm] {key}.span={np.array2string(span, precision=6)}")
        if np.any(span <= 1e-5):
            print(f"[norm:warning] {key} has q01-q99 span <= 1e-5")
    normalize.save(output, stats)
    print(f"[norm] wrote {output / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
