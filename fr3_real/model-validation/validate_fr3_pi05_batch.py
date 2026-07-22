#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/validate_fr3_pi05_batch.py --help
"""Validate converted FR3 data and one OpenPI pi0.5-DROID training batch."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


RAW_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "joint_position",
    "gripper_position",
    "actions",
)


def column_array(table, name: str, *, vector: bool = False) -> np.ndarray:
    if name not in table.column_names:
        raise RuntimeError(f"Converted Parquet data is missing column {name!r}")
    values = table[name].to_pylist()
    if vector:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values)


def load_scalar_data(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple]]]:
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No Parquet files found under {dataset_dir}")

    states = []
    actions = []
    episodes: dict[int, list[tuple]] = defaultdict(list)
    for path in parquet_files:
        schema_names = set(pq.read_schema(path).names)
        missing = set(RAW_COLUMNS) - schema_names
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        table = pq.read_table(path, columns=list(RAW_COLUMNS))
        episode = column_array(table, "episode_index").astype(int)
        frame = column_array(table, "frame_index").astype(int)
        timestamp = column_array(table, "timestamp").astype(float)
        joint = column_array(table, "joint_position", vector=True)
        gripper = column_array(table, "gripper_position", vector=True)
        action = column_array(table, "actions", vector=True)

        # LeRobot 0.1 serializes singleton vector features as scalar Parquet
        # columns, so PyArrow returns (N,) even though the declared feature is
        # (1,). Restore that feature dimension for the state concatenation.
        if gripper.ndim == 1:
            gripper = gripper[:, None]

        if joint.ndim != 2 or joint.shape[1] != 7:
            raise RuntimeError(f"{path}: joint_position shape is {joint.shape}, expected (N, 7)")
        if gripper.ndim != 2 or gripper.shape[1] != 1:
            raise RuntimeError(f"{path}: gripper_position shape is {gripper.shape}, expected (N, 1)")
        if action.ndim != 2 or action.shape[1] != 8:
            raise RuntimeError(f"{path}: actions shape is {action.shape}, expected (N, 8)")

        state = np.concatenate([joint, gripper], axis=1)
        states.append(state)
        actions.append(action)
        for i in range(len(episode)):
            episodes[int(episode[i])].append(
                (int(frame[i]), float(timestamp[i]), state[i], action[i])
            )

    return np.concatenate(states), np.concatenate(actions), episodes


def validate_raw_arrays(
    states: np.ndarray, actions: np.ndarray, episodes: dict[int, list[tuple]]
) -> tuple[list[str], list[str], tuple[int, np.ndarray]]:
    failures = []
    warnings = []
    if states.shape[1] != 8:
        failures.append(f"raw state width is {states.shape[1]}, expected 8")
    if actions.shape[1] != 8:
        failures.append(f"raw action width is {actions.shape[1]}, expected 8")
    if not np.isfinite(states).all():
        failures.append("raw states contain NaN or infinity")
    if not np.isfinite(actions).all():
        failures.append("raw actions contain NaN or infinity")

    transition_episode = None
    episodes_with_close = 0
    episodes_with_release = 0
    for episode_index, rows in episodes.items():
        rows.sort(key=lambda row: row[0])
        frames = np.asarray([row[0] for row in rows])
        timestamps = np.asarray([row[1] for row in rows])
        episode_actions = np.stack([row[3] for row in rows])
        if np.any(np.diff(frames) <= 0):
            failures.append(f"episode {episode_index}: frame indices are not strictly increasing")
        if np.any(np.diff(timestamps) <= 0):
            failures.append(f"episode {episode_index}: timestamps are not strictly increasing")

        closed = episode_actions[:, 7] >= 0.5
        rises = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
        falls = np.flatnonzero(closed[:-1] & (~closed[1:])) + 1
        episodes_with_close += bool(len(rises))
        episodes_with_release += bool(len(falls))
        if transition_episode is None and len(rises) and len(falls):
            transition_episode = (episode_index, episode_actions)

    total_episodes = len(episodes)
    if episodes_with_close != total_episodes:
        warnings.append(
            f"gripper close edge present in {episodes_with_close}/{total_episodes} episodes"
        )
    if episodes_with_release != total_episodes:
        warnings.append(
            f"gripper release edge present in {episodes_with_release}/{total_episodes} episodes"
        )
    if transition_episode is None:
        failures.append("no episode contains both a gripper close and release transition")
        transition_episode = (next(iter(episodes)), actions)

    max_velocity = float(np.max(np.abs(actions[:, :7])))
    if max_velocity > 2.5:
        failures.append(f"joint velocity action reaches {max_velocity:.3f} rad/s (>2.5)")
    elif max_velocity > 2.0:
        warnings.append(f"joint velocity action reaches {max_velocity:.3f} rad/s")

    return failures, warnings, transition_episode


def normalized_metrics(name: str, values: np.ndarray) -> tuple[dict, list[str], list[str]]:
    active = np.asarray(values, dtype=np.float64)
    metrics = {
        "shape": list(active.shape),
        "outside_q01_q99_fraction": float(np.mean(np.abs(active) > 1.0)),
        "outside_3x_fraction": float(np.mean(np.abs(active) > 3.0)),
        "max_abs": float(np.max(np.abs(active))),
    }
    failures = []
    warnings = []
    if not np.isfinite(active).all():
        failures.append(f"normalized {name} contains NaN or infinity")
    if metrics["outside_q01_q99_fraction"] > 0.35:
        failures.append(
            f"normalized {name}: {metrics['outside_q01_q99_fraction']:.1%} lies outside [-1, 1]"
        )
    elif metrics["outside_q01_q99_fraction"] > 0.10:
        warnings.append(
            f"normalized {name}: {metrics['outside_q01_q99_fraction']:.1%} lies outside [-1, 1]"
        )
    if metrics["outside_3x_fraction"] > 0.05 or metrics["max_abs"] > 10.0:
        failures.append(
            f"normalized {name} is extreme: max_abs={metrics['max_abs']:.3f}, "
            f"outside_abs3={metrics['outside_3x_fraction']:.1%}"
        )
    elif metrics["max_abs"] > 3.0:
        warnings.append(f"normalized {name}: max_abs={metrics['max_abs']:.3f}")
    return metrics, failures, warnings


def transition_chunk(actions: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
    closed = actions[:, 7] >= 0.5
    transitions = np.flatnonzero(closed[:-1] != closed[1:]) + 1
    center = int(transitions[0]) if len(transitions) else len(actions) // 2
    start = max(0, min(center - horizon // 2, len(actions) - horizon))
    chunk = actions[start : start + horizon]
    if len(chunk) != horizon:
        raise RuntimeError(f"Could not extract a {horizon}-step action chunk")
    return chunk, start


def validate_openpi_batch(config_name: str, batch_size: int):
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    config = dataclasses.replace(
        _config.get_config(config_name), batch_size=batch_size, num_workers=0
    )
    loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        framework="pytorch",
    )
    observation, model_actions = next(iter(loader))
    model_state = np.asarray(observation.state)
    model_actions = np.asarray(model_actions)
    return loader.data_config(), model_state, model_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--config", default="pi05_fr3_real_droid_full")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    failures: list[str] = []
    warnings: list[str] = []
    states, actions, episodes = load_scalar_data(dataset_dir)
    raw_failures, raw_warnings, representative = validate_raw_arrays(
        states, actions, episodes
    )
    failures.extend(raw_failures)
    warnings.extend(raw_warnings)

    data_config, model_state, model_actions = validate_openpi_batch(
        args.config, args.batch_size
    )
    expected_state_shape = (args.batch_size, 32)
    expected_action_shape = (args.batch_size, args.horizon, 32)
    if model_state.shape != expected_state_shape:
        failures.append(
            f"model state shape is {model_state.shape}, expected {expected_state_shape}"
        )
    if model_actions.shape != expected_action_shape:
        failures.append(
            f"model action shape is {model_actions.shape}, expected {expected_action_shape}"
        )

    state_metrics, metric_failures, metric_warnings = normalized_metrics(
        "state", model_state[..., :8]
    )
    failures.extend(metric_failures)
    warnings.extend(metric_warnings)
    action_metrics, metric_failures, metric_warnings = normalized_metrics(
        "actions", model_actions[..., :8]
    )
    failures.extend(metric_failures)
    warnings.extend(metric_warnings)

    if model_state.shape[-1] >= 32 and not np.allclose(model_state[..., 8:], 0.0):
        failures.append("model state padding dimensions 8:32 are not zero")
    if model_actions.shape[-1] >= 32 and not np.allclose(model_actions[..., 8:], 0.0):
        failures.append("model action padding dimensions 8:32 are not zero")

    episode_index, episode_actions = representative
    raw_chunk, chunk_start = transition_chunk(episode_actions, args.horizon)
    gripper_chunk = raw_chunk[:, 7].tolist()
    stats = data_config.norm_stats
    if stats is None or "actions" not in stats:
        failures.append("DROID action normalization statistics are unavailable")
        decoded_chunk = raw_chunk
        reconstruction_error = None
    else:
        action_stats = stats["actions"]
        if action_stats.q01 is None or action_stats.q99 is None:
            failures.append("DROID action statistics do not contain q01 and q99")
            decoded_chunk = raw_chunk
            reconstruction_error = None
        else:
            q01 = np.asarray(action_stats.q01, dtype=np.float64)[:8]
            q99 = np.asarray(action_stats.q99, dtype=np.float64)[:8]
            if q01.shape != (8,) or q99.shape != (8,):
                failures.append(
                    f"DROID action q01/q99 widths are {q01.shape}/{q99.shape}, expected (8,)"
                )
                decoded_chunk = raw_chunk
                reconstruction_error = None
            else:
                normalized_chunk = (raw_chunk - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                padded_chunk = np.zeros((args.horizon, 32), dtype=np.float64)
                padded_chunk[:, :8] = normalized_chunk
                decoded_chunk = (padded_chunk[:, :8] + 1.0) / 2.0 * (
                    q99 - q01 + 1e-6
                ) + q01
                reconstruction_error = float(np.max(np.abs(decoded_chunk - raw_chunk)))
                if reconstruction_error > 1e-5:
                    failures.append(
                        f"action normalization round-trip error is {reconstruction_error:.3e}"
                    )

    joint_deltas = np.diff(decoded_chunk[:, :7], axis=0)
    max_step_change = float(np.max(np.abs(joint_deltas)))
    if max_step_change > 1.5:
        failures.append(
            f"decoded chunk joint velocity changes by {max_step_change:.3f} rad/s in one 15 Hz step"
        )
    elif max_step_change > 0.75:
        warnings.append(
            f"decoded chunk joint velocity changes by {max_step_change:.3f} rad/s in one 15 Hz step"
        )

    report = {
        "dataset_dir": str(dataset_dir),
        "episodes": len(episodes),
        "frames": int(len(states)),
        "raw_state_shape": list(states.shape),
        "raw_action_shape": list(actions.shape),
        "model_state_shape": list(model_state.shape),
        "model_action_shape": list(model_actions.shape),
        "normalized_state": state_metrics,
        "normalized_actions": action_metrics,
        "decoded_chunk_episode": int(episode_index),
        "decoded_chunk_start": int(chunk_start),
        "decoded_chunk_gripper": gripper_chunk,
        "decoded_chunk_roundtrip_max_error": reconstruction_error,
        "decoded_chunk_max_joint_step_change": max_step_change,
        "warnings": warnings,
        "failures": failures,
    }

    print("=== FR3 pi0.5-DROID preflight ===")
    print(json.dumps(report, indent=2))
    print("\nDecoded 16-step action chunk [7 joint velocities, gripper]:")
    print(np.array2string(decoded_chunk, precision=4, suppress_small=True))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote report: {args.report}")

    if failures:
        raise SystemExit(f"PREFLIGHT FAILED with {len(failures)} issue(s)")
    print("\nPREFLIGHT PASSED")


if __name__ == "__main__":
    main()
