#!/usr/bin/env bash
# Usage: from fr3_real/, run: bash training/run_fr3_real_dell_full.sh
# Convert physical FR3 recordings, then launch the full pi0.5 DROID fine-tune.
# Intended for: setsid nohup ./run_fr3_real_dell_full.sh > full.log 2>&1 < /dev/null &

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FR3="${FR3:-${HOME}/franka_r3}"
OPENPI_DIR="${FR3}/openpi"
RAW_DIR="${FR3}/fr3_real_recordings_droid"
LEROBOT_CACHE="${FR3}/lerobot_cache"
REPO_ID="local/fr3_real_pick_place_droid"
EXP_NAME="${EXP_NAME:-fr3_real_droid_full_dell_v1}"

export HF_LEROBOT_HOME="${LEROBOT_CACHE}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

test -d "${RAW_DIR}"
test -f "${SCRIPT_DIR}/../data-pipeline/convert_raw_recordings_to_lerobot.py"
test -d "${OPENPI_DIR}"

echo "=== $(date -Is): conversion begins ==="
cd "${OPENPI_DIR}"
time uv run --no-sync python "${SCRIPT_DIR}/../data-pipeline/convert_raw_recordings_to_lerobot.py" \
  --raw_dir "${RAW_DIR}" \
  --output_dir "${LEROBOT_CACHE}" \
  --repo_id "${REPO_ID}" \
  --schema droid_joint_velocity \
  --task "Pick up the blue cube and place it in the basket." \
  --fps 15 \
  --image_width 320 \
  --image_height 240 \
  --gripper_max_width 0.08 \
  --overwrite

echo "=== $(date -Is): full fine-tune begins ==="
time uv run --no-sync scripts/train.py pi05_fr3_real_droid_full \
  --exp-name "${EXP_NAME}"
echo "=== $(date -Is): complete ==="
