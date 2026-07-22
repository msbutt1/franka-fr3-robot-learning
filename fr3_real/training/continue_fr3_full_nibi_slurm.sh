#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/continue_fr3_full_nibi_slurm.sh
# Resume the valid Nibi step-7000 full run to a 12k-step diagnostic.

#SBATCH --job-name=pi05_fr3_continue_12k
#SBATCH --account=def-mqp2259
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
# The prior 96G job was OOM-killed during final asynchronous checkpoint save.
# 128G leaves headroom for host-resident optimizer/checkpoint buffers.
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm may execute a staged copy from /var/spool; submit-dir is the project root.
FR3="${FR3:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
OPENPI_DIR="${FR3}/openpi"
CONFIG="${CONFIG:-pi05_fr3_real_droid_full}"
EXP_NAME="${EXP_NAME:-fr3_real_droid_full_v1}"
TARGET_STEPS="${TARGET_STEPS:-12000}"
CHECKPOINT_DIR="${OPENPI_DIR}/checkpoints/${CONFIG}/${EXP_NAME}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${FR3}/lerobot_cache}"
export HF_HOME="${HF_HOME:-${FR3}/huggingface_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FR3}/uv_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${FR3}/openpi_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FR3}/xdg_cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${FR3}/jax_cache}"
export JAX_COMPILATION_CACHE_MAX_SIZE="${JAX_COMPILATION_CACHE_MAX_SIZE:-21474836480}"
export WANDB_DIR="${WANDB_DIR:-${FR3}/wandb}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
  "${HF_HOME}" "${UV_CACHE_DIR}" "${OPENPI_DATA_HOME}" \
  "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${WANDB_DIR}"
test -d "${OPENPI_DIR}" || { echo "Missing OpenPI checkout: ${OPENPI_DIR}" >&2; exit 2; }
test -d "${CHECKPOINT_DIR}" || { echo "Missing checkpoint run: ${CHECKPOINT_DIR}" >&2; exit 2; }

latest_step="$({
  find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' 2>/dev/null || true
} | awk '/^[0-9]+$/ { print }' | sort -n | tail -1)"

if [[ -z "${latest_step}" ]]; then
  echo "No complete numeric checkpoint exists under ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}/${latest_step}/train_state" ]]; then
  echo "Checkpoint ${latest_step} has no train_state and cannot be resumed." >&2
  exit 2
fi
if (( latest_step >= TARGET_STEPS )); then
  echo "Latest checkpoint ${latest_step} already reached target ${TARGET_STEPS}."
  exit 0
fi

cd "${OPENPI_DIR}"
source .venv/bin/activate

echo "[continue] host=$(hostname) job=${SLURM_JOB_ID:-interactive}"
echo "[continue] config=${CONFIG} experiment=${EXP_NAME}"
echo "[continue] latest_complete_step=${latest_step} target_steps=${TARGET_STEPS}"
echo "[continue] original decay remains clamped at the final 1e-6 learning rate"
nvidia-smi

uv run --no-sync scripts/train.py "${CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --num-train-steps "${TARGET_STEPS}" \
  --resume
