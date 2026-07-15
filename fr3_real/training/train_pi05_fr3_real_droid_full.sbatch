#!/usr/bin/env bash
# Full fine-tune pi0.5-DROID on the physical FR3 LeRobot dataset.
# Run after conversion with: sbatch fr3_real/training/train_pi05_fr3_real_droid_full.sbatch

#SBATCH --job-name=pi05_fr3_real_full
#SBATCH --account=def-mqp2259
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scratch_env.sh"
CONFIG="${CONFIG:-pi05_fr3_real_droid_full}"
EXP_NAME="${EXP_NAME:-fr3_real_droid_full_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
CHECKPOINT="${CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_droid/params}"
ASSETS_DIR="${ASSETS_DIR:-gs://openpi-assets/checkpoints/pi05_droid/assets}"
ASSET_ID="${ASSET_ID:-droid}"

export HF_LEROBOT_HOME="${LEROBOT_CACHE}"

if [ "${BATCH_SIZE}" -gt 32 ]; then
    echo "Refusing BATCH_SIZE=${BATCH_SIZE}; requested maximum is 32." >&2
    exit 2
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Repo:        ${REPO_DIR}"
echo "OpenPI:      ${OPENPI_DIR}"
echo "Dataset id:  ${DATASET_REPO_ID}"
echo "Dataset dir: ${DATASET}"
echo "Config:      ${CONFIG}"
echo "Run:         ${EXP_NAME}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Steps:       ${NUM_TRAIN_STEPS}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Assets:      ${ASSETS_DIR} (${ASSET_ID})"
echo "UV cache:    ${UV_CACHE_DIR}"
echo "Temp dir:    ${TMPDIR}"
echo "OpenPI cache:${OPENPI_DATA_HOME}"

if [ ! -d "${OPENPI_DIR}" ]; then
    echo "Missing OpenPI checkout: ${OPENPI_DIR}" >&2
    exit 1
fi

if [ ! -f "${DATASET}/meta/info.json" ]; then
    echo "Missing converted dataset: ${DATASET}" >&2
    echo "Run convert_fr3_real_droid_joint_velocity.sbatch first." >&2
    exit 1
fi

cd "${OPENPI_DIR}"
source .venv/bin/activate

python "${REPO_DIR}/fr3_real/training/prepare_pi05_fr3_real_droid_full.py" \
    --openpi-dir "${OPENPI_DIR}" \
    --config-name "${CONFIG}" \
    --repo-id "${DATASET_REPO_ID}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --checkpoint "${CHECKPOINT}" \
    --assets-dir "${ASSETS_DIR}" \
    --asset-id "${ASSET_ID}"

python -c "import ast; ast.parse(open('src/openpi/training/config.py').read()); print('config.py parses OK')"
python "${REPO_DIR}/fr3_real/training/inspect_pi05_fr3_lerobot_dataset.py" "${DATASET}" --expected-fps 15
nvidia-smi

CKPT_DIR="${OPENPI_DIR}/checkpoints/${CONFIG}/${EXP_NAME}"
RESUME_FLAG=()
if [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    RESUME_FLAG=(--resume)
    echo "Resuming from ${CKPT_DIR}"
fi

uv run scripts/train.py "${CONFIG}" --exp-name="${EXP_NAME}" "${RESUME_FLAG[@]}" \
    2>&1 | tee "${FR3}/${EXP_NAME}.log"
