#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/train_pi05_fr3_real_droid_lora_r16_slurm.sh
# Train pi0.5-DROID on physical FR3 episodes using rank-16 attention LoRA.
# Run after conversion with: sbatch train_pi05_fr3_real_droid_lora_r16_slurm.sh

#SBATCH --job-name=pi05_fr3_real_droid_r16
#SBATCH --account=def-mqp2259
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FR3="${FR3:-${SCRIPT_DIR}}"
OPENPI_DIR="${FR3}/openpi"
LEROBOT_CACHE="${FR3}/lerobot_cache"
CONFIG="pi05_fr3_real_droid_lora_r16"
EXP_NAME="fr3_real_droid_r16_v1"

export HF_LEROBOT_HOME="${LEROBOT_CACHE}"
export UV_CACHE_DIR="${FR3}/uv_cache"
export HF_HOME="${FR3}/huggingface_cache"
export OPENPI_DATA_HOME="${FR3}/openpi_cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${OPENPI_DATA_HOME}"

cd "${OPENPI_DIR}"
source .venv/bin/activate

python "${SCRIPT_DIR}/prepare_pi05_fr3_real_droid_lora_r16.py"
python -c "import ast; ast.parse(open('src/openpi/training/config.py').read()); print('config.py parses OK')"

DATASET="${LEROBOT_CACHE}/local/fr3_real_pick_place_droid"
if [ ! -d "${DATASET}" ]; then
    echo "Missing converted dataset: ${DATASET}" >&2
    exit 1
fi

CKPT_DIR="${OPENPI_DIR}/checkpoints/${CONFIG}/${EXP_NAME}"
RESUME_FLAG=""
if [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    RESUME_FLAG="--resume"
fi

uv run --no-sync scripts/train.py "${CONFIG}" --exp-name "${EXP_NAME}" ${RESUME_FLAG} \
    2>&1 | tee "${FR3}/${EXP_NAME}.log"
