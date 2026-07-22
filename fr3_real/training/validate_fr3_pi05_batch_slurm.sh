#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/validate_fr3_pi05_batch_slurm.sh
# Validate the v3 LeRobot records and one OpenPI-transformed training batch.

#SBATCH --job-name=validate_fr3_v3_pi05
#SBATCH --account=def-mqp2259
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm copies the submitted script into /var/spool before executing it. Use
# the submission directory as the experiment root when running under Slurm.
FR3="${FR3:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
OPENPI_DIR="${FR3}/openpi"
LEROBOT_CACHE="${FR3}/lerobot_cache"
DATASET="${LEROBOT_CACHE}/local/fr3_real_pick_place_droid_v3"

export HF_LEROBOT_HOME="${LEROBOT_CACHE}"
export UV_CACHE_DIR="${FR3}/uv_cache"
export HF_HOME="${FR3}/huggingface_cache"
export OPENPI_DATA_HOME="${FR3}/openpi_cache"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

cd "${OPENPI_DIR}"
source .venv/bin/activate

uv run --no-sync python "${SCRIPT_DIR}/../model-validation/validate_fr3_pi05_batch.py" \
    --dataset_dir "${DATASET}" \
    --config pi05_fr3_real_droid_full \
    --batch_size 32 \
    --horizon 16 \
    --report "${FR3}/fr3_pi05_preflight_report.json"
