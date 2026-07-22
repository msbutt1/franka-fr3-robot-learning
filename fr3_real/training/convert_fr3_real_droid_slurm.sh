#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/convert_fr3_real_droid_slurm.sh
# Convert successful physical FR3 RealSense episodes to DROID-keyed LeRobot.
# The project root defaults to this script's directory and can be overridden
# at submission time with: sbatch --export=ALL,FR3=/path/to/franka_r3 ...

#SBATCH --job-name=convert_fr3_real_droid
#SBATCH --account=def-mqp2259
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FR3="${FR3:-${SCRIPT_DIR}}"
OPENPI_DIR="${FR3}/openpi"
RAW_DIR="${FR3}/fr3_real_recordings_droid"
LEROBOT_CACHE="${FR3}/lerobot_cache"
REPO_ID="local/fr3_real_pick_place_droid"

export UV_CACHE_DIR="${FR3}/uv_cache"
export HF_HOME="${FR3}/huggingface_cache"
export OPENPI_DATA_HOME="${FR3}/openpi_cache"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${OPENPI_DATA_HOME}"

if [ ! -d "${RAW_DIR}" ]; then
    echo "Missing raw recordings: ${RAW_DIR}" >&2
    exit 1
fi

TMP_CACHE="${SLURM_TMPDIR}/lerobot_cache"
mkdir -p "${TMP_CACHE}"
export HF_LEROBOT_HOME="${TMP_CACHE}"

cd "${OPENPI_DIR}"
source .venv/bin/activate

python "${SCRIPT_DIR}/../data-pipeline/convert_raw_recordings_to_lerobot.py" \
    --raw_dir "${RAW_DIR}" \
    --output_dir "${TMP_CACHE}" \
    --repo_id "${REPO_ID}" \
    --schema droid_joint_velocity \
    --task "Pick up the blue cube and place it in the basket." \
    --fps 15 \
    --image_width 320 \
    --image_height 240 \
    --gripper_max_width 0.08 \
    --overwrite

mkdir -p "${LEROBOT_CACHE}/$(dirname "${REPO_ID}")"
rm -rf "${LEROBOT_CACHE:?}/${REPO_ID}"
cp -r "${TMP_CACHE}/${REPO_ID}" "${LEROBOT_CACHE}/${REPO_ID}"
du -sh "${LEROBOT_CACHE}/${REPO_ID}"
