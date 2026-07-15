#!/usr/bin/env bash
# Convert FR3 raw recordings to a LeRobot dataset for pi0.5-DROID full fine-tuning.
# Submit with: sbatch fr3_real/training/convert_fr3_real_droid_joint_velocity.sbatch

#SBATCH --job-name=convert_fr3_droid_jv
#SBATCH --account=def-mqp2259
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
FR3="${FR3:-${HOME}/projects/def-mqp2259/franka_r3}"
OPENPI_DIR="${OPENPI_DIR:-${FR3}/openpi}"
LEROBOT_CACHE="${LEROBOT_CACHE:-${FR3}/lerobot_cache}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/fr3_real_pick_place_droid}"
DATASET_DIR="${DATASET_DIR:-${LEROBOT_CACHE}/${DATASET_REPO_ID}}"
RAW_DIR="${RAW_DIR:-${REPO_DIR}/fr3_raw_recordings}"
TASK="${TASK:-Pick up the cube from the cell and place it in the basket.}"
FPS="${FPS:-15}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-4}"
GRIPPER_MAX_WIDTH="${GRIPPER_MAX_WIDTH:-0.08}"
JOINT_VELOCITY_KEY="${JOINT_VELOCITY_KEY:-dq_d}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

source "${SCRIPT_DIR}/scratch_env.sh"
export HF_LEROBOT_HOME="${LEROBOT_CACHE}"

echo "Repo:        ${REPO_DIR}"
echo "OpenPI:      ${OPENPI_DIR}"
echo "Raw data:    ${RAW_DIR}"
echo "Dataset id:  ${DATASET_REPO_ID}"
echo "Dataset dir: ${DATASET_DIR}"
echo "FPS:         ${FPS}"
echo "Stride:      ${SAMPLE_STRIDE}"
echo "Velocity:    ${JOINT_VELOCITY_KEY}"
echo "UV cache:    ${UV_CACHE_DIR}"
echo "Temp dir:    ${TMPDIR}"
echo "OpenPI cache:${OPENPI_DATA_HOME}"

if [ ! -d "${RAW_DIR}" ]; then
    echo "Missing raw recordings: ${RAW_DIR}" >&2
    exit 1
fi

if [ ! -d "${OPENPI_DIR}" ]; then
    echo "Missing OpenPI checkout: ${OPENPI_DIR}" >&2
    exit 1
fi

mkdir -p "${LEROBOT_CACHE}" "$(dirname "${DATASET_DIR}")"

cd "${OPENPI_DIR}"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python "${REPO_DIR}/fr3_real/datasets/convert_raw_recordings_to_lerobot.py" \
    --raw_dir "${RAW_DIR}" \
    --output_dir "${DATASET_DIR}" \
    --repo_id "${DATASET_REPO_ID}" \
    --task "${TASK}" \
    --fps "${FPS}" \
    --schema openpi_droid_joint_velocity \
    --camera_indices 0 1 \
    --sample_stride "${SAMPLE_STRIDE}" \
    --gripper_max_width "${GRIPPER_MAX_WIDTH}" \
    --joint_velocity_key "${JOINT_VELOCITY_KEY}" \
    ${OVERWRITE_FLAG}

python "${REPO_DIR}/fr3_real/training/inspect_pi05_fr3_lerobot_dataset.py" \
    "${DATASET_DIR}" \
    --expected-fps "${FPS}" \
    --strict-fps

echo "Converted dataset is ready: ${DATASET_DIR}"
