#!/usr/bin/env bash
#SBATCH --job-name=pi05-fr3-smoke
#SBATCH --account=def-mqp2259_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/pi05-fr3-smoke-%j.out
#SBATCH --error=logs/pi05-fr3-smoke-%j.err

set -euo pipefail

# Submit with:
#   DATASET=/absolute/path/to/lerobot_dataset OUTPUT=/absolute/path/to/pi05_smoke sbatch fr3_real/jobs/pi05_lora_smoke.sbatch
#
# Optional overrides:
#   IMAGE=/absolute/path/to/ioaitech_train_openpi_pi05.sif
#   RUNTIME=apptainer|docker
#   STEPS=10
#   SAVE_INTERVAL=10
#   ACTION_HORIZON=50
#   PROMPT="Pick up the cube from the cell and place it in the basket."

DATASET="${DATASET:?Set DATASET to the LeRobot dataset root containing meta/info.json}"
OUTPUT="${OUTPUT:?Set OUTPUT to a writable output directory for this smoke run}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-$OUTPUT/workspace}"
IMAGE="${IMAGE:-containers/train_openpi_pi05.sif}"
RUNTIME="${RUNTIME:-apptainer}"
STEPS="${STEPS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
PROMPT="${PROMPT:-Pick up the cube from the cell and place it in the basket.}"

mkdir -p logs "$OUTPUT" "$CONTAINER_WORKSPACE"

module load apptainer/1.4.5

echo "[smoke] host=$(hostname)"
echo "[smoke] date=$(date --iso-8601=seconds)"
echo "[smoke] dataset=$DATASET"
echo "[smoke] output=$OUTPUT"
echo "[smoke] container workspace=$CONTAINER_WORKSPACE"
echo "[smoke] image=$IMAGE"
echo "[smoke] runtime=$RUNTIME"

test -f "$DATASET/meta/info.json" || { echo "[smoke:error] missing $DATASET/meta/info.json"; exit 2; }
test -d "$DATASET/data" || { echo "[smoke:error] missing $DATASET/data"; exit 2; }
test -d "$DATASET/videos" || { echo "[smoke:error] missing $DATASET/videos"; exit 2; }

echo "[smoke] dataset info:"
python3 - "$DATASET/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
keys = [
    "repo_id",
    "codebase_version",
    "robot_type",
    "fps",
    "total_episodes",
    "total_frames",
    "features",
]
for key in keys:
    value = info.get(key)
    if key == "features" and isinstance(value, dict):
        print(f"  features={list(value)}")
    else:
        print(f"  {key}={value}")
PY

echo "[smoke] critical setting: --num_workers 0 avoids multiprocessing pickling during norm stats"
echo "[smoke] train args: --batch_size 1 --steps $STEPS --save_interval $SAVE_INTERVAL --learning_rate 2.5e-5 --action_horizon $ACTION_HORIZON --num_workers 0 --gpus all --fsdp_devices 1 --lora auto --prompt $PROMPT"

case "$RUNTIME" in
  apptainer|singularity)
    command -v "$RUNTIME" >/dev/null 2>&1 || { echo "[smoke:error] $RUNTIME not found"; exit 127; }
    "$RUNTIME" run --nv \
      --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
      --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
      --env WANDB_MODE=disabled \
      --bind "$DATASET:/data/input:ro" \
      --bind "$OUTPUT:/data/output" \
      --bind "$CONTAINER_WORKSPACE:/workspace" \
      "$IMAGE" \
      --batch_size 1 \
      --steps "$STEPS" \
      --save_interval "$SAVE_INTERVAL" \
      --learning_rate 2.5e-5 \
      --action_horizon "$ACTION_HORIZON" \
      --num_workers 0 \
      --gpus all \
      --fsdp_devices 1 \
      --lora auto \
      --prompt "$PROMPT"
    ;;
  docker)
    command -v docker >/dev/null 2>&1 || { echo "[smoke:error] docker not found"; exit 127; }
    DOCKER_IMAGE="${IMAGE#docker://}"
    docker run --rm --gpus all \
      -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
      -e WANDB_MODE=disabled \
      -v "$DATASET:/data/input:ro" \
      -v "$OUTPUT:/data/output" \
      "$DOCKER_IMAGE" \
      --batch_size 1 \
      --steps "$STEPS" \
      --save_interval "$SAVE_INTERVAL" \
      --learning_rate 2.5e-5 \
      --action_horizon "$ACTION_HORIZON" \
      --num_workers 0 \
      --gpus all \
      --fsdp_devices 1 \
      --lora auto \
      --prompt "$PROMPT"
    ;;
  *)
    echo "[smoke:error] unsupported RUNTIME=$RUNTIME; use apptainer, singularity, or docker"
    exit 2
    ;;
esac

echo "[smoke] done"
echo "[smoke] expected output under $OUTPUT/docker_train/train/"
