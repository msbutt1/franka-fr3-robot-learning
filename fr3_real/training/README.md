# FR3 pi0.5-DROID Full Fine-Tuning

This folder contains the local setup for training pi0.5-DROID on the physical
FR3 recordings. It does not vendor or clone OpenPI; the Slurm jobs expect an
existing OpenPI checkout on the server.

## What This Uses

- Base checkpoint: `gs://openpi-assets/checkpoints/pi05_droid/params`
- Normalization assets: `gs://openpi-assets/checkpoints/pi05_droid/assets`, asset id `droid`
- Training config name: `pi05_fr3_real_droid_full`
- Dataset id: `local/fr3_real_pick_place_droid`
- Batch size: `32`
- Action space: 8D DROID action, `[dq_d joint velocities (7), gripper position (1)]`
- FPS: 15 Hz, using `--sample_stride 4` for 60 Hz raw recordings
- Fine-tuning mode: full parameter fine-tuning, no LoRA freeze filter and no LoRA model variants

The converter writes the two DROID inputs used by pi0.5-DROID:

- `exterior_image_1_left`
- `wrist_image_left`
- `joint_position`
- `gripper_position`
- `actions`

## Expected Server Layout

Defaults follow the existing project convention:

```bash
${HOME}/projects/def-mqp2259/franka_r3/openpi
${HOME}/projects/def-mqp2259/franka_r3/lerobot_cache
```

The scripts infer this repository from the script path. Override paths if your
server layout differs:

```bash
export FR3="${HOME}/projects/def-mqp2259/franka_r3"
export OPENPI_DIR="${FR3}/openpi"
export LEROBOT_CACHE="${FR3}/lerobot_cache"
export RAW_DIR="/path/to/fr3_raw_recordings"
```

## Run

Convert raw recordings to LeRobot:

```bash
sbatch fr3_real/training/convert_fr3_real_droid_joint_velocity.sbatch
```

Launch the full fine-tune:

```bash
sbatch fr3_real/training/train_pi05_fr3_real_droid_full.sbatch
```

The training job patches `${OPENPI_DIR}/src/openpi/training/config.py`
idempotently by adding a managed FR3 data config and train config block, then
runs:

```bash
uv run scripts/train.py pi05_fr3_real_droid_full --exp-name=fr3_real_droid_full_v1
```

If a checkpoint already exists under
`${OPENPI_DIR}/checkpoints/pi05_fr3_real_droid_full/fr3_real_droid_full_v1`,
the job automatically adds `--resume`.

## Useful Overrides

```bash
DATASET_REPO_ID=local/fr3_real_pick_place_droid \
EXP_NAME=fr3_real_droid_full_v2 \
NUM_TRAIN_STEPS=30000 \
sbatch fr3_real/training/train_pi05_fr3_real_droid_full.sbatch
```

Keep `BATCH_SIZE<=32`; both the shell job and the OpenPI patcher refuse larger
values.

To use actual measured velocities instead of desired velocities:

```bash
JOINT_VELOCITY_KEY=dq sbatch fr3_real/training/convert_fr3_real_droid_joint_velocity.sbatch
```

The default is `dq_d`, which is the desired joint velocity stream and is the
closest match to DROID's commanded joint-velocity action field.
