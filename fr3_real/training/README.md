# FR3 pi0.5-DROID Full Fine-Tuning

This folder contains the local setup for training pi0.5-DROID on the physical
FR3 recordings. It does not vendor or clone OpenPI; the Slurm jobs expect an
existing OpenPI checkout on the server.

## What This Uses

- Default checkpoint: `gs://openpi-assets/checkpoints/pi05_droid/params`
- Normalization assets: `gs://openpi-assets/checkpoints/pi05_droid/assets`, asset id `droid`
- Training config name: `pi05_fr3_real_droid_full`
- Dataset id: `local/fr3_real_pick_place_droid`
- Batch size: `32`
- Action space: 8D DROID action, `[dq_d joint velocities (7), gripper position (1)]`
- FPS: 15 Hz, using `--sample_stride 4` for 60 Hz raw recordings
- Fine-tuning mode: full parameter fine-tuning, no LoRA freeze filter and no LoRA model variants

This default follows OpenPI's custom DROID fine-tuning config: start from the
pi0.5-DROID checkpoint and reuse DROID normalization assets. That is the
recommended path for FR3 data converted into DROID-style joint velocity actions.
The general pi0.5 base checkpoint is also available at
`gs://openpi-assets/checkpoints/pi05_base/params`, but using it is a different
experiment: it starts before DROID specialization and may need more training and
more careful normalization validation.

The converter writes the two DROID inputs used by pi0.5-DROID:

- `exterior_image_1_left`
- `wrist_image_left`
- `joint_position`
- `gripper_position`
- `actions`

## Expected Server Layout

Defaults follow the existing project convention:

```bash
/home/lihuyue/scratch/anastasia_fr3/openpi
/home/lihuyue/scratch/anastasia_fr3/lerobot_cache
/home/lihuyue/scratch/anastasia_fr3/openpi_cache
```

The scripts infer this repository from the script path. Override paths if your
server layout differs:

```bash
export FR3="/home/lihuyue/scratch/anastasia_fr3"
export OPENPI_DIR="${FR3}/openpi"
export LEROBOT_CACHE="${FR3}/lerobot_cache"
export RAW_DIR="/path/to/fr3_raw_recordings"
source fr3_real/training/scratch_env.sh
```

Source `scratch_env.sh` before OpenPI install, conversion, and training. It
keeps `uv`, `pip`, Hugging Face, OpenPI checkpoint, W&B, Torch, Matplotlib, and
temporary-file caches under `${FR3}` instead of `${HOME}`.

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

The generated OpenPI config loads the checkpoint from source with:

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_droid/params"
)
```

To explicitly patch OpenPI with the recommended pi0.5-DROID source checkpoint:

```bash
python fr3_real/training/prepare_pi05_fr3_real_droid_full.py \
  --openpi-dir "${OPENPI_DIR}" \
  --checkpoint gs://openpi-assets/checkpoints/pi05_droid/params \
  --assets-dir gs://openpi-assets/checkpoints/pi05_droid/assets \
  --asset-id droid
```

Before submitting the training job, verify that the server can see the source
checkpoint:

```bash
gsutil ls gs://openpi-assets/checkpoints/pi05_droid/params
gsutil ls gs://openpi-assets/checkpoints/pi05_droid/assets/droid
```

If the compute node cannot read GCS during training, prefetch the checkpoint and
assets on a login node, then point the job at local paths:

```bash
mkdir -p "${FR3}/openpi_assets/checkpoints"
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid "${FR3}/openpi_assets/checkpoints/"

CHECKPOINT="${FR3}/openpi_assets/checkpoints/pi05_droid/params" \
ASSETS_DIR="${FR3}/openpi_assets/checkpoints/pi05_droid/assets" \
ASSET_ID=droid \
sbatch fr3_real/training/train_pi05_fr3_real_droid_full.sbatch
```

To run the Slurm training job from the general pi0.5 base checkpoint instead:

```bash
CHECKPOINT=gs://openpi-assets/checkpoints/pi05_base/params \
ASSETS_DIR=gs://openpi-assets/checkpoints/pi05_base/assets \
ASSET_ID=droid \
sbatch fr3_real/training/train_pi05_fr3_real_droid_full.sbatch
```

For this FR3/DROID-style run, keep the default `pi05_droid` source unless you
intentionally want a less-specialized base-model experiment.

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
