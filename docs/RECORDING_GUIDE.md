# FR3 Recording Guide

This guide describes how to collect new real-robot pick-and-place
demonstrations for the FR3 dataset. Run commands from `fr3_real/`. Collection
commands move the arm and gripper. Keep the E-stop reachable, inspect the
workspace before every confirmation prompt, and stop rather than improvising
when robot behavior differs from the expected path.

## What One Recording Contains

One successful episode starts with the cube on a table cell and the arm at a
consistent policy-start pose. The arm picks the cube, transports it to the
basket, opens the gripper, and returns home. During that segment the recorder
saves two RGB camera videos, camera timestamps, robot state samples, gripper
state, action records, events, and episode metadata. The result is a raw
episode folder, not yet a LeRobot dataset episode.

The normal `--recycle_from_basket` mode automates the repeatable setup: it
picks the cube from the basket, places it on the selected cell, returns home,
then records the cell-to-basket demonstration. That keeps the initial cube
pose and arm start condition consistent across episodes.

## 1. Prepare The Workcell

1. Confirm the workspace is clear, the cube matches the recorded cube size and
   orientation, the basket is in its calibrated location, and the E-stop is
   reachable.
2. Activate the robot-control environment and run read-only checks:

```bash
cd /path/to/franka-fr3-robot-learning/fr3_real
conda activate fr3-recording

python setup/verify_fr3_setup.py
python franka-tests/test_connection.py --robot_ip 172.16.0.2
```

3. Confirm both cameras before collecting data:

```bash
python franka-tests/preview_realsense.py --help
python franka-tests/test_realsense_recording.py --help
```

4. Use the active workcell calibration at `configs/probed_points.json`. Reprobe
   the workcell with `franka-tests/probe_points.py` after changing the table,
   basket, camera mounts, robot base, or coordinate convention. Do not copy a
   calibration file from another robot or workcell.

## 2. Choose Cells And Tracking Files

Cell lists specify which table coordinates to collect; tracker spreadsheets
record their status. The available run-selection files live in `configs/`.
Common examples are `working_cells.json`, `all_working_cells.json`, and the
corresponding `fr3_*_cell_tracker.xlsx` workbook.

Before a new collection pass, inspect the selected list rather than assuming a
tracker and JSON file describe the same cells:

```bash
python data-pipeline/filter_grid_tracker.py --help
python data-pipeline/filter_cells_by_tracker_status.py --help
```

For a small commissioning run, select one known-safe printed cell with
`--printed_cell`. For a resumed pass, use `--resume_from_file` and a run-local
resume file under `configs/`; do not overwrite a tracker or resume file from a
different collection campaign.

## 3. Record A Demonstration

Start with one cell. Replace the cell list and tracker paths only with files
that belong to the same collection campaign. `--record_dir` must point to a
new or existing directory outside Git; camera recordings are intentionally
ignored by the repository.

```bash
python franka-tests/pick_and_place.py \
  --robot_ip 172.16.0.2 \
  --points configs/probed_points.json \
  --cells_json configs/working_cells.json \
  --status_tracker configs/fr3_all_working_cell_tracker.xlsx \
  --printed_cell 104 \
  --recycle_from_basket \
  --record_dir /data/fr3_raw_recordings \
  --camera_width 640 \
  --camera_height 480 \
  --camera_fps 60
```

The script first performs unrecorded setup from basket to cell. It starts the
recording only after the cube is placed and the arm has returned to the
demonstrated start pose. A successful recorded segment is automatically marked
as `PASS` in the selected tracker and advances the resume file when one is
used. A failed recorded attempt is stored as unsuccessful raw data and should
not be treated as training data without a deliberate failure-data experiment.

For a multi-cell run, remove `--printed_cell`, set the appropriate
`--resume_file`, and add `--resume_from_file`:

```bash
python franka-tests/pick_and_place.py \
  --robot_ip 172.16.0.2 \
  --points configs/probed_points.json \
  --cells_json configs/working_cells.json \
  --status_tracker configs/fr3_all_working_cell_tracker.xlsx \
  --recycle_from_basket \
  --record_dir /data/fr3_raw_recordings \
  --resume_file configs/next_cell_to_record.txt \
  --resume_from_file
```

Use `--help` before changing speed, force, clearance, retry, or fallback
flags. Those parameters affect both safety and dataset consistency.

## 4. Inspect The Raw Data

Each episode directory should contain at least:

- `metadata.json`: task, cell, success status, camera setup, and timing.
- `robot_state.jsonl`: synchronized robot and gripper observations.
- `actions.jsonl`: recorded arm and gripper commands.
- `camera_*_rgb.mp4`: RGB video for each camera.
- `camera_*_timestamps.jsonl`: timestamp records for each video stream.

Validate the full raw collection before conversion:

```bash
python model-validation/validate_raw_recordings.py /data/fr3_raw_recordings
```

The validator excludes failed or unfinished episodes and exits nonzero when a
successful episode is missing required state, timing, or camera data. Fix or
quarantine those episodes before conversion. For broader data checks and
evaluation-leakage auditing, also run:

```bash
python model-validation/audit_fr3_new_demos.py --help
python model-validation/validate_fr3_pi05_batch.py --help
```

## 5. Convert To LeRobot

Conversion normally includes only successful raw episodes. The current
pi0.5-DROID deployment expects seven joint velocities plus gripper closedness
at 15 Hz, so use the `droid_joint_velocity` schema unless intentionally
training a different action representation.

```bash
python data-pipeline/convert_raw_recordings_to_lerobot.py \
  --raw_dir /data/fr3_raw_recordings \
  --output_dir /data/lerobot_cache/local \
  --repo_id msbutt1/fr3-pick-place-lerobot \
  --schema droid_joint_velocity \
  --fps 15
```

The converter writes the dataset below
`/data/lerobot_cache/local/msbutt1/fr3-pick-place-lerobot`. Keep the raw
recordings as the source of truth; the LeRobot output is derived data and can
be rebuilt after converter changes.

## 6. Publish Or Train

Review the converted dataset locally before publishing. When ready, use the
Hugging Face CLI to upload the dataset repository:

```bash
hf auth login
hf upload msbutt1/fr3-pick-place-lerobot \
  /data/lerobot_cache/local/msbutt1/fr3-pick-place-lerobot \
  --repo-type dataset
```

Use the launchers in `training/` only after data validation, conversion, and
normalization statistics are complete. Training should run on the OpenPI/GPU
environment, not on the robot-control workstation.

## Recording Rules

- Do not mix calibrations, cube geometry, camera placement, action format, or
  control frequency within one dataset version without recording that change.
- Do not mark visual near-misses as successful grasps. The collected episode
  must complete the physical pick and release sequence.
- Do not use live policy execution to generate demonstrations. The collection
  script produces the supervised actions; policy evaluation belongs under
  `model-running/`.
- Keep raw recordings, tracker state, conversion command, and dataset version
  together so each uploaded dataset can be reproduced.
