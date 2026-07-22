# FR3 Script Guide

Run commands below from `fr3_real/` unless a command says otherwise. Every
executable also has a purpose and usage line at its own top of file; use
`python <path> --help` before a first run. Scripts in `franka-tests/` and
`model-running/` can affect hardware. Keep the E-stop reachable and read the
prompt before confirming a live action.

## Hardware Checks

| Script | Purpose | Typical use |
| --- | --- | --- |
| `franka-tests/test_connection.py` | Read-only FCI connection check. | `python franka-tests/test_connection.py --robot_ip 172.16.0.2` |
| `franka-tests/diagnose_fr3_kinematics.py` | Read-only kinematic/state inspection. | `python franka-tests/diagnose_fr3_kinematics.py --help` |
| `franka-tests/check_touch_frame.py` | Compare a hand-guided physical touch with the configured frame. | `python franka-tests/check_touch_frame.py --robot_ip 172.16.0.2` |
| `franka-tests/check_cartesian_axes.py` | Small Cartesian direction commissioning motion. | `python franka-tests/check_cartesian_axes.py --help` |
| `franka-tests/probe_points.py` | Record workcell calibration points. | `python franka-tests/probe_points.py --robot_ip 172.16.0.2` |
| `franka-tests/preview_realsense.py` | Display RealSense camera streams. | `python franka-tests/preview_realsense.py --help` |
| `franka-tests/test_realsense_recording.py` | Camera-only recording test. | `python franka-tests/test_realsense_recording.py --help` |
| `franka-tests/test_grasp_cycle.py` | Grasp tuning cycle without basket transport. | `python franka-tests/test_grasp_cycle.py --help` |
| `franka-tests/pick_and_place.py` | Demonstration collection and scripted pick/place workflow. | `python franka-tests/pick_and_place.py --help` |
| `franka-tests/indicate_and_confirm.py` | Point above planned cells for operator cube placement. | `python franka-tests/indicate_and_confirm.py --help` |

`franka_motion.py`, `grid_utils.py`, and `franka-tests/realsense_recorder.py`
are shared modules imported by these tools, not standalone commands.

For the complete demonstration-collection procedure, see the
[FR3 Recording Guide](RECORDING_GUIDE.md).

## Model Validation

These tools do not command arm motion.

| Script | Purpose | Typical use |
| --- | --- | --- |
| `model-validation/test_fr3_policy_server.py` | Synthetic pi0.5-DROID policy-server smoke test. | `python model-validation/test_fr3_policy_server.py --host <server-ip>` |
| `model-validation/test_fr3_cartesian_policy_server.py` | Legacy Cartesian policy-server smoke test. | `python model-validation/test_fr3_cartesian_policy_server.py --help` |
| `model-validation/shadow_fr3_policy.py` | Live-camera, read-only policy inference. | `python model-validation/shadow_fr3_policy.py --help` |
| `model-validation/replay_fr3_joint_velocity_policy.py` | Offline replay of recorded joint-velocity observations. | `python model-validation/replay_fr3_joint_velocity_policy.py --help` |
| `model-validation/rank_fr3_joint_velocity_cells.py` | Rank recorded cells by replay agreement. | `python model-validation/rank_fr3_joint_velocity_cells.py --help` |
| `model-validation/validate_raw_recordings.py` | Check raw episode completeness before conversion. | `python model-validation/validate_raw_recordings.py --help` |
| `model-validation/validate_fr3_pi05_batch.py` | Validate LeRobot data and an OpenPI training batch. | `python model-validation/validate_fr3_pi05_batch.py --help` |
| `model-validation/audit_fr3_new_demos.py` | Detect held-out evaluation leakage in new demos. | `python model-validation/audit_fr3_new_demos.py --help` |
| `model-validation/compare_policy_image_domains.py` | Compare policy outputs for recorded and live image domains. | `python model-validation/compare_policy_image_domains.py --help` |

The `*_cartesian_*` replay/shadow tools are retained for the legacy Cartesian
policy and are not part of the current joint-velocity deployment path.

## Live Model Running

| Script | Purpose | Typical use |
| --- | --- | --- |
| `run_streaming_validation_trial.py` | One guarded end-to-end policy trial. | `python run_streaming_validation_trial.py --help` |
| `run_streaming_validation_batch.py` | Run multiple manifest positions sequentially. | `python run_streaming_validation_batch.py --help` |
| `run_fr3_policy_eval_trial.py` | Scripted setup, recovery, and turnover phases. | `python run_fr3_policy_eval_trial.py --help` |
| `manual_fr3_policy_grasp.py` | Stationary gripper close/open after streamer stop. | `python manual_fr3_policy_grasp.py --help` |
| `model-running/commission_recorded_cell104_approach.py` | Replay cell 104 approach to commission control timing. | `python model-running/commission_recorded_cell104_approach.py --help` |
| `model-running/commission_fr3_gripper_grasp.py` | Commission one stationary physical grasp. | `python model-running/commission_fr3_gripper_grasp.py --help` |
| `model-running/reset_to_recorded_policy_start.py` | Return to a recorded policy-start joint pose. | `python model-running/reset_to_recorded_policy_start.py --help` |
| `model-running/retreat_policy_test_home.py` | Guarded arm-only recovery to recorded home. | `python model-running/retreat_policy_test_home.py --help` |
| `model-running/record_fr3_eval_result.py` | Append or replace one evaluation ledger entry. | `python model-running/record_fr3_eval_result.py --help` |

`model-running/streaming/` contains the local UDP protocol, policy feeder,
warmup, recorded-action replay, and dry-run protocol test. Build the C++
streamer there once with CMake, then use `--help` on each Python launcher.

## Data And Training

| Area | Purpose | Entry point |
| --- | --- | --- |
| `data-pipeline/convert_raw_recordings_to_lerobot.py` | Convert raw RealSense episodes to LeRobot. | `python data-pipeline/convert_raw_recordings_to_lerobot.py --help` |
| `data-pipeline/stage_fr3_merged_recordings.py` | Build a symlinked merged raw-data set. | `python data-pipeline/stage_fr3_merged_recordings.py --help` |
| `data-pipeline/create_*` | Generate trackers and held-out/targeted manifests. | `python data-pipeline/<script>.py --help` |
| `data-pipeline/filter_*` | Export working cells from tracker status. | `python data-pipeline/<script>.py --help` |
| `data-pipeline/build_fr3_phase_sampling_manifest.py` | Create phase-aware training sample indices. | `python data-pipeline/build_fr3_phase_sampling_manifest.py --help` |
| `data-pipeline/compute_fr3_custom_norm_stats.py` | Compute OpenPI normalization statistics. | `python data-pipeline/compute_fr3_custom_norm_stats.py --help` |
| `training/prepare_fr3_v3_dataset_nibi_slurm.sh` | Audit, stage, and convert the v3 data set on Nibi. | `sbatch training/prepare_fr3_v3_dataset_nibi_slurm.sh` |
| `training/train_fr3_v3_droid_full_nibi_slurm.sh` | Train or resume the retained full pi0.5-DROID v3 run. | `sbatch training/train_fr3_v3_droid_full_nibi_slurm.sh` |
| `training/validate_fr3_pi05_batch_slurm.sh` | Validate the converted v3 dataset and an OpenPI batch. | `sbatch training/validate_fr3_pi05_batch_slurm.sh` |

Training helpers require an OpenPI checkout and are designed to operate on the
OpenPI environment, not the robot-control Conda environment.

## Setup And Configuration

| Path | Purpose |
| --- | --- |
| `setup/bootstrap_fr3_environment.sh` | Create/install the robot-control Python environment. |
| `setup/verify_fr3_setup.py` | Verify local imports, calibration, build tools, and optionally policy-server TCP access. |
| `configs/` | Workcell calibration, manifests, trackers, and environment specification. |
| `requirements/fr3-recording.txt` | Pip dependencies for the robot-control environment. |

Do not commit generated `eval_logs/`, `shadow_logs/`, camera recordings, model
checkpoints, or workstation-specific calibration replacements unless they are
intentionally part of a reproducible release.
