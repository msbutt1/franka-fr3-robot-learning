# FR3 Real-Robot Workspace

This folder is organized by purpose:

- `robot/`: scripts that connect to or command the FR3.
- `camera/`: RealSense preview and recording checks.
- `grid/`: calibration-grid and tracker generation/filtering tools.
- `datasets/`: converters from raw episodes into training datasets.
- `training/`: OpenPI pi0.5-DROID conversion, config patching, and full fine-tune jobs.
- `common/`: shared motion, grid, and camera helpers imported by scripts.
- `config/`: environment files.
- `data/calibration/`: probed robot/table points.
- `data/cells/`: runnable cell JSON exports.
- `data/trackers/`: tracker spreadsheets.
- `data/state/`: small runtime state files, such as the next cell to record.
- `jobs/`: cluster and batch-job entry points.

Most scripts can be run directly from the repo root. Defaults point at the
checked-in files under `fr3_real/data`, so common commands do not need extra
path flags:

```bash
python fr3_real/robot/test_connection.py --robot_ip 172.16.0.2
python fr3_real/robot/probe_points.py --robot_ip 172.16.0.2
python fr3_real/grid/create_grid_tracker.py
python fr3_real/grid/filter_grid_tracker.py
python fr3_real/robot/pick_and_place.py --robot_ip 172.16.0.2 --cells_json fr3_real/data/cells/working_cells.json
```

Runtime recordings are still written under `fr3_real/recordings/` when selected,
and raw recording dumps outside this package live in `fr3_raw_recordings/`.
