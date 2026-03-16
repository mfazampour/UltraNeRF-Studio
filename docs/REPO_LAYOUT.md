# Repository Layout

This repository is script-driven. The top-level Python files are not all equal in
importance, so use this map when deciding where to start.

## Main Entry Points

- `run_ultranerf.py`
  Baseline PyTorch ultrasound NeRF training.

- `run_barf.py`
  Pose-refinement training with BARF-style coarse-to-fine encoding.

- `run_ultranerf_reconstruction.py`
  Reconstruction-aware training with image and label supervision.

- `run_reconstruction.py`
  Offline reconstruction export from trained checkpoints.

- `run_reconstruction_from_pts.py`
  Offline dense point / occupancy querying and geometry export.

- `run_noisy_barfs.py`
  Batch launcher for noisy-pose experiments.

## Core Implementation Files

- `model.py`
  Main PyTorch model definitions.

- `nerf_utils.py`
  Model creation, embedding, rendering dispatch, and loss helpers.

- `rendering.py`
  Ultrasound-specific rendering logic.

- `load_us.py`
  Dataset loading and pose preprocessing.

- `unerf_config.py`
  Shared config parser for the main training paths.

- `camera.py`
  Pose and Lie algebra utilities.

## Supporting Directories

- `configs/`
  Experiment config presets.

- `data/`
  Datasets and sample inputs.

- `rendering_utils/`
  Reflection and denoising helpers used by the renderer.

- `scripts/`
  Small preprocessing or conversion utilities.

- `slurm/`
  Cluster job helpers.

- `tests/`
  PyTorch-only unit tests and regression checks.

## Recommended Reading Order

1. `README.md`
2. `run_ultranerf.py`
3. `unerf_config.py`
4. `load_us.py`
5. `nerf_utils.py`
6. `model.py`
7. `rendering.py`
8. `run_barf.py`
9. `run_ultranerf_reconstruction.py`
