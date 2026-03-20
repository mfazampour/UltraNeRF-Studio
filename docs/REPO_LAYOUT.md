# Repository Layout

This repository is script-driven. The top-level Python files are CLI entry
points. Reusable library code lives under `src/ultranerf/`.

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

- `src/ultranerf/model.py`
  Main PyTorch model definitions.

- `src/ultranerf/nerf_utils.py`
  Model creation, embedding, rendering dispatch, and loss helpers.

- `src/ultranerf/rendering.py`
  Ultrasound-specific rendering logic.

- `src/ultranerf/load_us.py`
  Dataset loading and pose preprocessing.

- `src/ultranerf/unerf_config.py`
  Shared config parser for the main training paths.

- `src/ultranerf/camera.py`
  Pose and Lie algebra utilities.

## Supporting Directories

- `configs/`
  Experiment config presets.

- `data/`
  Datasets and sample inputs.

- `src/ultranerf/rendering_utils/`
  Reflection and denoising helpers used by the renderer.

- `src/ultranerf/visualization/`
  Visualization and napari UI modules.

- `scripts/`
  Small preprocessing or conversion utilities.

- `slurm/`
  Cluster job helpers.

- `tests/`
  PyTorch-only unit tests and regression checks.

## Recommended Reading Order

1. `README.md`
2. `run_ultranerf.py`
3. `src/ultranerf/unerf_config.py`
4. `src/ultranerf/load_us.py`
5. `src/ultranerf/nerf_utils.py`
6. `src/ultranerf/model.py`
7. `src/ultranerf/rendering.py`
8. `run_barf.py`
9. `run_ultranerf_reconstruction.py`
