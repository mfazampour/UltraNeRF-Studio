# Scripts

This folder contains one-off preprocessing and conversion utilities used during
dataset preparation and experimentation.

Examples:

- `save_images_to_numpy.py`
  Convert image folders into `.npy` arrays expected by the loaders.

- `compute_points_from_poses.py`
  Generate point samples from stored poses.

- `process_for_occ.py`
  Prepare data for occupancy-style experiments.

- `safe_param_img.py`
  Save parameter images or derived outputs for inspection.

These scripts are support tools rather than part of the core training loop.
