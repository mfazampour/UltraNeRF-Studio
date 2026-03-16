# UltraNeRF

PyTorch research code for learning an ultrasound image formation model from tracked 2D ultrasound frames.

This repository starts from UltraNeRF and NeRF-style MLPs, but the core renderer is customized for linear ultrasound probes rather than RGB image synthesis.

Original upstream references:
- UltraNeRF: https://github.com/magdalena-wysocki/ultra-nerf
- NeRF PyTorch baseline: https://github.com/yenchenlin/nerf-pytorch

## What this repo does

At a high level, the code learns a 3D field from ultrasound data:

1. Load tracked 2D ultrasound images and probe poses.
2. Cast linear ultrasound rays from the probe geometry.
3. Sample 3D points along each ray.
4. Use an MLP to predict per-point ultrasound parameters.
5. Integrate those parameters with a custom ultrasound renderer.
6. Compare the rendered image to the measured ultrasound frame.

This is not a standard NeRF that predicts RGB and density. The active training path predicts ultrasound-oriented quantities such as attenuation, reflection, and scattering amplitude, then renders a grayscale ultrasound-like image from them.

## Repository structure

The project is script-driven. There is no package layout or single library entry point; most workflows are top-level Python scripts.

### Main training and evaluation scripts

- `run_ultranerf.py`
  Main training loop for the baseline ultrasound NeRF model.

- `run_barf.py`
  Training loop with BARF-style pose refinement. This jointly optimizes the scene model and pose corrections.

- `run_ultranerf_reconstruction.py`
  Joint or staged training for ultrasound rendering plus reconstruction labels.

- `reconstruction_network.py`
  Trains a separate reconstruction network from NeRF-derived features.

- `render_demo_us.py`
  Loads a trained checkpoint and exports rendered outputs and point clouds.

- `run_reconstruction.py`
  Postprocessing utility for building a point cloud / mesh style reconstruction from model outputs.

- `run_reconstruction_from_pts.py`
  Similar reconstruction utility, but starts from sampled points rather than image-space inference.

### Core modules

- `load_us.py`
  Dataset loading and pose preprocessing.

- `model.py`
  Defines the MLPs:
  - `NeRF`: baseline ultrasound field model
  - `BARF`: NeRF with coarse-to-fine positional encoding for pose refinement
  - `Reconstruction`: occupancy / segmentation style head
  - `PoseRefine`: learned SE(3) pose offsets

- `nerf_utils.py`
  Model creation, positional encoding, ray batching, rendering dispatch, and loss helpers.

- `rendering.py`
  Ultrasound-specific forward model. This is the most important file for understanding how predicted 3D quantities become a 2D ultrasound image.

- `camera.py`
  Pose and Lie algebra utilities used by BARF pose refinement.

### Utilities and experiments

- `rendering_utils/`
  Helper functions for reflection modeling and wavelet-based denoising / backscatter extraction.

- `occupancy_network.py`
  Standalone occupancy-grid experiment. This is not tightly integrated with the main training path.

- `scripts/`
  Small preprocessing helpers.

- `slurm/`
  Batch job scripts for cluster runs.

- `tests/`
  Lightweight comparison / experiment code rather than a full regression test suite.

## Core pipeline

### 1. Data loading

`load_us.py` loads arrays from disk and normalizes them for training.

Expected baseline files in a dataset directory:

- `images.npy`
- `poses.npy`

Optional files used by specialized workflows:

- `confidence_maps.npy`
- `labels.npy`
- `poses_labels.npy`

Important details:

- Images are converted to `float32` and scaled to `[0, 1]`.
- Pose translations are scaled from millimeters to meters.
- A holdout view is selected automatically as the pose closest to the average pose.

The bundled sample dataset under `data/synthetic_testing/l2` includes baseline image / pose data. Reconstruction workflows require additional label arrays that are not present in that sample directory.

### 2. Ultrasound ray generation

The repo uses a linear probe model, not a pinhole camera model.

`get_rays_us_linear()` in `nerf_utils.py` creates:

- one ray origin per lateral probe position
- one shared forward ray direction
- a near/far interval based on probe depth

This matches the acquisition geometry of a linear ultrasound transducer.

### 3. MLP prediction

Points sampled along each ray are fed through positional encoding and then into an MLP.

The main models are:

- `NeRF`
  Baseline field predictor.

- `BARF`
  Adds BARF-style coarse-to-fine positional encoding and is paired with `PoseRefine`.

- `Reconstruction`
  Predicts a binary or occupancy-like reconstruction signal from features or points.

### 4. Ultrasound rendering

`rendering.py` contains several rendering variants. The active path used by the main renderer is `render_method_3()`.

Conceptually it does the following:

- interprets raw network outputs as ultrasound parameters
- accumulates attenuation along depth
- computes reflection / transmission effects
- models backscattering with a Gaussian point spread function
- combines these terms into a final intensity map
- also exposes intermediate maps such as confidence, attenuation, reflection, and scatter amplitude

The main rendering entry points are:

- `render_us()` in `nerf_utils.py`
- `render_rays_us()` in `rendering.py`

## Main workflows

### Baseline ultrasound NeRF

Train the standard model:

```bash
python run_ultranerf.py --config configs/config_base_nerf.txt --expname my_run
```

What it does:

- loads ultrasound frames and poses
- samples one training image per iteration
- renders the full frame from the current model
- optimizes image-space loss, optionally with regularization
- writes checkpoints to `logs/<expname>/`

Configuration is parsed from `unerf_config.py`, with example configs in `configs/`.

### BARF pose refinement

Train with learned pose corrections:

```bash
python run_barf.py --config configs/config_barf.txt
```

Additional behavior compared with baseline training:

- creates a `PoseRefine` module with one learned SE(3) offset per pose
- updates both the scene model and the pose parameters
- uses a coarse-to-fine positional encoding schedule controlled by BARF parameters such as `L` and `c2f`

### Reconstruction experiments

There are two related reconstruction paths:

```bash
python run_ultranerf_reconstruction.py --config <config>
python reconstruction_network.py --config <config>
```

These workflows use:

- `labels.npy` as a reconstruction target
- `poses_labels.npy` as precomputed 3D point samples
- a second MLP (`Reconstruction`) trained on points and/or NeRF-derived features

In practice, these scripts are more experimental than the baseline renderer and have more branching logic.

### Offline rendering and export

After training, use:

```bash
python render_demo_us.py --logdir logs --expname <run_name> --model_epoch <step>
```

This loads a saved checkpoint, renders outputs for stored poses, and can export intermediate maps and point cloud style reconstructions.

## Configuration files

Example configs live in `configs/`. Common ones include:

- `config_base_nerf.txt`
  Baseline ultrasound NeRF.

- `config_barf.txt`
  BARF pose refinement.

- `config_confmap.txt`
  Confidence-map training mode.

- `config_liver.txt`, `config_patient13.txt`, `config_patient21.txt`, etc.
  Dataset-specific experiment presets.

Common parameters:

- `datadir`
  Dataset directory.

- `probe_depth`, `probe_width`
  Physical probe dimensions used to derive ray spacing.

- `N_samples`
  Number of depth samples per ray.

- `multires`
  Positional encoding frequency depth.

- `lrate`, `lrate_decay`
  Learning rate and exponential decay schedule.

- `reg`, `r_tv_penalty`, `r_lcc_penalty`
  Optional image regularization terms.

- `reconstruction`, `confidence`, `rec_only_theta`, `rec_only_occ`
  Experimental reconstruction controls.

## Data format notes

From the code, the baseline dataset contract is:

- `images.npy`: shape roughly `[N, H, W]`
- `poses.npy`: one 4x4 pose per image

For reconstruction experiments:

- `labels.npy`: binary or grayscale reconstruction labels
- `poses_labels.npy`: 3D point locations associated with image pixels or reconstruction queries

The loader assumes numpy arrays and does not build a generic dataset abstraction. If you bring in new data, matching these filenames and conventions is the easiest path.

## Dependencies

The main dependencies listed in `requirements.txt` are:

- PyTorch
- torchvision
- MONAI
- PyWavelets
- ptwt
- matplotlib
- configargparse
- tensorboard
- tqdm
- opencv-python

Some utility scripts also import extra packages that are not listed in `requirements.txt`, including:

- `open3d`
- `mcubes`
- `imageio`
- `tensorflow` for a comparison script in `tests/`
- `trimesh`

If you plan to use the reconstruction and export scripts, expect to install additional packages beyond the base requirements.

## Caveats and current state

This is research code and it shows. A few important points:

- The main code path is the baseline renderer in `run_ultranerf.py`.
- Several scripts duplicate logic rather than sharing common abstractions.
- Multiple rendering variants exist in `rendering.py`, but the active training path is hardwired to `render_method_3()`.
- The common config value `output_ch = 5` is broader than what the active renderer actually consumes in its default path.
- Some scripts contain commented-out branches and experimental code paths.
- Test coverage is minimal.

That said, the core training loop, pose-refinement loop, and visualization / export scripts are understandable once you think of the project as:

`tracked ultrasound frames + probe poses -> 3D field MLP -> ultrasound renderer -> image loss`

## Recommended reading order

If you are new to the codebase, read files in this order:

1. `run_ultranerf.py`
2. `unerf_config.py`
3. `load_us.py`
4. `nerf_utils.py`
5. `model.py`
6. `rendering.py`
7. `run_barf.py`
8. `run_ultranerf_reconstruction.py`

That sequence matches how the main workflow is assembled.
