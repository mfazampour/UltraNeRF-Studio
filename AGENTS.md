# AGENTS.md

Guidance for contributors and coding agents working in this repository.

## Purpose

This is a research-oriented ultrasound NeRF repository with multiple experimental scripts, partial workflows, and dataset-specific assumptions. Treat it as an evolving codebase rather than a polished library.

The goal when making changes is to improve clarity, correctness, and reproducibility without breaking the main training and rendering paths.

## Working Principles

- Prefer understanding the active code path before editing.
- Preserve existing experiment behavior unless the task explicitly asks for behavioral changes.
- Make small, targeted changes rather than broad refactors.
- Keep documentation aligned with code changes.
- Do not treat experimental scripts as fully interchangeable; several top-level scripts are similar but not identical.

## Repo Priorities

When in doubt, prioritize these paths:

1. `run_ultranerf.py`
2. `unerf_config.py`
3. `load_us.py`
4. `nerf_utils.py`
5. `model.py`
6. `rendering.py`

These files define the main ultrasound rendering and training workflow.

Treat these as more experimental or secondary:

- `run_ultranerf_reconstruction.py`
- `reconstruction_network.py`
- `run_reconstruction.py`
- `run_reconstruction_from_pts.py`
- `occupancy_network.py`
- ad hoc scripts in `scripts/`

## Before Making Changes

- Read the relevant entry script first.
- Trace how configuration is passed from `configargparse` into model creation and rendering.
- Confirm whether the code path is part of the baseline workflow or a reconstruction / experiment branch.
- Check whether a script is intended for training, visualization, or one-off preprocessing.

## Coding Best Practices

- Keep changes local to the task.
- Reuse existing helpers before adding new abstractions.
- Prefer explicit names over clever shortcuts.
- Preserve current file and CLI conventions unless there is a strong reason to change them.
- Avoid adding heavy dependencies unless they are necessary for the requested task.
- If adding a new dependency, update documentation accordingly.
- Keep comments short and useful; explain non-obvious behavior, not trivial assignments.

## Repo-Specific Expectations

- The renderer is ultrasound-specific. Do not assume standard NeRF semantics.
- Keep the repository PyTorch-only. Do not introduce TensorFlow or dual-framework maintenance paths.
- The active rendering path is wired through `render_us()` and `render_rays_us()`.
- Baseline datasets are expected to use `.npy` arrays with filenames such as `images.npy` and `poses.npy`.
- Reconstruction workflows may additionally expect `labels.npy` and `poses_labels.npy`.
- Pose translations are scaled from millimeters to meters in the loader. Be careful not to apply scaling twice.
- Probe geometry is derived from `probe_depth` and `probe_width`; keep units consistent.

## Validation

For code changes, validate at the smallest useful level:

- syntax check if the change is small
- targeted script execution if the workflow is clear
- config parsing when changing CLI options
- a short training or render smoke test when changing core training or rendering logic

If full validation is not practical, state what was not verified.

## Documentation

Update `README.md` when changing:

- major workflows
- dataset requirements
- entry points
- dependencies
- expected commands for training or rendering

If a new script is added, document what it is for and whether it is baseline, experimental, or utility-only.

## Git and Branching

- Work on a dedicated branch for non-trivial changes.
- Keep commits focused and logically grouped.
- Commit after each coherent set of changes you make, using a message that describes the actual outcome.
- Do not rewrite unrelated user changes.
- Do not delete logs, datasets, or generated artifacts unless explicitly asked.

## Safety Rules

- Do not overwrite dataset files in `data/` unless the task explicitly requires it.
- Do not assume checkpoints are disposable.
- Be careful with scripts that write to `logs/`, export meshes, or generate point clouds.
- Avoid destructive cleanup of outputs unless requested.
- Any file or directory created in the workspace should be left writable without root access. If a task creates files as root-owned artifacts, adjust permissions before finishing.

## When Touching Experimental Code

If changing reconstruction or occupancy code:

- note that these paths are less standardized than the baseline renderer
- verify tensor shapes carefully
- verify whether inputs are points, image-aligned features, or NeRF-derived parameters
- document any assumptions introduced by the change

## Preferred Change Style

A good change in this repository usually has these properties:

- minimal surface area
- clear reason for the modification
- preserved CLI behavior unless intentionally changed
- updated docs when user-facing behavior changes
- at least basic validation or a clear statement of what remains unverified
