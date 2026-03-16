# T18 - Add NeRF Checkpoint Loading to the Visualization CLI

## Goal

Extend the visualization launch path so the app can load a trained UltraNeRF
checkpoint and prepare an inference session directly from the command line.

This ticket is about application wiring and runtime configuration. It is not yet
about displaying the rendered image in the GUI.

## Why This Matters

The current sweep viewer can display the fused input volume, the trajectory, and
the virtual probe overlay, but it has no way to attach a trained NeRF model to
the session. Without this step, the interactive visualization remains a static
scene explorer.

The repo already has backend pieces for arbitrary-pose inference in
`visualization/nerf_session.py`, but there is no user-facing way to initialize
them from the visualization entry point.

## Required Work

- Add CLI arguments to `run_visualize_sweeps.py` for:
  - checkpoint path
  - optional config path
  - optional render resolution overrides
  - optional render trigger mode
- Validate that the provided checkpoint path exists before launching the GUI.
- Reuse the existing `visualization.nerf_session.NerfSession` wrapper rather
  than duplicating model-loading logic.
- Decide and document how viewer-space millimeter poses are converted into the
  meter-based runtime expected by the current renderer.
- Instantiate a `RenderController` when a checkpoint is provided.
- Keep the visualization app usable without a checkpoint. The NeRF path should
  be optional, not mandatory.
- Ensure the launch flow fails with a clear error message when:
  - the checkpoint is missing
  - the config is incompatible
  - model loading fails
- Surface a concise runtime summary to the user so they can confirm:
  - whether NeRF is enabled
  - which checkpoint was loaded
  - which render trigger mode is active

## Suggested Implementation

- Add a small configuration dataclass in the visualization app layer that holds:
  - checkpoint path
  - config path
  - trigger mode
  - render kwargs
- Thread that dataclass from the CLI into `visualization.app.launch_visualization_app`.
- Keep `NerfSession.from_checkpoint(...)` as the only place that directly
  touches the training/inference runtime.
- Avoid importing heavy NeRF runtime code unless the user actually requested a
  checkpoint-backed session.

## What Needs To Be Checked

- Launch without a checkpoint still works exactly as before.
- Launch with a valid checkpoint creates a render controller.
- Viewer pose values still remain in millimeters, while inference receives the
  correct meter-scaled pose.
- Error handling is readable and points to the misconfigured input.

## Output of This Ticket

- A checkpoint-aware visualization CLI.
- A render controller attached to the viewer session when requested.
- Tests covering argument parsing and session construction with fake NeRF
  runtime objects.

## Acceptance Criteria

- The visualization CLI accepts a checkpoint path and starts successfully when a
  valid model is available.
- The session can run in both:
  - sweep-only mode
  - sweep + NeRF mode
- Automated tests cover the checkpoint-enabled launch path.

## Dependencies

- T10
- T15

## Blocks

- T19
- T21
- T22
