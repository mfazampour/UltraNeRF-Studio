# T10 - Wrap NeRF Inference for Arbitrary Poses

## Goal

Create a reusable runtime wrapper that loads a trained checkpoint and renders
ultrasound outputs for any probe pose chosen in the viewer.

## Why This Matters

The current repository can render from a known pose during training or from
offline scripts, but it does not expose a clean interactive API for “render this
arbitrary user-selected probe pose now”.

## Required Work

1. Create a dedicated inference session module.
   Suggested file:
   - `visualization/nerf_session.py`

2. Implement:
   - checkpoint loading
   - config loading
   - model instantiation
   - device selection
   - pose input handling
   - image rendering from arbitrary `c2w` / probe pose

3. Expose a simple API, for example:
   - `load_session(checkpoint, config)`
   - `render_pose(pose)`

4. Support returning:
   - intensity map
   - confidence map
   - reflection map
   - attenuation map
   - other intermediate outputs if requested

5. Keep the model loaded in memory for repeated rendering.

6. Make sure the wrapper uses the same spatial conventions defined in T01.

## Performance Considerations

- Avoid reloading the model on every render
- Make rendering chunk size configurable
- Keep UI and inference loosely coupled so rendering can later be run in a worker

## What Needs To Be Checked

- Arbitrary viewer-selected poses are accepted without shape or type mismatch
- The rendered image orientation matches the scan plane representation
- Inference uses the same probe width/depth assumptions as the volume viewer

## Output of This Ticket

- A reusable PyTorch inference wrapper for interactive rendering
- A stable backend API for the viewer

## Acceptance Criteria

- A caller can render a valid ultrasound image from a manually supplied probe
  pose without using the training scripts
- Intermediate rendering outputs can be requested and displayed

## Dependencies

- T01

## Blocks

- T11
- T12
- T15
