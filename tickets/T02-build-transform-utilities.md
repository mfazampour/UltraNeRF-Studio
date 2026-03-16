# T02 - Build Transform Utilities

## Goal

Implement a reusable transformation module that converts among image-space,
probe-local, world, and voxel coordinates according to the conventions agreed in
T01.

## Why This Matters

Today, the codebase performs coordinate handling implicitly in multiple places.
That is manageable for research scripts, but not for an interactive
visualization application. The visualization stack needs a single authoritative
set of transformation utilities.

## Required Work

1. Create a new module, likely under `visualization/` or `visualization_core/`.
   Suggested name:
   - `visualization/transforms.py`

2. Add data structures or helper functions for:
   - image pixel to probe-local point
   - probe-local point to world point
   - world point to voxel index
   - voxel index to world point
   - pose composition / inversion for visualization use

3. Implement helpers for probe geometry:
   - lateral sample positions along the probe
   - depth sample positions along the beam
   - scan plane corner coordinates in world space

4. Keep the API independent from GUI code.
   The transform utilities should not depend on napari, Qt, or any rendering
   framework.

5. Define the expected input and output types.
   Prefer explicit types and shape conventions, for example:
   - `numpy.ndarray` for batch processing
   - `torch.Tensor` only where direct integration with NeRF inference matters

6. Add conversion utilities between homogeneous and non-homogeneous coordinates
   if needed.

## Recommended API Surface

- `pixel_to_probe_local(...)`
- `probe_local_to_world(...)`
- `world_to_probe_local(...)`
- `world_to_voxel(...)`
- `voxel_to_world(...)`
- `probe_plane_corners(...)`
- `pose_to_axes(...)`

## What Needs To Be Checked

- Transforms are numerically consistent and invertible where expected
- Returned coordinates match the world frame defined in T01
- Lateral / depth axes are not swapped
- The helper outputs align with `get_rays_us_linear()` semantics

## Output of This Ticket

- A reusable transform module
- Unit tests using simple synthetic poses and coordinates
- A reliable foundation for sweep fusion and probe interaction

## Acceptance Criteria

- A known pixel and pose can be converted to the correct world-space point
- A world-space point can be mapped into a voxel grid and back with known error
- All helper functions are covered by deterministic tests

## Dependencies

- T01

## Blocks

- T03
- T05
- T07
- T10
