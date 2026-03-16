# T01 - Define Coordinate Conventions

## Goal

Create a clear and unambiguous definition of all coordinate systems used by the
visualization stack so that sweep fusion, probe placement, MPR interaction, and
NeRF rendering all operate in the same spatial frame.

This ticket is foundational. Most of the later work should not start until this
is written down and agreed on.

## Why This Matters

The current repository mixes several implicit spaces:

- image pixel coordinates
- probe-local sampling coordinates
- world coordinates from tracked poses
- voxel coordinates for volumetric fusion
- meters vs millimeters

Without a single documented convention, the viewer can look correct while still
being spatially wrong, especially when linking sweep volume display to arbitrary
NeRF probe rendering.

## Required Work

1. Define the canonical world-space unit.
   The recommended choice for the visualization stack is millimeters, because:
   - the scenes are spatially small
   - medical imaging viewers typically operate in millimeters
   - user-facing probe placement and volume inspection are easier to reason
     about in millimeters than in very small meter-scale coordinates

   This ticket should explicitly document the boundary between:
   - the current training / rendering code paths, which already scale pose
     translation from millimeters to meters in the loader
   - the visualization layer, which should expose spatial quantities in
     millimeters

   If the NeRF runtime continues to expect meter-scale poses internally, this
   ticket should define a single, explicit conversion point between the viewer
   state and the model inference state.

2. Define the canonical world-space axes.
   This should specify:
   - which axis is lateral in probe-local space
   - which axis is depth along the ultrasound beam
   - which axis is elevation / out of plane

3. Define image-space indexing conventions.
   Clarify:
   - whether image arrays are interpreted as `[H, W]` or `[depth, lateral]`
   - whether origin is top-left or center-based
   - how lateral and depth map onto probe-local coordinates

4. Define the probe-local frame.
   Clarify:
   - where the probe origin is located
   - whether the origin is at the center of the array or at an image corner
   - which axis corresponds to beam direction
   - how `probe_width` and `probe_depth` map to local coordinates

5. Define the pose transform meaning.
   Document what `poses.npy` represents:
   - probe-to-world
   - world-to-probe
   - camera-to-world equivalent
   - whether the code expects `c2w` semantics in the rendering path

6. Define the voxel grid frame for fused sweep volumes.
   Clarify:
   - whether the voxel volume uses world coordinates directly
   - how origin and spacing are stored
   - how world points map into voxel indices

7. Define the relationship between the virtual interactive probe and the tracked probe.
   State whether the virtual probe pose is always expressed in the same world
   frame as `poses.npy`.

## Recommended Deliverables

- A markdown design note under `docs/`, for example `docs/COORDINATE_SYSTEMS.md`
- A short diagram showing image, probe-local, and world frames
- One or two concrete examples using a small pose and a known pixel location

## What Needs To Be Checked

- The coordinate description matches the actual ray-generation behavior in
  `get_rays_us_linear()`
- The documented pose interpretation matches how `render_us()` consumes poses
- The chosen image origin and axis ordering are consistent across:
  - sweep fusion
  - MPR placement
  - live NeRF rendering
- Unit consistency is explicit everywhere

## Output of This Ticket

- A written, reviewed coordinate-system specification
- A fixed decision on spatial units and axis directions
- Reduced ambiguity for all downstream tickets

## Acceptance Criteria

- A developer unfamiliar with the code can derive a world-space point from an
  image pixel and a pose by following the document alone
- No open ambiguity remains about unit scale or probe orientation

## Dependencies

- None

## Blocks

- T02 through T11
