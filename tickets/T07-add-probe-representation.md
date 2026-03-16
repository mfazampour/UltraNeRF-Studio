# T07 - Add Probe Representation

## Goal

Represent the ultrasound probe and its scan plane visually inside the 3D
visualization scene.

## Why This Matters

Users need to see where the probe is, what direction it is facing, and what
plane it is sampling. Without a clear probe representation, interactive placement
and NeRF rendering will be difficult to interpret.

## Required Work

1. Define a simple probe geometry.
   This does not need to be photorealistic. A practical first version can show:
   - probe center
   - local axes
   - image plane rectangle
   - beam direction

2. Implement a function that derives probe visualization primitives from a 4x4
   world pose and probe geometry parameters.

3. Show the current scan plane in the 3D view.

4. Make the representation updateable so later tickets can move and rotate it.

5. Consider showing:
   - current plane corners
   - local x/y/z axes
   - optional label or ID

## What Needs To Be Checked

- The scan plane orientation matches the actual rendering frame
- The beam direction matches the direction used by `get_rays_us_linear()`
- The probe dimensions correspond to `probe_width` and `probe_depth`

## Output of This Ticket

- A reusable visual probe model
- A 3D overlay that later tickets can manipulate

## Acceptance Criteria

- A given world pose produces a visually correct probe and scan plane
- The representation can be updated in real time without recreating the app

## Dependencies

- T01
- T02
- T05

## Blocks

- T08
- T09
- T11
- T13
