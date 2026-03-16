# T09 - Add Probe Orientation Controls

## Goal

Allow the user to adjust the probe image plane direction and beam orientation
after placing the probe.

## Why This Matters

Placing only the probe center is not enough. The virtual probe must be rotatable
so the user can choose the scanning plane that will be sent to the NeRF
renderer.

## Required Work

1. Decide on the interaction mechanism.
   Options include:
   - rotation sliders
   - Euler-angle spin boxes
   - axis-angle controls
   - vector controls for plane normal and in-plane direction

2. Update the probe pose in world coordinates from these controls.

3. Update:
   - probe axes
   - scan plane rectangle
   - optional beam direction visualization

4. Preserve orthonormality of the probe frame.
   The controls should not create invalid or skewed transforms.

5. Expose the updated 4x4 pose to the NeRF rendering backend.

## Recommended First Version

- Numeric controls for:
  - yaw
  - pitch
  - roll
- Optional reset-to-default orientation button

## What Needs To Be Checked

- Rotation updates change the displayed scan plane correctly
- The resulting pose remains valid and right-handed
- Orientation controls correspond to the documented probe-local frame

## Output of This Ticket

- A manipulable virtual probe orientation
- A pose object ready for arbitrary-pose NeRF rendering

## Acceptance Criteria

- A user can rotate the probe plane and see the result immediately
- The updated pose is available in the application state as a valid transform

## Dependencies

- T07

## Blocks

- T11
