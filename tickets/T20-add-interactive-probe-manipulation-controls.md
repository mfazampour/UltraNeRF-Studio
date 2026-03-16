# T20 - Add Interactive Probe Manipulation Controls

## Goal

Expose user controls that let the virtual probe be repositioned and reoriented
inside the viewer, rather than only initializing it from a recorded sweep pose.

## Why This Matters

The point of the interactive viewer is not just to inspect the recorded sweep.
It is to let a user place a probe at an arbitrary pose in the scene and see the
corresponding NeRF output. That requires controls for both translation and
orientation.

The current code already has backend math for:

- probe placement from MPR-derived positions
- pose orientation updates
- probe geometry overlays

What is missing is the user-facing control surface.

## Required Work

- Choose an initial control strategy that is practical in napari:
  - dock widget with numeric fields and buttons
  - sliders for translation and rotation
  - selection from recorded trajectory + incremental adjustments
- Support at least:
  - probe center translation in millimeters
  - yaw
  - pitch
  - roll
- Make the current probe pose visible in the UI so the user can understand the
  active state.
- Update the 3D probe overlay immediately when controls change.
- Keep controls and internal scene state synchronized in both directions where
  practical.
- Avoid implementing a fragile custom 3D drag tool as the first version unless
  napari interaction proves stable enough.

## Recommended First Version

- Provide a dock widget with:
  - current recorded pose index
  - translation fields: `x_mm`, `y_mm`, `z_mm`
  - rotation fields: `yaw_deg`, `pitch_deg`, `roll_deg`
  - a button to snap to the nearest recorded pose
  - a button to reset to a selected recorded frame
- Keep this state backed by the existing visualization controller rather than
  duplicating transforms in the UI layer.

## What Needs To Be Checked

- Editing translation moves the probe overlay to the expected location.
- Editing orientation rotates the scan plane and beam line correctly.
- Repeated updates do not introduce numerical drift or broken poses.
- The UI remains responsive during pose edits.

## Output of This Ticket

- Practical probe manipulation controls in the viewer.
- Tests for pose-update callbacks and state synchronization.

## Acceptance Criteria

- A user can move the probe away from the recorded trajectory.
- A user can rotate the probe and see the updated scan plane.
- The UI controls and 3D overlay remain consistent after repeated edits.

## Dependencies

- T06
- T08
- T09
- T15

## Blocks

- T21
- T22
