# T05 - Add a Basic 3D Volume Viewer

## Goal

Build the first interactive viewer for displaying the fused sweep volume in 3D
with opacity and color controls.

## Why This Matters

This is the first visible milestone for the branch. It turns the backend volume
into something a user can inspect and validates that the sweep fusion pipeline is
producing useful spatial structure.

## Required Work

1. Choose the viewer stack.
   For the intended workflow, napari is acceptable for the first version if the
   viewer is kept modular.

2. Load the cached or freshly fused volume from T03/T04.

3. Display the scalar volume in 3D.

4. Add basic rendering controls:
   - opacity
   - colormap
   - display mode
   - optional threshold

5. Ensure the volume is displayed in the correct world orientation.

6. Keep the viewer code separate from NeRF inference logic.

## Suggested Implementation

- A viewer module under `visualization/volume_viewer.py`
- One launchable script or function that opens the volume and exposes the basic
  controls

## What Needs To Be Checked

- The displayed volume is not mirrored or rotated incorrectly
- Users can inspect the sweep coverage in 3D
- The viewer can load cached volumes from disk
- The rendering remains responsive at typical volume sizes

## Output of This Ticket

- A working 3D volume rendering of the input sweep data
- The first interactive artifact a user can explore

## Acceptance Criteria

- A developer can launch the viewer and inspect a fused sweep volume
- Basic transfer-function-like controls are available
- Volume loading and display require no code edits

## Dependencies

- T03
- T04

## Blocks

- T06
- T07
- T13
- T14
- T15
