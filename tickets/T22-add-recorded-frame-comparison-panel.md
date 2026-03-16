# T22 - Add a Recorded-Frame Comparison Panel

## Goal

Expose a side-by-side comparison between the NeRF-rendered output for the
current virtual probe pose and the nearest recorded real frame from the sweep.

## Why This Matters

Interactive rendering is much easier to trust and debug when the user can see:

- what the model predicts
- which recorded frame is considered the closest spatial match
- how far apart those poses are

The backend matching logic already exists, but it is not surfaced in the app.

## Required Work

- Add a dedicated UI element for the nearest recorded frame.
- Decide how comparison information should be displayed:
  - rendered image panel + recorded frame panel
  - tabbed display
  - side-by-side dock widgets
- Display matching metadata, at minimum:
  - matched frame index
  - translation distance in mm
  - rotation distance in degrees
- Refresh the comparison panel whenever the active probe pose changes or a new
  render is requested.
- Ensure comparison still works when the pose is not exactly on the recorded
  trajectory.
- Keep the comparison path optional when NeRF mode is not enabled.

## What Needs To Be Checked

- The nearest frame chosen by the viewer is plausible for several probe poses.
- The frame index and distance values update when the probe moves.
- The recorded image orientation is consistent with the rendered image.
- Comparison state remains correct after repeated probe edits and renders.

## Output of This Ticket

- A visible nearest-frame comparison panel in the app.
- Pose-distance metadata shown to the user.
- Tests covering comparison refresh logic and payload display adapters.

## Acceptance Criteria

- A user can see both:
  - the NeRF-rendered image
  - the closest recorded frame
- The panel reports which frame was matched and by how much pose distance.
- The app remains stable when switching between sweep-only and sweep + NeRF
  modes.

## Dependencies

- T12
- T19
- T21

## Blocks

- T23
