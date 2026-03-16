# T11 - Connect Viewer Pose to NeRF Rendering

## Goal

Link the interactive probe pose from the visualization app to the arbitrary-pose
NeRF inference wrapper so that moving the probe updates the rendered ultrasound
output.

## Why This Matters

This is the central interactive feature requested by the user: place or rotate a
virtual probe in 3D and immediately visualize the NeRF output for that pose.

## Required Work

1. Connect the application state to the NeRF session from T10.

2. Trigger rendering when:
   - the probe center changes
   - the probe orientation changes
   - the user explicitly requests a render

3. Display the rendered image in a dedicated panel or viewer.

4. Prevent the UI from freezing during rendering.
   A first version can render synchronously if acceptable, but the code should
   be structured so background execution can be added later.

5. Decide on update policy:
   - render on every pose change
   - render on mouse release
   - render on explicit button press

6. Expose errors cleanly if the pose is invalid or outside the meaningful field
   of view.

## What Needs To Be Checked

- The rendered image changes consistently with probe movement
- The orientation shown in the 3D viewer matches the rendered image orientation
- UI responsiveness is acceptable

## Output of This Ticket

- A live or semi-live NeRF rendering panel driven by the virtual probe

## Acceptance Criteria

- A user can move the probe and trigger a new render without leaving the viewer
- The rendered output corresponds to the current application pose state

## Dependencies

- T08
- T09
- T10

## Blocks

- T12
- T15
