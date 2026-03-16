# T08 - Add Probe Placement From MPR

## Goal

Allow the user to select a probe center or initial probe pose from the MPR views
rather than only from the 3D view.

## Why This Matters

Placing a probe center is easier and more precise in orthogonal slice views.
This is the interaction style the user specifically asked for.

## Required Work

1. Define the interaction model.
   Example:
   - click in one MPR to set center
   - click in another MPR to refine position
   - use a control widget to confirm the probe center

2. Convert selected MPR positions into a world-space probe center.

3. Update the 3D probe representation when the selected center changes.

4. Preserve the current orientation or use a default orientation until T09 is
   implemented.

5. Ensure the selected point is visible in all views.

## Interaction Considerations

- The user may place the center in a slice that does not contain strong signal
- The UX should distinguish between “move crosshair” and “commit probe center”
- The first version can use simple click-to-place behavior

## What Needs To Be Checked

- The selected point maps correctly from MPR voxel space to world coordinates
- The 3D probe moves to the expected location
- The displayed plane stays centered at the chosen location

## Output of This Ticket

- An MPR-driven probe placement workflow
- World-space probe center updates exposed to the application state

## Acceptance Criteria

- A user can place the probe from MPR views without editing code
- The probe center updates consistently across MPR and 3D views

## Dependencies

- T06
- T07

## Blocks

- T11
