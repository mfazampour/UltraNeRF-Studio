# T06 - Add Synchronized MPR Views

## Goal

Add synchronized orthogonal slice views alongside the 3D volume view so that
users can inspect the sweep volume in MPR form and use those views for probe
placement.

## Why This Matters

The user wants to place the probe from MPR-like views and later refine its
orientation. That workflow is much easier in 2D orthogonal views than by trying
to manipulate a probe directly in 3D.

## Required Work

1. Add three linked slice viewers or an equivalent MPR layout.

2. Synchronize the selected world point across all viewers.

3. Show the current crosshair or focus point in each MPR pane.

4. Keep the 3D view synchronized with the current selected location.

5. Decide whether the MPR views are:
   - fixed orthogonal world planes
   - reformat views tied to the current volume axes

6. Expose the selected point in world coordinates for downstream probe placement.

## Suggested Output Layout

- one 3D view
- three 2D views
- optional control panel or dock widget

## What Needs To Be Checked

- The same world point appears consistently in all MPR views
- The MPR index mapping respects voxel origin and spacing
- Selection updates are stable and do not drift due to axis confusion

## Output of This Ticket

- Linked MPR + 3D visualization of the sweep volume
- A selection model that later tickets can use for probe placement

## Acceptance Criteria

- Clicking or selecting a point updates all slice viewers consistently
- The selected world coordinate can be retrieved programmatically

## Dependencies

- T05

## Blocks

- T08
