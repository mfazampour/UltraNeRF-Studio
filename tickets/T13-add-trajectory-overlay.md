# T13 - Add Trajectory Overlay

## Goal

Visualize the tracked probe trajectory and optionally frame-by-frame probe
orientations inside the 3D scene.

## Why This Matters

Trajectory context helps users understand sweep coverage, missing regions, and
how the interactive probe relates to the acquisition path.

## Required Work

1. Load the tracked poses and extract probe centers.

2. Display the trajectory in the 3D scene as:
   - a polyline
   - point markers
   - optional pose axes at selected intervals

3. Optionally highlight:
   - current nearest recorded frame
   - selected frame
   - sweep start and end

4. Expose toggles for trajectory visibility and density.

## What Needs To Be Checked

- Trajectory points align with the sweep volume
- Probe orientation overlays point in the correct directions
- The trajectory does not clutter the scene excessively

## Output of This Ticket

- A clear 3D representation of the recorded sweep path

## Acceptance Criteria

- A user can see where the original probe moved during acquisition
- Trajectory overlays help relate the virtual probe to real data coverage

## Dependencies

- T05
- T07

## Blocks

- None
