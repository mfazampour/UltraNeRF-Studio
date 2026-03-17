# T33 - Add Multi-Sweep Scene Composition to Napari

## Goal

Teach the napari viewer layer to render multi-sweep scenes instead of only the
single-sweep volume/trajectory/probe setup.

## Why This Matters

The current napari composition logic assumes:

- one fused volume
- one trajectory
- one nearest-frame search space

Multi-sweep use requires the viewer to manage:

- several trajectories
- several per-sweep overlays
- one optional aggregate volume
- one active sweep context

This is the point where the backend becomes visible to end users.

## Required Work

- Add multi-sweep-aware layer composition to the viewer.
- Decide how the scene should present:
  - per-sweep trajectories
  - per-sweep volumes
  - aggregate fused volume
- Use stable layer naming so scene refreshes do not leak stale layers.
- Distinguish sweeps visually by:
  - color
  - naming
  - visibility state
- Ensure the active sweep is visually identifiable.
- Decide whether the probe overlay remains single and global or whether
  sweep-local probe context needs to be shown separately.

## What Needs To Be Checked

- Layers for several sweeps appear in the correct world-space positions.
- Toggling between aggregate and per-sweep display modes is stable.
- Layer refreshes do not duplicate old trajectory or volume layers.
- Color assignment remains consistent across updates.

## Output of This Ticket

- Multi-sweep napari scene composition.
- Tests using fake viewers to verify layer creation and updates.

## Acceptance Criteria

- A multi-sweep session can be displayed in napari with meaningful overlays.
- The scene remains readable when more than one sweep is present.

## Dependencies

- T28
- T30
- T32

## Blocks

- T34
- T35
