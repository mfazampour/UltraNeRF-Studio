# T30 - Add Multi-Sweep UI Controls and Viewer State Management

## Goal

Add the viewer controls and state-management logic needed to work with multiple
sweeps in one session.

## Why This Matters

Once multiple sweeps exist in the scene, the user needs explicit control over
what is visible and what is considered active.

Without a dedicated UI layer, the scene quickly becomes confusing:

- too many overlapping trajectories
- unclear source of the nearest recorded frame
- no obvious way to isolate one sweep
- no way to rebuild the global fused volume from only selected sweeps

## Required Work

- Extend app state so it can track:
  - active sweep id
  - enabled sweep ids
  - per-sweep visibility
  - comparison policy
  - whether the aggregate fused volume is displayed
- Add UI controls for:
  - sweep list
  - enable/disable toggles
  - active sweep selection
  - comparison policy selection
  - aggregate-vs-per-sweep display mode
- Update napari layer management so the scene can add, remove, and refresh
  per-sweep layers without leaking stale layers.
- Decide how colors are assigned and surfaced in the UI.
- Ensure probe reset and snap actions are understandable in a multi-sweep
  context.
- Surface alignment-validation status from T26 in the viewer, at least as text
  or warning state.

## Suggested Implementation

- Keep business logic out of the dock widgets where possible.
- Extend the existing `VisualizationUIController` rather than duplicating scene
  update logic.
- Add one dedicated multi-sweep control panel rather than scattering related
  controls across several unrelated widgets.

## What Needs To Be Checked

- Enabling and disabling sweeps updates the scene correctly.
- The active sweep is visible in the UI and affects the appropriate actions.
- Aggregate fusion refreshes when the enabled-sweep set changes.
- Comparison metadata updates when the comparison policy changes.
- Layer cleanup is correct and does not leave duplicated overlays behind.

## Output of This Ticket

- Multi-sweep UI controls.
- Multi-sweep-aware viewer state management.
- Tests for state transitions and layer update logic.

## Acceptance Criteria

- A user can load multiple sweeps and understand which ones are active.
- The scene remains navigable and controllable when many sweeps are present.
- Layer updates remain stable when visibility settings change repeatedly.

## Dependencies

- T28
- T29

## Blocks

- T31
