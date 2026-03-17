# T34 - Wire Multi-Sweep Controls and Comparison Into the Live Viewer

## Goal

Connect the multi-sweep control state, comparison policy, and live viewer
updates into one running app session.

## Why This Matters

The backend state controller and comparison logic exist, but the user still
needs the app to react when they:

- change the active sweep
- enable or disable sweeps
- switch comparison policy
- request nearest-frame matching in multi-sweep mode

Without that wiring, the viewer cannot actually behave like a multi-sweep
inspection tool.

## Required Work

- Add the multi-sweep controls dock to the running viewer session.
- Connect control changes to:
  - scene refresh
  - aggregate fusion refresh
  - comparison policy updates
  - active sweep tracking
- Ensure nearest-frame comparison metadata includes the matched sweep.
- Ensure probe reset/snap behavior is explicit in multi-sweep mode.
- If NeRF rendering is enabled, confirm that render comparison still uses the
  current multi-sweep comparison policy.

## What Needs To Be Checked

- Changing the active sweep updates the scene state.
- Enabling/disabling sweeps refreshes the aggregate volume correctly.
- Comparison metadata switches sweep ids when expected.
- Live viewer updates remain stable under repeated control changes.

## Output of This Ticket

- Multi-sweep dock wiring in the live viewer.
- Tests for state-driven viewer refresh behavior.

## Acceptance Criteria

- A user can control multi-sweep visibility and comparison policy from the app.
- The viewer responds correctly to those changes without restarting.

## Dependencies

- T29
- T30
- T32
- T33

## Blocks

- T35
