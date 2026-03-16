# T21 - Add Live Render Triggering and Throttling

## Goal

Connect probe manipulation to the NeRF renderer so that pose changes can trigger
image updates in a controlled and responsive way.

## Why This Matters

Once the app can both:

- load a NeRF checkpoint
- display rendered output
- let the user move the probe

the next problem is interaction quality. Rendering on every tiny pose change may
be too expensive. Rendering only on a button press may be too cumbersome.

This ticket defines the application behavior that makes the viewer usable.

## Required Work

- Expose render trigger modes such as:
  - manual
  - on pose change
  - on pose change with debounce or throttle
- Decide where render requests originate:
  - UI widget callbacks
  - scene controller state updates
  - explicit render buttons
- Ensure only one render request is active at a time, or define how overlap is
  handled.
- Avoid runaway render loops caused by UI updates triggering more UI updates.
- Update the rendered-output panel in-place after each successful render.
- Show basic render state such as:
  - idle
  - rendering
  - failed
- Preserve the last valid rendered image when a new render fails.

## Performance Considerations

- Rendering may be slow enough that the app needs:
  - throttling
  - debouncing
  - optional background execution
- The first version can remain synchronous if that keeps behavior simple, but
  the control flow should not make a later async refactor impossible.

## What Needs To Be Checked

- Manual mode renders only when explicitly requested.
- Automatic mode updates when the probe moves.
- Throttled mode does not flood the renderer during rapid input changes.
- Failed renders do not break the viewer state.

## Output of This Ticket

- End-to-end connection from probe pose changes to rendered image updates.
- Tests for controller behavior and trigger-mode transitions.

## Acceptance Criteria

- A user can move the probe and observe the rendered image update according to
  the selected trigger policy.
- The viewer remains usable during repeated updates.
- No duplicate or runaway render loops occur in normal use.

## Dependencies

- T18
- T19
- T20

## Blocks

- T22
