# T29 - Add Sweep-Aware Comparison and Probe Context

## Goal

Make nearest-recorded-frame matching and probe context aware of multiple
sweeps.

## Why This Matters

In a multi-sweep scene, "nearest recorded frame" is ambiguous unless the search
policy is explicit.

Possible policies include:

- nearest frame across all enabled sweeps
- nearest frame within the active sweep only
- nearest frame within a user-selected subset

The viewer should support these intentionally. Otherwise, users may compare the
NeRF render against a frame from an unexpected sweep and misinterpret the
result.

## Required Work

- Extend the comparison backend so it can search across multiple sweeps.
- Return comparison metadata that includes:
  - sweep id
  - sweep label
  - frame index
  - translation difference
  - rotation difference
- Define a comparison policy object or enum.
- Update probe context so the viewer can report:
  - current active sweep
  - current nearest sweep
  - whether the nearest frame came from a different sweep than the active one
- Ensure comparison always uses transformed world-space poses from T27.
- Decide how probe reset / snap actions behave:
  - snap to nearest frame globally
  - snap to nearest frame in active sweep
- Keep single-sweep behavior unchanged as the trivial case.

## Suggested Implementation

- Extend:
  - `visualization/comparison.py`
  or add:
  - `visualization/multi_sweep_comparison.py`
- Represent matches as structured objects rather than raw dicts where possible.
- Keep the search backend independent from the UI so it can be tested in CLI
  mode.

## What Needs To Be Checked

- Comparison across all sweeps chooses the closest valid frame.
- Restricting to the active sweep changes the result appropriately.
- Returned metadata clearly identifies which sweep the frame belongs to.
- Probe snap actions do not silently switch sweeps unless configured to do so.

## Output of This Ticket

- Multi-sweep-aware comparison backend.
- Sweep-aware match metadata for the viewer.
- Tests for comparison policy behavior.

## Acceptance Criteria

- The user can tell which sweep a matched frame belongs to.
- Comparison policy is explicit and configurable.
- Single-sweep comparison behavior remains unchanged.

## Dependencies

- T25
- T27

## Blocks

- T30
- T31
