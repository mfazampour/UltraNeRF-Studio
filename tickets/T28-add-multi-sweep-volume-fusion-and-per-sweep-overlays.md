# T28 - Add Multi-Sweep Volume Fusion and Per-Sweep Overlays

## Goal

Extend the sweep fusion and scene-overlay pipeline so the viewer can handle
multiple sweeps at once.

## Why This Matters

For multi-sweep usage, the user needs more than a single fused scalar volume.
They need to inspect:

- each sweep individually
- several sweeps overlaid together
- a single fused volume built from all enabled sweeps

Those views are complementary. Individual sweep overlays help diagnose coverage
and alignment. A global fused volume helps interpret the combined acquisition.

## Required Work

- Extend sweep fusion logic so it can:
  - fuse one sweep
  - fuse a selected subset of sweeps
  - fuse all enabled sweeps
- Keep provenance available:
  - which sweep contributed to which volume
  - whether a volume is per-sweep or global
- Add per-sweep trajectory overlays with distinct display identity:
  - color
  - name
  - visibility
- Define whether the global fused volume is:
  - one scalar accumulation from all sweeps
  - or multiple layers plus an aggregate
- Decide how voxel bounds are chosen for multi-sweep fusion:
  - union of enabled sweep bounds
  - user-provided bounds
  - cached bounds from a manifest
- Ensure all fusion uses transformed world-space poses from T27.
- Update caching logic if needed so per-sweep and global fused volumes can be
  reused.

## Suggested Implementation

- Add modules such as:
  - `visualization/multi_sweep_volume.py`
  - `visualization/sweep_registry.py`
- Keep per-sweep and global fused outputs separate in the API.
- Preserve the existing single-sweep fusion path as a special case of the new
  multi-sweep backend rather than maintaining two unrelated implementations.

## What Needs To Be Checked

- Per-sweep volumes render in the correct place in the shared world frame.
- Global fused volume reflects all enabled sweeps only.
- Disabling a sweep removes its contribution from the aggregate volume.
- Overlay trajectories line up with their corresponding sweep volume support.
- Cache invalidation is correct when enabled sweeps change.

## Output of This Ticket

- Multi-sweep volume fusion backend.
- Per-sweep trajectory and volume overlays.
- Tests for union bounds, enabled-sweep filtering, and transform-aware fusion.

## Acceptance Criteria

- The toolkit can show:
  - one selected sweep
  - multiple visible sweeps
  - one aggregate fused volume from all enabled sweeps
- Sweep-specific overlays remain distinguishable in the scene.

## Dependencies

- T25
- T26
- T27

## Blocks

- T30
- T31
