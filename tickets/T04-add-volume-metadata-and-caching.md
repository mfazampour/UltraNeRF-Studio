# T04 - Add Volume Metadata and Caching

## Goal

Persist fused sweep volumes to disk together with enough metadata to reload them
without recomputing the fusion process on every application launch.

## Why This Matters

Sweep fusion may become expensive for large datasets. An interactive
visualization tool should load cached volume data when possible and only rebuild
when configuration or source inputs change.

## Required Work

1. Define a cached volume file format.
   Reasonable choices:
   - `npz`
   - `zarr`
   - `npy` plus JSON metadata

2. Store:
   - scalar volume
   - optional weight volume
   - voxel spacing
   - origin
   - bounds
   - source dataset identifier
   - probe geometry parameters
   - volume resolution
   - fusion parameters

3. Implement save and load helpers.

4. Add cache invalidation rules.
   At minimum, the cache should become invalid if:
   - source dataset changes
   - volume resolution changes
   - probe geometry changes
   - fusion algorithm parameters change

5. Decide where cached volumes live.
   Suggested options:
   - `logs/volumes/`
   - `cache/`
   - dataset-local cache folder

6. Ensure the cache format is versioned so future changes can invalidate older
   files cleanly.

## What Needs To Be Checked

- Reloaded cached data reproduces the same volume and metadata
- Bounds and spacing are not lost or reordered
- Cache invalidation behaves predictably
- The cache path does not interfere with source data

## Output of This Ticket

- A portable cached volume representation
- Fast reload behavior for the visualization app

## Acceptance Criteria

- A fused volume can be saved and loaded with no meaningful data loss
- Cache metadata is sufficient to reconstruct world-to-voxel mapping
- The app can detect whether a cached volume is reusable

## Dependencies

- T03

## Blocks

- T05
- T15
