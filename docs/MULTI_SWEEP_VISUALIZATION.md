# Multi-Sweep Visualization

This document describes how the visualization toolkit should handle multiple
tracked ultrasound sweeps captured from different angles or acquisition passes.

## Core Assumption

If the NeRF was trained successfully on the multi-sweep dataset, the sweep
poses should already align in one common world frame in theory.

The visualization toolkit should still validate that assumption before showing:

- a fused aggregate volume
- cross-sweep nearest-frame comparisons
- overlaid trajectories from several sweeps

Do not assume that successful training is, by itself, sufficient proof of
visual alignment quality.

## Coordinate Convention

- visualization space uses `mm`
- all sweeps should be interpreted in one shared world frame
- each sweep may optionally carry an additional rigid
  `world_transform_mm` if external registration or correction is needed

The effective world pose used by visualization is:

`T_probe_to_world_effective = T_sweep_to_world @ T_probe_to_world_recorded`

In the default case, `T_sweep_to_world` is identity and the recorded poses are
already assumed to be in the shared world frame.

## Data Model

The multi-sweep backend represents one scene as:

- a set of `SweepRecord` objects
- one `MultiSweepScene`
- optional alignment-validation results
- optional per-sweep fused volumes
- one aggregate fused volume built from enabled sweeps

Each `SweepRecord` contains:

- `sweep_id`
- `display_name`
- `images`
- `raw_poses_mm`
- `poses_mm`
- `world_transform_mm`
- `probe_geometry`
- `enabled`
- `alignment_source`
- optional metadata and display color

## Loading Modes

Current backend support includes:

- manifest-based loading from JSON
- directory-of-sweeps discovery

The manifest should define:

- shared probe geometry, unless provided externally
- sweep ids
- dataset directories
- optional display names
- optional colors
- optional per-sweep transforms
- optional metadata

## Alignment Checks Before Visualization

Run cross-sweep validation before interpreting the scene.

The current backend checks:

- sweep support bounds
- trajectory center statistics
- pairwise centroid distance
- nearest center distance
- pairwise support overlap fraction

These checks are heuristic. They are intended to flag suspicious offsets, not
to certify registration quality.

## Recommended Manual QA

Use this checklist before trusting a multi-sweep visualization:

1. Load the scene manifest and confirm the expected number of sweeps.
2. Check that the active sweep and enabled sweeps are what you expect.
3. Inspect the alignment-validation output and note any warned sweep pairs.
4. Show sweeps one at a time and confirm each trajectory sits inside its own
   scan support.
5. Overlay several sweeps and look for obvious doubled or offset trajectories.
6. Toggle the aggregate fused volume on and off and confirm it changes when
   enabled sweeps change.
7. If nearest-frame comparison is used, verify which sweep the matched frame
   came from.
8. Repeat comparison in both:
   - all enabled sweeps mode
   - active sweep only mode
9. If a sweep looks offset, apply or inspect its `world_transform_mm` before
   trusting any fused or comparative view.

## Current Backend Pieces

Implemented multi-sweep backend modules:

- `visualization/multi_sweep.py`
- `visualization/multi_sweep_loader.py`
- `visualization/alignment_validation.py`
- `visualization/multi_sweep_volume.py`
- `visualization/multi_sweep_comparison.py`
- `visualization/multi_sweep_ui.py`

These provide:

- scene types
- manifest loading
- alignment validation
- per-sweep transforms
- aggregate fusion
- sweep-aware nearest-frame comparison
- multi-sweep viewer state management

## Known Gap

The current napari launch path is still centered on the single-sweep viewer.

The multi-sweep backend is ready, but full app wiring still needs:

- a multi-sweep launch entry point
- napari layer composition for multiple sweeps
- a dock widget wired into the live viewer session
- aggregate/per-sweep volume toggling in the running app

## Example Manifest Shape

```json
{
  "probe_geometry": {
    "width_mm": 80.0,
    "depth_mm": 139.0
  },
  "active_sweep_id": "sweep_a",
  "comparison_policy": "all_enabled",
  "sweeps": [
    {
      "sweep_id": "sweep_a",
      "dataset_dir": "data/session_a",
      "display_name": "Anterior Sweep",
      "color_rgb": [0.9, 0.7, 0.2]
    },
    {
      "sweep_id": "sweep_b",
      "dataset_dir": "data/session_b",
      "display_name": "Lateral Sweep",
      "color_rgb": [0.2, 0.7, 0.9],
      "world_transform_mm": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ]
    }
  ]
}
```
