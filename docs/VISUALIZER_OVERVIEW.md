# Visualizer Overview

This document explains what the current ultrasound visualizer is, how it is
laid out, and how the main state concepts relate to the underlying data.

## Purpose

The visualizer is a napari-based review workspace for:

- tracked 2D ultrasound sweeps
- fused 3D sweep volumes
- per-sweep trajectory inspection
- interactive probe placement
- nearest recorded-frame comparison
- checkpoint-backed NeRF rendering from arbitrary probe poses

It is designed as a research tool, not a polished clinical workstation.

## Entry Points

Single-sweep launch:

```bash
python run_visualize_sweeps.py --dataset-dir <dataset_dir>
```

Multi-sweep launch:

```bash
python run_visualize_multi_sweeps.py --manifest-path <manifest.json>
```

Checkpoint-backed multi-sweep launch:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/spine_phantom/multi_sweep_manifest.json \
  --checkpoint-path logs/spine_phantom_multi_v2/005000.tar \
  --config-path logs/spine_phantom_multi_v2/args.txt \
  --device cuda \
  --render-trigger-mode manual
```

## Current Workspace Layout

The current multi-sweep workspace uses four regions:

1. left column: `Probe Controls`
2. center: main 3D viewer
3. center-right: review panels
4. far-right column: sweep-state controls

More concretely:

- left:
  - probe translation and rotation
  - recorded frame index
  - reset/snap actions

- center:
  - aggregate fused volume or per-sweep volume display
  - probe scan plane
  - beam line
  - trajectory overlays

- center-right:
  - `Nearest Recorded Frame`
  - `NeRF Render`

- far-right:
  - `Multi-Sweep Controls`
  - `Sweep Selection`

The layout is intentionally not the default napari dock layout. The primary
panes are placed in a custom workspace so the startup arrangement is stable.

## Data Model In The Viewer

The visualizer distinguishes between three related ideas:

- `Active Sweep`
- `Enabled`
- `Visible`

These are not the same.

### Active Sweep

The active sweep controls:

- which sweep the frame index refers to
- what `Reset To Frame` uses
- which sweep is considered "active" for `Active Sweep Only` comparison mode
- which trajectory is emphasized in aggregate mode

Only one sweep is active at a time.

### Enabled

Enabled sweeps are eligible for:

- nearest-frame comparison
- `Snap To Nearest`
- `All Enabled Sweeps` comparison mode

Enabled does not necessarily mean visible in the 3D scene.

### Visible

Visible sweeps control which per-sweep volumes are shown when aggregate mode is
off.

This is intentionally independent from `Enabled`, because users often want:

- broad comparison/search scope
- but only a small subset of sweeps displayed at once

By default, leaving aggregate mode shows only the active sweep volume, not all
enabled sweeps.

## Aggregate vs Per-Sweep Mode

### Aggregate Mode

When `Show Aggregate Volume` is enabled:

- one fused all-sweeps volume is shown
- only the active trajectory is emphasized
- the scene is used mainly for spatial context

This mode is useful for:

- understanding global anatomy
- approximate probe placement
- seeing the shared sweep support

### Per-Sweep Mode

When `Show Aggregate Volume` is disabled:

- the aggregate volume is hidden
- per-sweep volumes are shown for the sweeps marked `Visible`
- trajectories are shown for the visible sweeps

This mode is useful for:

- isolating one sweep
- comparing a small subset of sweeps
- reducing visual clutter

## Nearest Recorded Frame

The nearest-frame panel shows the recorded frame whose pose is closest to the
current virtual probe pose.

Closeness is computed from:

- translation distance in mm
- rotation distance in degrees

The current implementation uses a simple combined score:

`score = translation_mm + rotation_deg`

Candidate sweeps are controlled by the comparison policy:

- `All Enabled Sweeps`
- `Active Sweep Only`

## Snap To Nearest

`Snap To Nearest` moves the virtual probe to the recorded pose with the lowest
pose-distance score.

If that nearest pose belongs to another sweep, the viewer may switch the active
sweep as part of the snap operation.

## NeRF Render

If a checkpoint is provided, the viewer also renders the NeRF prediction for the
current virtual probe pose.

That render uses:

- the current probe pose from the viewer
- the configured probe geometry
- the existing `NerfSession` bridge to the training/runtime code

The viewer itself works in millimeters. The model runtime converts to the
meter-scale format expected by the training code at the inference boundary.

## Performance Notes

The expensive parts of the viewer are usually not the pose math. They are:

- building per-sweep fused volumes
- creating/updating many napari volume layers
- switching from aggregate mode to many visible per-sweep volumes

This is why:

- aggregate mode is faster and lighter
- per-sweep mode defaults to showing only the active sweep
- loading/busy feedback is shown for heavy state changes

## Important Current Limitations

- per-sweep volume switching can still be slow when many sweeps are made visible
- the viewer is optimized for inspection, not final production UX
- nearest-frame matching uses a simple weighted pose score, not a learned or
  anatomy-aware metric
- the aggregate transfer behavior is still a practical visualization preset, not
  a full medical transfer-function editor
