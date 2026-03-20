# Visualizer Workflow

This document is a practical guide for using the current visualizer.

## 1. Start The Viewer

Multi-sweep example:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/spine_phantom/multi_sweep_manifest.json
```

Checkpoint-backed example:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/spine_phantom/multi_sweep_manifest.json \
  --checkpoint-path logs/spine_phantom_multi_v2/005000.tar \
  --config-path logs/spine_phantom_multi_v2/args.txt \
  --device cuda \
  --render-trigger-mode manual
```

If you only want to inspect the prepared state without opening the GUI:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/spine_phantom/multi_sweep_manifest.json \
  --no-gui
```

## 2. Understand The Panels

### Probe Controls

Use this panel to:

- move the probe in `x/y/z`
- rotate the probe with `yaw/pitch/roll`
- jump back to a recorded frame with `Reset To Frame`
- snap to the nearest recorded pose with `Snap To Nearest`

### Multi-Sweep Controls

Use this panel to:

- choose the active sweep
- choose the comparison policy
- switch aggregate mode on/off

### Sweep Selection

Each sweep has two independent controls:

- `Enabled`
- `Visible`

Meaning:

- `Enabled`
  Used for nearest-frame comparison and `Snap To Nearest`

- `Visible`
  Used only for per-sweep display when aggregate mode is off

This separation is intentional.

## 3. Typical Review Flow

Recommended sequence:

1. start in aggregate mode
2. position the probe approximately in the shared scene
3. inspect the nearest recorded frame
4. if a checkpoint is loaded, render the NeRF output
5. switch aggregate mode off
6. keep only the sweeps you want `Visible`
7. compare per-sweep 3D support with the recorded and rendered images

## 4. Aggregate Mode

Use aggregate mode when you want:

- global anatomical context
- rough positioning
- a lighter initial scene

In aggregate mode:

- one fused volume is shown
- the active sweep trajectory is emphasized
- other sweeps still matter for comparison if they remain `Enabled`

## 5. Per-Sweep Mode

Use per-sweep mode when you want:

- less clutter
- one or a few sweeps at a time
- closer inspection of how a specific sweep sits in space

Important current behavior:

- turning aggregate mode off no longer shows all sweeps by default
- it starts with the active sweep visible
- then you can add more visible sweeps explicitly

This is the main control for reducing visual overload.

## 6. Choosing Enabled vs Visible

A good default pattern is:

- keep several sweeps `Enabled`
- keep only one or two sweeps `Visible`

That gives:

- broad comparison/search behavior
- low visual clutter

Example:

- `Enabled`: all nine sweeps
- `Visible`: just `right1_1`

Then:

- nearest-frame matching can still search all enabled sweeps
- but the 3D viewer only shows one per-sweep volume

## 7. Snap To Nearest

`Snap To Nearest` uses the current probe pose and finds the recorded pose with
the smallest combined pose-distance score.

The score is based on:

- translation distance in mm
- rotation distance in degrees

Candidate sweeps depend on the comparison policy:

- `Active Sweep Only`
- `All Enabled Sweeps`

If the nearest pose belongs to another sweep, the active sweep may change.

## 8. When A Loading Dialog Appears

Some actions still perform heavy synchronous work.

Examples:

- switching aggregate mode
- applying a different visible-sweep selection

The viewer now shows shared busy feedback during these transitions so you can
see that it is still working.

This is a usability improvement, not a final performance solution.

## 9. Common Troubleshooting

### The scene looks too cluttered

- turn aggregate mode off
- keep only one or two sweeps `Visible`
- leave more sweeps `Enabled` only if you still want them for comparison

### The wrong sweep keeps showing

Check:

- `Active Sweep`
- `Visible` checkboxes
- `Apply Selection`

`Visible` is now independent from `Enabled`, so the displayed sweep set should
match exactly what you selected.

### Snap To Nearest jumps to an unexpected sweep

Check the comparison policy:

- `All Enabled Sweeps`
  searches across all enabled sweeps

- `Active Sweep Only`
  restricts the search to the current active sweep

### Turning aggregate off is still slow

That usually means:

- too many sweeps are visible
- the viewer is updating several heavy volume layers at once

Keep the visible set small unless you specifically need a larger overlay.

## 10. Recommended QA For New Datasets

When using a new multi-sweep dataset:

1. launch with `--no-gui` first
2. check the alignment warnings
3. start in aggregate mode
4. inspect one active sweep at a time
5. enable a checkpoint only after the geometric scene looks plausible
6. compare:
   - 3D probe pose
   - nearest recorded frame
   - NeRF render

This usually makes geometry and registration issues obvious early.
