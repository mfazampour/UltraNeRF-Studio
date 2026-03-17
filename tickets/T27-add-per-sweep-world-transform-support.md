# T27 - Add Per-Sweep World Transform Support

## Goal

Allow each sweep to carry an optional rigid transform into the shared viewer
world frame.

## Why This Matters

The preferred case is that all sweep poses are already aligned in one common
frame because they were prepared consistently for NeRF training.

However, the visualization toolkit should still support explicit per-sweep
registration transforms for at least three reasons:

- some datasets may be prepared in sweep-local tracker frames
- users may want to correct small offsets discovered during validation
- external registration workflows may produce better alignment than the raw
  training inputs

If the toolkit cannot apply per-sweep transforms cleanly, it becomes difficult
to inspect or repair multi-sweep scenes.

## Required Work

- Define where the per-sweep transform lives in the data model.
- Apply that transform consistently to:
  - trajectory overlays
  - probe poses
  - sweep-to-volume fusion
  - nearest-frame comparison
  - any rendered scene geometry derived from the sweep
- Keep the distinction clear between:
  - raw recorded poses
  - transformed poses used for visualization
- Decide whether transforms are:
  - immutable scene metadata
  - editable in memory for experimentation
  - both
- Provide helper functions to transform all poses for a sweep without
  scattering matrix-multiplication logic across the codebase.
- Ensure transforms remain millimeter-based in the visualization layer.

## Suggested Implementation

- Extend the T24 data structures with:
  - `raw_poses_mm`
  - `world_transform_mm`
  - `world_poses_mm`
- Put transform application helpers in the multi-sweep backend rather than in
  the napari UI code.
- Keep transform composition rules explicit and documented.

## What Needs To Be Checked

- Identity transforms leave sweep geometry unchanged.
- Non-identity transforms move trajectories and frame poses as expected.
- The same transform is used consistently across all downstream consumers.
- Comparison and fusion do not accidentally mix raw and transformed poses.

## Output of This Ticket

- Per-sweep world transform support in the data model and backend helpers.
- Tests covering identity and non-identity transform behavior.

## Acceptance Criteria

- A sweep can be visualized either:
  - directly in its existing world frame
  - or through an explicit additional world transform
- Downstream modules consume one consistent transformed pose set.

## Dependencies

- T24

## Blocks

- T28
- T29
- T30
- T31
