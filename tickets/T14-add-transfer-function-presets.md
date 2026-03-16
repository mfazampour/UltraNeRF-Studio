# T14 - Add Transfer-Function Presets

## Goal

Add useful preset volume rendering configurations so the fused sweep data can be
inspected quickly without manual tuning every time.

## Why This Matters

Raw volume rendering controls are often cumbersome. Presets help users see the
input data immediately and make the visualization tool feel usable.

## Required Work

1. Define a small set of presets, for example:
   - soft tissue / general intensity
   - high-contrast edges
   - sparse high-signal structures
   - confidence-weighted visualization if available

2. Implement preset application to the viewer.

3. Expose presets in the UI.

4. Preserve manual override after applying a preset.

## What Needs To Be Checked

- Presets produce clearly distinct visualizations
- They work across at least a small set of representative datasets
- Switching presets does not corrupt viewer state

## Output of This Ticket

- Usable transfer-function presets for common exploration tasks

## Acceptance Criteria

- A user can switch among presets and immediately improve sweep visibility

## Dependencies

- T05

## Blocks

- None
