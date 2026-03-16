# T23 - Add Workflow Docs and Manual QA for Interactive Rendering

## Goal

Document how to run, validate, and troubleshoot the interactive NeRF viewer
once the rendered-output workflow is available in the GUI.

## Why This Matters

The interactive viewer will combine:

- sweep fusion
- 3D visualization
- probe manipulation
- model loading
- live rendering
- frame comparison

Without a clear workflow document and a manual QA checklist, it will be hard for
other developers to verify that a change preserves behavior.

## Required Work

- Update the main README with a dedicated section for the interactive viewer.
- Add concrete launch commands for:
  - sweep-only mode
  - sweep + NeRF mode
  - headless cache-building mode
- Document which dataset files are required.
- Document which model files are required for checkpoint-backed rendering.
- Explain the current unit convention clearly:
  - viewer coordinates in mm
  - current NeRF runtime conversion boundary to meters
- Add a manual QA checklist that covers:
  - launch success
  - probe movement
  - render update behavior
  - comparison-panel correctness
  - basic failure handling
- Add troubleshooting notes for:
  - missing Qt / X11
  - missing checkpoint
  - invalid config
  - empty-looking volume
  - render too slow

## What Needs To Be Checked

- The documented commands work on a clean environment.
- The QA checklist reflects the actual UI behavior and current controls.
- The docs distinguish clearly between implemented features and planned ones.

## Output of This Ticket

- Updated user-facing documentation for the interactive viewer.
- A repeatable manual QA checklist for future changes.

## Acceptance Criteria

- Another developer can launch the viewer and perform a structured manual check
  without reading the implementation code first.
- The README and the ticket list reflect the actual state of the feature.

## Dependencies

- T18
- T19
- T20
- T21
- T22

## Blocks

- None
