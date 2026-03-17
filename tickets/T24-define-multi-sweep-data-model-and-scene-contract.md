# T24 - Define the Multi-Sweep Data Model and Scene Contract

## Goal

Define how the visualization toolkit represents more than one ultrasound sweep
in a single session.

This ticket is about the core application contract. It should answer:

- what a single sweep object contains
- how multiple sweeps are grouped into one scene
- what coordinate frame the viewer uses
- what assumptions are made about alignment between sweeps

## Why This Matters

The current visualization code assumes exactly one sweep:

- one `images.npy`
- one `poses.npy`
- one trajectory
- one fused volume
- one nearest-recorded-frame search space

That model breaks as soon as multiple sweeps are loaded. If the code tries to
support multi-sweep inputs without a clear data model, alignment bugs,
ambiguous UI behavior, and incorrect frame matching will appear immediately.

This ticket should establish one explicit rule:

- the visualization scene operates in a shared world frame expressed in
  millimeters

Even if the NeRF training setup implies that all sweeps should already align in
that frame, the toolkit must still treat cross-sweep alignment as something to
verify rather than assume blindly.

## Required Work

- Define a `SweepRecord`-style structure that contains:
  - sweep identifier
  - sweep label or display name
  - tracked images
  - tracked poses
  - optional metadata
  - optional color / visibility defaults
  - optional registration transform into the global scene frame
- Define a `MultiSweepScene` or equivalent container that contains:
  - a list of sweeps
  - global scene metadata
  - the canonical visualization unit system
  - viewer-level defaults such as active sweep and enabled sweeps
- Document the expected coordinate semantics:
  - input frame pixel space
  - probe-local sweep pose
  - sweep-local frame, if any
  - shared world frame
- Decide how the toolkit behaves when no per-sweep registration transform is
  provided:
  - assume poses are already in the shared world frame
  - mark that assumption in metadata
  - allow later validation to confirm or challenge it
- Define what the "active sweep" means in the UI and application state.
- Define how nearest-frame comparison behaves in a multi-sweep scene:
  - search across all enabled sweeps by default
  - optionally restrict to the active sweep
- Define what should happen when sweeps have inconsistent image sizes or probe
  metadata.

## Suggested Implementation

- Add new data classes in a dedicated module such as:
  - `visualization/multi_sweep.py`
- Keep the structures small and explicit. This is not the place to add heavy
  logic.
- Reuse the existing single-sweep `VisualizationAppState` pieces where
  possible, but do not force multi-sweep data into single-sweep fields.
- Add enough metadata fields that later tickets can attach:
  - fused per-sweep volumes
  - trajectory overlays
  - alignment validation results

## What Needs To Be Checked

- The data model supports:
  - one sweep
  - multiple sweeps
  - sweeps with optional per-sweep transforms
- The distinction between:
  - sweep-local coordinates
  - global world coordinates
  is unambiguous.
- The model clearly records whether sweep alignment is:
  - assumed from training inputs
  - externally registered
  - validated by toolkit checks

## Output of This Ticket

- A documented multi-sweep scene contract.
- Data classes or typed structures for sweeps and scene state.
- Unit tests validating the data model behavior.

## Acceptance Criteria

- A developer can read the ticket output and understand how to represent
  multiple sweeps without guessing.
- The canonical unit system remains millimeters.
- The shared world frame contract is explicit.
- Alignment assumptions are encoded, not left implicit.

## Dependencies

- T01

## Blocks

- T25
- T26
- T27
- T28
- T29
- T30
- T31
