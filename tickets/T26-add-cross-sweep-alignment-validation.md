# T26 - Add Cross-Sweep Alignment Validation

## Goal

Add explicit validation tooling that checks whether multiple sweeps appear
consistent in one shared world frame before they are fused or compared.

## Why This Matters

In theory, if the NeRF has been trained on multiple sweeps successfully, the
poses used for training should already be in a common aligned frame.

In practice, visualization should not trust that assumption blindly. A small
misregistration between sweeps can make:

- fused volumes look blurred or doubled
- trajectories appear offset
- nearest-frame comparison choose misleading matches
- interactive probe placement feel incorrect

The toolkit therefore needs a validation step that makes alignment quality
visible before the user interprets the scene.

## Required Work

- Define what "alignment validation" means for this toolkit.
- Add numeric checks that can flag obvious cross-sweep inconsistencies.
- Candidate checks include:
  - trajectory overlap statistics
  - bounding-box overlap between sweeps
  - relative pose distribution checks
  - frame-support overlap in fused voxel space
  - nearest-neighbor distance between sampled probe centers
- Provide outputs that are useful to a developer and to a visualization user:
  - per-sweep summary
  - pairwise sweep summary
  - warnings for suspicious offsets
- Distinguish between:
  - alignment assumed from input data
  - alignment validated by heuristics
  - alignment corrected by explicit transforms
- Define thresholds carefully and keep them configurable. The point is to warn
  about suspicious misalignment, not to claim absolute registration quality.
- Surface validation results in a form later UI tickets can display.

## Suggested Implementation

- Add a module such as:
  - `visualization/alignment_validation.py`
- Represent results as structured objects rather than free-form strings.
- Start with robust geometric checks using probe centers and sweep bounds before
  attempting more image-content-based checks.
- Keep the API usable from both:
  - CLI validation commands
  - viewer initialization

## What Needs To Be Checked

- Aligned synthetic test inputs pass without false alarms.
- Intentionally shifted sweeps trigger warnings.
- Validation output identifies which sweep pair is suspicious.
- Unit assumptions remain millimeters end-to-end.

## Output of This Ticket

- Cross-sweep validation metrics and warnings.
- Structured validation results suitable for CLI and GUI display.
- Tests using mocked aligned and intentionally misaligned sweeps.

## Acceptance Criteria

- The toolkit can report whether loaded sweeps appear plausibly aligned.
- Validation can run before fusion or interactive viewing.
- The output is specific enough to help diagnose which sweep is offset.

## Dependencies

- T24
- T25

## Blocks

- T28
- T29
- T30
- T31
