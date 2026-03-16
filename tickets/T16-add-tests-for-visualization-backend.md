# T16 - Add Tests for Visualization Backend

## Goal

Add automated tests for the non-GUI visualization backend so the feature is
maintainable and robust.

## Why This Matters

Interactive viewers are hard to test directly, but the spatial math and data
fusion behind them are testable. This ticket protects the most failure-prone
parts of the system.

## Required Work

1. Add tests for transform utilities:
   - pixel to world
   - world to voxel
   - inverse consistency

2. Add tests for sweep fusion:
   - synthetic single-slice fusion
   - multi-slice sparse coverage
   - bounds and spacing correctness

3. Add tests for caching:
   - save / load round trip
   - metadata integrity

4. Add tests for arbitrary-pose NeRF session API:
   - checkpoint loading path
   - pose input shape validation
   - output dictionary structure

5. Add lightweight smoke tests for viewer setup where practical, but keep GUI
   dependencies minimal.

## What Needs To Be Checked

- Tests do not depend on large real datasets
- Tests remain CPU-safe where possible
- Spatial correctness is validated with deterministic fixtures

## Output of This Ticket

- A robust test suite for the backend of the visualization feature

## Acceptance Criteria

- Core backend modules are covered by deterministic automated tests
- Major regressions in coordinate handling or fusion would fail CI

## Dependencies

- T02
- T03
- T04
- T10

## Blocks

- None
