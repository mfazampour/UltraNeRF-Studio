# T25 - Add Multi-Sweep Dataset Loading and Manifest Support

## Goal

Add a loading path that can ingest multiple sweeps as one visualization
session.

The output should be a `MultiSweepScene`-style object rather than a single
`images.npy` / `poses.npy` pair.

## Why This Matters

Single-sweep loading is not sufficient once the dataset contains multiple
tracked acquisitions captured from different angles or sessions.

The visualization toolkit needs one consistent way to answer:

- which sweeps belong to this scene
- where their files live
- what metadata applies to each sweep
- whether any per-sweep world transform should be applied

Without a manifest-driven loading layer, every multi-sweep experiment will end
up with custom ad hoc scripts and brittle assumptions.

## Required Work

- Define a manifest format for multi-sweep sessions.
- The manifest should support, at minimum, for each sweep:
  - sweep id
  - dataset directory
  - image path override, if not default
  - pose path override, if not default
  - optional registration transform path or inline transform
  - optional display color
  - optional display name
  - optional enabled-by-default flag
- Support at least two loading modes:
  - explicit manifest file
  - directory-of-sweeps convention, if that pattern exists locally
- Reuse the existing loading code where appropriate rather than duplicating
  frame parsing logic.
- Preserve the current single-sweep path for backward compatibility.
- Validate that all loaded sweeps have enough metadata to participate in one
  scene:
  - images and poses exist
  - pose counts match image counts
  - probe metadata is either shared or explicitly overridden
- Decide how the loader handles mismatches in:
  - image size
  - probe width
  - probe depth
  - missing transforms

## Suggested Implementation

- Add a manifest parser module such as:
  - `visualization/multi_sweep_loader.py`
- Use a simple, readable format such as JSON or YAML.
- Include a repo example manifest under `docs/` or `data/` once the format is
  stable.
- Return structured objects from T24 rather than raw dicts.

## What Needs To Be Checked

- Loading one sweep through the multi-sweep path still works.
- Loading several sweeps produces the correct sweep count and metadata.
- Missing files fail early with readable errors.
- Manifest transforms are parsed with the correct unit assumptions.
- Backward compatibility with the existing single-sweep CLI remains intact.

## Output of This Ticket

- A multi-sweep manifest format.
- A loader that returns structured multi-sweep scene state.
- Tests for manifest parsing and loading validation.

## Acceptance Criteria

- A developer can point the toolkit at a manifest and get a usable multi-sweep
  scene object.
- File validation errors are clear.
- Single-sweep workflows remain supported.

## Dependencies

- T24

## Blocks

- T26
- T28
- T29
- T30
- T31
