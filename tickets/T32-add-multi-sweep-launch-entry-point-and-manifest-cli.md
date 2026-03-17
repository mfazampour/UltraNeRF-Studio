# T32 - Add a Multi-Sweep Launch Entry Point and Manifest CLI

## Goal

Expose the multi-sweep backend through a user-facing launch path.

This ticket should add a CLI that accepts a multi-sweep manifest and prepares a
live visualization session without requiring a developer to import backend
modules manually.

## Why This Matters

The backend now supports:

- multi-sweep scene types
- manifest loading
- alignment validation
- per-sweep transforms
- aggregate fusion
- sweep-aware comparison

But none of that is available from a direct launch command yet. Until a
multi-sweep CLI exists, the feature is still effectively internal-only.

## Required Work

- Add a new launch path for multi-sweep sessions.
- Decide whether to:
  - extend `run_visualize_sweeps.py`
  - or add a dedicated `run_visualize_multi_sweep.py`
- Accept, at minimum:
  - manifest path
  - cache path or cache root
  - spacing
  - pixel stride
  - initial active sweep
  - optional checkpoint/config for NeRF-backed rendering
- Run alignment validation before the viewer launches and report its summary.
- Make the summary visible in:
  - headless mode
  - GUI launch logs
- Preserve the current single-sweep CLI behavior unchanged.

## What Needs To Be Checked

- A valid manifest launches successfully.
- Headless mode prints enough metadata to debug a scene before opening napari.
- Missing manifest fields fail with readable messages.
- Alignment warnings appear before the scene is trusted visually.

## Output of This Ticket

- A multi-sweep launch command.
- Manifest-driven scene preparation.
- Tests covering argument parsing and headless launch summaries.

## Acceptance Criteria

- A developer can launch a multi-sweep session from the command line.
- The CLI clearly reports whether alignment validation passed or warned.
- Single-sweep launch commands remain unchanged.

## Dependencies

- T24
- T25
- T26
- T27
- T28
- T29
- T30
- T31

## Blocks

- T33
- T34
- T35
