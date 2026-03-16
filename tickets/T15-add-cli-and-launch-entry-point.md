# T15 - Add CLI and Launch Entry Point

## Goal

Provide a clean command-line entry point for launching the visualization
application without code edits.

## Why This Matters

A visualization feature is not complete if it requires a developer to edit
Python files manually to launch it. This ticket turns the underlying components
into a usable application entry point.

## Required Work

1. Create a launch script.
   Suggested name:
   - `run_visualize_sweeps.py`

2. Support command-line arguments for:
   - dataset directory
   - checkpoint path
   - config path
   - cached volume path
   - device selection
   - optional rendering mode flags

3. Decide launch behavior when a cached volume is missing.
   Options:
   - build automatically
   - prompt user
   - fail with clear message

4. Add clear logging on startup.

5. Document the command in the README after implementation.

## What Needs To Be Checked

- Launch works from a clean terminal session
- Missing paths produce actionable errors
- Startup order is deterministic and understandable

## Output of This Ticket

- A single documented command for starting the visualization workflow

## Acceptance Criteria

- A developer can start the visualization app by running one command
- The command works without modifying source code

## Dependencies

- T05
- T10
- T11
- optionally T12, depending on the desired first release scope

## Blocks

- T17
