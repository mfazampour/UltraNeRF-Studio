# T17 - Add Documentation and User Guide

## Goal

Document the new visualization workflow so other developers and end users can
launch it, understand the controls, and extend it safely.

## Why This Matters

The visualization branch will touch geometry, UI, model inference, and data
fusion. Without documentation, future developers will struggle to extend or even
use the feature correctly.

## Required Work

1. Update the main README with:
   - feature overview
   - dependencies
   - launch command
   - expected inputs

2. Add a dedicated visualization architecture document.
   Suggested file:
   - `docs/VISUALIZATION_ARCHITECTURE.md`

3. Document the coordinate conventions from T01.

4. Document the GUI workflow:
   - loading data
   - placing the probe
   - rotating the image plane
   - viewing NeRF output
   - comparison workflow

5. Add troubleshooting notes.
   Include common failure cases:
   - no cached volume
   - wrong checkpoint
   - misaligned coordinates
   - slow rendering

## What Needs To Be Checked

- Documentation matches actual CLI flags and file names
- Screenshots or figures are added if helpful
- Developer-facing docs and user-facing docs are both present

## Output of This Ticket

- A documented visualization workflow
- Lower onboarding cost for future contributors

## Acceptance Criteria

- A new developer can launch and use the feature from the docs alone
- The architecture and coordinate assumptions are documented in one place

## Dependencies

- T01
- T15

## Blocks

- None
