# T35 - Add End-to-End Multi-Sweep QA and Example Launch Workflows

## Goal

Close the loop on the multi-sweep feature with runnable examples and a final QA
pass that reflects the integrated viewer, not just the backend.

## Why This Matters

Backend tests and design docs are necessary, but once the app wiring exists the
project also needs:

- example launch commands
- example manifests
- a manual QA flow for the actual viewer
- a clear record of what to verify before trusting a multi-sweep scene

This is especially important because multi-sweep visualization depends on
alignment assumptions that should always be checked explicitly.

## Required Work

- Add example launch commands for:
  - headless validation
  - full GUI launch
  - checkpoint-backed multi-sweep rendering, if available
- Add or document at least one example manifest shape that matches the actual
  launcher implementation.
- Add a final manual QA checklist for:
  - alignment warnings
  - per-sweep visibility
  - aggregate fusion
  - comparison policy switching
  - matched sweep reporting
  - probe behavior in multi-sweep mode
- Update docs so the launcher and the QA workflow are in one discoverable place.

## What Needs To Be Checked

- The documented commands match the actual CLI.
- The example manifest is valid.
- A developer can follow the QA checklist end to end without reading code.

## Output of This Ticket

- Final integrated multi-sweep workflow docs.
- Example commands and example manifest.
- Manual QA checklist for the live viewer.

## Acceptance Criteria

- Another developer can run and evaluate the multi-sweep viewer from the docs.
- The docs clearly require cross-sweep alignment inspection before interpreting
  fused or comparative results.

## Dependencies

- T32
- T33
- T34
