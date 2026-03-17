# T31 - Add Multi-Sweep Workflow Docs and Manual QA

## Goal

Document how multi-sweep visualization is expected to work and define the
manual checks needed before trusting the output.

## Why This Matters

Multi-sweep scenes are more powerful than single-sweep scenes, but they are
also much easier to misread.

Even if the NeRF training data suggests that all sweeps should already align,
users still need a repeatable checklist for verifying:

- alignment between sweeps
- the meaning of active vs enabled sweeps
- the source of nearest-frame comparisons
- whether aggregate fusion is showing what they think it is showing

Without clear docs and QA guidance, the viewer may be technically correct but
still easy to misuse.

## Required Work

- Update the README or visualization docs with a dedicated multi-sweep section.
- Document:
  - manifest format
  - coordinate assumptions
  - alignment assumptions
  - optional per-sweep transforms
  - comparison policies
  - aggregate fusion behavior
- Add a manual QA checklist for multi-sweep sessions.
- The checklist should include:
  - loading a known aligned multi-sweep scene
  - confirming per-sweep trajectories overlap plausibly
  - checking alignment warnings from T26
  - toggling sweeps on and off
  - confirming nearest-frame metadata identifies the correct sweep
  - verifying aggregate fusion changes when enabled sweeps change
  - checking probe reset / snap behavior in active-sweep and global modes
- Add at least one example command for launching a multi-sweep session.
- If possible, add a small synthetic or mocked multi-sweep test fixture for
  automated and manual use.

## Suggested Implementation

- Put the workflow documentation somewhere a developer will actually read it:
  - `README.md`
  - `docs/`
  - or both
- Keep the QA checklist concrete and task-oriented.
- Include absolute expectations where possible rather than vague guidance.

## What Needs To Be Checked

- The docs match the actual CLI and UI behavior.
- The QA checklist is specific enough for another developer to follow without
  tribal knowledge.
- Alignment verification is treated as a required step, not an optional note.

## Output of This Ticket

- Multi-sweep documentation.
- Manual QA checklist.
- Example launch commands and, ideally, example manifests.

## Acceptance Criteria

- Another developer can launch and evaluate a multi-sweep session without
  reverse-engineering the implementation.
- The docs clearly state that cross-sweep alignment should be checked before
  interpreting fused or comparative visualizations.

## Dependencies

- T24
- T25
- T26
- T27
- T28
- T29
- T30
