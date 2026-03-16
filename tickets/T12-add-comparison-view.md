# T12 - Add Comparison View

## Goal

Show the rendered NeRF output together with a relevant real acquired frame or
other comparison signal.

## Why This Matters

Once arbitrary-pose rendering works, users will want context. A comparison view
helps answer whether the pose is near a real recorded frame and whether the NeRF
output is plausible.

## Required Work

1. Define a comparison strategy.
   Options include:
   - nearest tracked pose from `poses.npy`
   - exact tracked frame if the user selects one
   - optional difference view between real and rendered image

2. Implement a nearest-pose lookup.

3. Display:
   - NeRF output
   - matched real frame
   - optional pose distance metric

4. Optionally show additional comparison panels:
   - confidence
   - reflection
   - attenuation

## What Needs To Be Checked

- Nearest-pose matching is measured in a meaningful pose distance metric
- Displayed frame orientation matches rendered image orientation
- Comparison labels are clear to the user

## Output of This Ticket

- A side-by-side or tabbed comparison UI
- Better interpretability for interactive exploration

## Acceptance Criteria

- A user can see the rendered image and a matched real image in one workflow
- The matched real frame is explainable and reproducible

## Dependencies

- T11

## Blocks

- T15
