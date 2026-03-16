# T19 - Add a Rendered Output Panel to the Napari App

## Goal

Add a visible rendered-output panel to the napari-based visualization app so the
current virtual probe pose can produce a 2D NeRF-rendered ultrasound image that
the user can inspect.

## Why This Matters

The current implementation already has:

- a virtual probe pose
- a backend render controller
- an arbitrary-pose NeRF wrapper

But the rendered output is not shown anywhere in the UI. Until there is a
dedicated panel for the predicted image, the interactive viewer does not expose
the core value of the feature.

## Required Work

- Decide how the rendered image should be displayed:
  - second napari viewer
  - dock widget with an image canvas
  - plugin-style side panel
- Keep the main 3D sweep viewer separate from the 2D output view so that 3D
  navigation does not interfere with image inspection.
- Add a clear label for the rendered image panel.
- Support at least one grayscale image output from the NeRF renderer.
- Normalize or scale the rendered output into a displayable range without
  changing the underlying numerical output contract.
- Handle the initial state when no render has yet been performed.
- Handle rendering failures cleanly so the viewer does not crash when one render
  call fails.

## Design Considerations

- The display path should accept whichever output key is treated as canonical by
  the current renderer, for example `intensity_map`.
- The image panel should be replaceable later if the team decides to move from
  plain napari widgets to a richer Qt panel.
- This ticket should avoid deep UI customization if a simple docked image view
  will solve the immediate problem.

## What Needs To Be Checked

- The image panel appears on launch when NeRF mode is enabled.
- The panel updates after a successful render.
- The image orientation is sensible and consistent across repeated renders.
- The grayscale range is visible and not saturated black or white.

## Output of This Ticket

- A visible rendered-image panel in the visualization app.
- A tested adapter between the render controller output and the UI image layer.

## Acceptance Criteria

- A user can launch the app with a checkpoint and see a 2D rendered image after
  triggering or requesting a render.
- The image updates in-place rather than creating a new layer every time.
- The panel remains stable when no checkpoint is configured.

## Dependencies

- T18

## Blocks

- T21
- T22
