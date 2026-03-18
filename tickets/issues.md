# Visualization Issues

## Issue 1: 2D Pixel Size / Aspect Ratio Is Not Respected
- Status: Fixed
- Problem: The visualization does not preserve the physical pixel spacing or aspect ratio of the 2D ultrasound frames.
- Expected behavior: The displayed 2D frames should use the correct spatial scaling so anatomy is not stretched or compressed.
- Resolution: The multi-view napari image panels now receive per-image physical pixel scales derived from probe width/depth and image shape.

## Issue 2: Move Multi-Sweep Controls to the Right Side
- Status: Fixed
- Problem: The `Multi-Sweep Controls` panel is currently placed on the left side and contributes to crowding there.
- Expected behavior: Move this panel to the right side, or otherwise rebalance the layout so the left side is less overloaded.
- Resolution: `Multi-Sweep Controls` now dock on the right side.

## Issue 3: Duplicate Aggregate Volumes Are Shown
- Status: Fixed
- Problem: Two aggregate volumes appear in the viewer, both named `sweep_volume__aggregate`.
- Expected behavior: Only one aggregate volume should be created, and its layer naming should be unambiguous.
- Resolution: The controller now reuses an existing aggregate layer instead of creating a second one during scene refresh.

## Issue 4: Add Profiling / Timing Logs for Slow Startup
- Status: Fixed
- Problem: The application still takes a long time before the napari window becomes usable, and it is not clear which stage is the bottleneck.
- Expected behavior: Add logging or profiling output for the major startup stages, such as volume fusion, layer creation, and UI initialization. Make sure these logs are accessible to later check and fix the issues.
- Resolution: Multi-sweep startup now records stage timings to a JSON log under `logs/visualization/profiling/`, and the launcher summary reports the log path and timing map.

## Issue 5: Remove Frame Coordinate Axes
- Status: Fixed
- Problem: The coordinate axes currently shown for frames add visual clutter and are not useful in practice.
- Expected behavior: Remove these frame axes from the default visualization.
- Resolution: Per-frame trajectory axes and per-frame center markers are no longer created in the default multi-sweep scene; only the trajectory path remains.

## Issue 6: `Snap To Nearest` Does Not Search Across All Sweeps
- Status: Fixed
- Problem: `Snap To Nearest` only considers the active sweep when looking for the nearest recorded frame, even when comparison mode is set to `All enabled sweeps`.
- Expected behavior: When `All enabled sweeps` is selected, nearest-frame lookup should consider all enabled sweeps rather than only the active one.
- Resolution: `Snap To Nearest` now uses the same multi-sweep comparison policy and enabled-sweep set as the comparison panel, and switches the active sweep when the nearest pose comes from another sweep.
