# Visualization Tickets

This folder breaks the 3D sweep visualization and interactive probe rendering
work into implementation-sized tickets.

The intended end state is:

- tracked 2D ultrasound sweeps fused into a 3D volume
- transfer-function-based 3D visualization of the input data
- synchronized MPR views
- interactive probe placement and orientation control
- live NeRF rendering from arbitrary probe poses

## Recommended Delivery Order

1. [T01 - Define Coordinate Conventions](T01-define-coordinate-conventions.md)
2. [T02 - Build Transform Utilities](T02-build-transform-utilities.md)
3. [T03 - Fuse Sweeps Into a 3D Volume](T03-fuse-sweeps-into-volume.md)
4. [T04 - Add Volume Metadata and Caching](T04-add-volume-metadata-and-caching.md)
5. [T05 - Add a Basic 3D Volume Viewer](T05-add-basic-3d-volume-viewer.md)
6. [T06 - Add Synchronized MPR Views](T06-add-synchronized-mpr-views.md)
7. [T07 - Add Probe Representation](T07-add-probe-representation.md)
8. [T08 - Add Probe Placement From MPR](T08-add-probe-placement-from-mpr.md)
9. [T09 - Add Probe Orientation Controls](T09-add-probe-orientation-controls.md)
10. [T10 - Wrap NeRF Inference for Arbitrary Poses](T10-wrap-nerf-inference-for-arbitrary-poses.md)
11. [T11 - Connect Viewer Pose to NeRF Rendering](T11-connect-viewer-pose-to-nerf-rendering.md)
12. [T12 - Add Comparison View](T12-add-comparison-view.md)
13. [T13 - Add Trajectory Overlay](T13-add-trajectory-overlay.md)
14. [T14 - Add Transfer-Function Presets](T14-add-transfer-function-presets.md)
15. [T15 - Add CLI and Launch Entry Point](T15-add-cli-and-launch-entry-point.md)
16. [T16 - Add Tests for Visualization Backend](T16-add-tests-for-visualization-backend.md)
17. [T17 - Add Documentation and User Guide](T17-add-documentation-and-user-guide.md)

## Dependency Overview

- T01 is required before all geometry-heavy tickets.
- T02 depends on T01.
- T03 depends on T01 and T02.
- T04 depends on T03.
- T05 depends on T03 and T04.
- T06 depends on T05.
- T07 depends on T01 and T05.
- T08 depends on T06 and T07.
- T09 depends on T07.
- T10 depends on T01 and the existing rendering pipeline.
- T11 depends on T08, T09, and T10.
- T12 depends on T11.
- T13 depends on T05 and T07.
- T14 depends on T05.
- T15 depends on T05 through T11, depending on what is exposed in the first release.
- T16 should be added alongside T02, T03, T04, and T10, but can be tracked as a dedicated hardening ticket.
- T17 should be updated continuously, but closes after the feature becomes usable.

## Suggested Milestones

### Milestone 1: Static Sweep Visualization

- T01
- T02
- T03
- T04
- T05
- T13
- T14

### Milestone 2: Interactive Probe Placement

- T06
- T07
- T08
- T09

### Milestone 3: Live NeRF Interaction

- T10
- T11
- T12
- T15

### Milestone 4: Hardening and Documentation

- T16
- T17
