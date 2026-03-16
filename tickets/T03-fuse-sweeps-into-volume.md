# T03 - Fuse Sweeps Into a 3D Volume

## Goal

Convert tracked 2D ultrasound sweeps into a dense 3D scalar volume that can be
visualized with transfer functions and MPR views.

## Why This Matters

The user-facing visualization goal is to see the input data itself in 3D. That
requires converting the tracked 2D slices into a world-aligned volume. This is
the core backend for the 3D sweep viewer.

## Required Work

1. Create a reusable sweep fusion module.
   Suggested file:
   - `visualization/sweep_volume.py`

2. Load:
   - `images.npy`
   - `poses.npy`
   - probe geometry parameters such as width and depth

3. For each image:
   - compute the world-space position of each sampled pixel or a configurable
     subset of sampled pixels
   - map those points into a target voxel grid
   - accumulate intensity values into the grid

4. Implement a weighted accumulation strategy.
   At minimum:
   - sum of intensities
   - per-voxel hit count or weight accumulation
   - normalized output volume `sum / weight`

5. Support configurable volume definition:
   - explicit bounds
   - automatically derived bounds from sweep poses
   - configurable voxel spacing or resolution

6. Consider interpolation strategy.
   First version may use nearest-neighbor splatting, but the implementation
   should be structured so trilinear or Gaussian splatting can be added later.

7. Provide output as a data object containing:
   - volume array
   - weight array
   - origin
   - spacing
   - bounds

## Design Considerations

- This should be a backend module, not tied to a specific viewer
- It should support small test grids for unit tests
- It should be deterministic
- It should handle sparse sweep coverage gracefully

## What Needs To Be Checked

- The fused volume aligns with known tracked slice positions
- Intensities are not mirrored or transposed incorrectly
- World-to-voxel mapping is correct
- Empty regions remain distinguishable from low-intensity regions if needed
- Volume bounds cover all sweep samples

## Output of This Ticket

- A reusable fused 3D scalar volume from tracked sweeps
- Metadata needed for later rendering and MPR
- A small CLI or function entry point for generating volumes offline

## Acceptance Criteria

- A synthetic sweep dataset produces a correctly positioned sparse volume
- A real dataset can be fused without viewer code
- The resulting volume can be loaded by a downstream visualization module

## Dependencies

- T01
- T02

## Blocks

- T04
- T05
- T06
