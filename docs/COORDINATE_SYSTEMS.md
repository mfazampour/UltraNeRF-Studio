# Coordinate Systems

This document defines the coordinate conventions for the 3D visualization work.

The key decision is:

- the visualization stack uses **millimeters**
- the current training and rendering code internally uses **meters** in several
  places because `load_us.py` scales pose translation by `0.001`

Any visualization code that calls the existing NeRF renderer should therefore
perform a single explicit conversion from millimeters to meters at the inference
boundary.

## Why Millimeters

The visualization and interactive probe workflow should use millimeters because:

- the scenes are spatially small
- medical imaging viewers typically use millimeters
- users think about probe placement and anatomy in millimeters
- volume origin, spacing, and MPR coordinates are easier to interpret in
  millimeters than in small meter-scale values

## Canonical Spaces

The visualization stack will explicitly work with these spaces:

1. image space
2. probe-local space
3. world space
4. voxel space
5. NeRF inference space

## 1. Image Space

Source arrays are loaded as 2D grayscale images with shape `[H, W]`.

For visualization purposes:

- `H` corresponds to **depth samples** along the ultrasound beam
- `W` corresponds to **lateral samples** across the probe face

Image indexing follows array convention:

- row index increases downward
- column index increases to the right

The pixel grid itself should not be treated as world coordinates. Image-space
locations must first be mapped into probe-local coordinates using the physical
probe dimensions.

## 2. Probe-Local Space

Probe-local space is the intrinsic frame attached to the ultrasound probe and
scan plane.

For consistency with the existing renderer in `get_rays_us_linear()`:

- local **x**: lateral direction across probe width
- local **y**: beam direction / depth direction
- local **z**: out-of-plane elevation direction

The probe-local origin for visualization should be defined as:

- the center of the probe face
- at depth `0`

That means:

- lateral coordinate range is centered around `0`
- depth starts at `0` and increases along positive local `y`

For a probe width `probe_width_mm` and probe depth `probe_depth_mm`:

- lateral spans approximately `[-probe_width_mm / 2, +probe_width_mm / 2]`
- depth spans approximately `[0, probe_depth_mm]`

This matches the current rendering code’s lateral centering and forward beam
direction.

## 3. World Space

World space is the shared 3D frame in which:

- tracked probe poses live
- sweep volumes are stored
- MPR interaction is defined
- the interactive virtual probe is placed

For visualization, world-space units are **millimeters**.

All visualization-facing APIs should accept and return millimeter values unless
explicitly documented otherwise.

## Pose Semantics

The rendering path currently passes pose matrices into `render_us()` as `c2w`
style transforms. In practice, the pose is treated as a transform whose:

- rotation describes the probe-local axes in world space
- translation gives the probe origin in world space

For visualization planning, we treat `poses.npy` as **probe-to-world** poses.

This assumption should be validated against:

- `get_rays_us_linear()`
- `render_us()`
- how the tracked data was generated

If later validation shows a different interpretation, the transform utilities
should be updated in one place, but the visualization APIs should still expose a
single consistent pose meaning.

## 4. Voxel Space

Voxel space indexes the fused sweep volume.

Voxel coordinates are integer index-like coordinates into the volume array.
Voxel space is not user-facing; it exists only to store sampled sweep data and
support volume rendering / MPR views.

A fused volume must carry:

- `origin_mm`: world-space location of voxel `(0, 0, 0)` in millimeters
- `spacing_mm`: voxel spacing in millimeters
- `shape`: volume shape in index coordinates

World-to-voxel mapping should follow:

- `voxel = (world_mm - origin_mm) / spacing_mm`

with axis ordering documented in the implementation.

## 5. NeRF Inference Space

The current NeRF training code does not operate in the same user-facing unit
system as the visualization stack.

Important existing behavior:

- `load_us.py` scales pose translations from millimeters to meters
- probe width and probe depth are converted with a `0.001` factor in training
  scripts

Therefore:

- viewer state should remain in millimeters
- cached volumes should remain in millimeters
- MPR interaction should remain in millimeters
- interactive probe placement should remain in millimeters
- only the NeRF inference adapter should convert pose and probe geometry into
  meter-scale values if the existing runtime requires it

This conversion boundary should be centralized in the future
`visualization/nerf_session.py` implementation.

## Pixel-to-World Mapping

For a pixel `(row, col)` in an image of shape `[H, W]`:

1. compute lateral position from column index using `probe_width_mm`
2. compute depth position from row index using `probe_depth_mm`
3. set probe-local point:
   - `x = lateral_mm`
   - `y = depth_mm`
   - `z = 0`
4. transform that point into world coordinates with the probe pose

The exact half-pixel convention should be fixed in the transform utilities and
applied consistently in:

- sweep fusion
- probe plane generation
- NeRF render alignment

## Practical Rules for Developers

- Use **mm everywhere** in the visualization code unless explicitly crossing
  into the current NeRF runtime.
- Convert to **m only once**, inside the inference adapter.
- Treat the probe pose as **probe-to-world** in visualization code.
- Treat `W` as lateral and `H` as depth.
- Keep the probe-local beam direction aligned with positive local `y`.
- Store volume origin and spacing explicitly with every fused volume.

## Open Validation Items

These should be verified early during implementation:

1. Confirm that `poses.npy` is correctly interpreted as probe-to-world.
2. Confirm that the chosen image-depth/lateral mapping matches the training
   renderer orientation.
3. Confirm that probe-local origin at the centered probe face matches the
   tracked data generation assumptions closely enough for visualization.

## Minimal Example

Suppose:

- `probe_width_mm = 80`
- `probe_depth_mm = 140`
- image shape is `[140, 80]`
- pose translation is `(10, 20, 30)` mm in world space
- pose rotation is identity

Then:

- center column maps near `x = 0` mm
- top row maps near `y = 0` mm
- bottom row maps near `y = 140` mm
- the image center projects near world point `(10, 90, 30)` mm

This kind of example should be reproduced in unit tests once the transform
utilities are implemented.
