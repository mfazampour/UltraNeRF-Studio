"""Backend helpers for synchronized MPR state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from visualization.transforms import VolumeGeometry, voxel_to_world, world_to_voxel


MPRViewName = Literal["sagittal", "coronal", "axial"]


@dataclass(frozen=True)
class MPRSelection:
    """Linked MPR selection in world and voxel coordinates."""

    world_point_mm: np.ndarray
    voxel_point: np.ndarray
    voxel_indices: tuple[int, int, int]


def clamp_voxel_point(voxel_point: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Clamp floating voxel coordinates to the valid volume extent."""
    point = np.asarray(voxel_point, dtype=np.float32)
    if point.shape != (3,):
        raise ValueError("voxel_point must have shape (3,)")
    max_index = np.asarray(volume_shape, dtype=np.float32) - 1.0
    return np.clip(point, 0.0, max_index).astype(np.float32)


def selection_from_world_point(
    world_point_mm: np.ndarray,
    geometry: VolumeGeometry,
    volume_shape: tuple[int, int, int],
) -> MPRSelection:
    """Create a synchronized MPR selection from a world-space point."""
    voxel_point = world_to_voxel(world_point_mm, geometry).reshape(3)
    clamped_voxel = clamp_voxel_point(voxel_point, volume_shape)
    indices = tuple(int(round(v)) for v in clamped_voxel)
    world_point = voxel_to_world(np.array(indices, dtype=np.float32), geometry).reshape(3)
    return MPRSelection(
        world_point_mm=world_point.astype(np.float32),
        voxel_point=clamped_voxel,
        voxel_indices=indices,
    )


def update_selection_for_view_click(
    current_selection: MPRSelection,
    *,
    view: MPRViewName,
    first_axis_value: float,
    second_axis_value: float,
    geometry: VolumeGeometry,
    volume_shape: tuple[int, int, int],
) -> MPRSelection:
    """Update the linked selection from a click in one orthogonal MPR view.

    Axis mapping for the internal volume order `[X, Y, Z]`:
    - sagittal: click on `(Y, Z)` while keeping `X` fixed
    - coronal: click on `(X, Z)` while keeping `Y` fixed
    - axial: click on `(X, Y)` while keeping `Z` fixed
    """
    voxel_point = np.array(current_selection.voxel_point, dtype=np.float32)
    if view == "sagittal":
        voxel_point[1] = first_axis_value
        voxel_point[2] = second_axis_value
    elif view == "coronal":
        voxel_point[0] = first_axis_value
        voxel_point[2] = second_axis_value
    elif view == "axial":
        voxel_point[0] = first_axis_value
        voxel_point[1] = second_axis_value
    else:
        raise ValueError(f"Unknown MPR view: {view}")
    clamped = clamp_voxel_point(voxel_point, volume_shape)
    return selection_from_world_point(voxel_to_world(clamped, geometry).reshape(3), geometry, volume_shape)


def orthogonal_slice_indices(selection: MPRSelection) -> dict[str, int]:
    """Return the active slice index for each orthogonal MPR view."""
    x_idx, y_idx, z_idx = selection.voxel_indices
    return {
        "sagittal": x_idx,
        "coronal": y_idx,
        "axial": z_idx,
    }
