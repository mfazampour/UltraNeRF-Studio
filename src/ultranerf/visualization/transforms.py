"""Coordinate transforms for the visualization stack.

The visualization layer uses millimeters as its canonical spatial unit. These
helpers convert among image-space, probe-local, world, and voxel coordinates
without depending on any GUI framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ProbeGeometry:
    """Physical probe geometry in millimeters."""

    width_mm: float
    depth_mm: float


@dataclass(frozen=True)
class VolumeGeometry:
    """Voxel volume geometry in millimeters."""

    origin_mm: np.ndarray
    spacing_mm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_mm", np.asarray(self.origin_mm, dtype=np.float32))
        object.__setattr__(self, "spacing_mm", np.asarray(self.spacing_mm, dtype=np.float32))
        if self.origin_mm.shape != (3,):
            raise ValueError("origin_mm must have shape (3,)")
        if self.spacing_mm.shape != (3,):
            raise ValueError("spacing_mm must have shape (3,)")
        if np.any(self.spacing_mm <= 0):
            raise ValueError("spacing_mm must be strictly positive")


def as_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a homogeneous coordinate of 1 to point rows."""
    pts = np.asarray(points, dtype=np.float32)
    pts = np.atleast_2d(pts)
    if pts.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return np.concatenate([pts, ones], axis=-1)


def ensure_pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Return a 4x4 pose matrix from a 3x4 or 4x4 input pose."""
    pose_arr = np.asarray(pose, dtype=np.float32)
    if pose_arr.shape == (4, 4):
        return pose_arr
    if pose_arr.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return np.concatenate([pose_arr, bottom], axis=0)
    raise ValueError("pose must have shape (3, 4) or (4, 4)")


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Invert a probe-to-world pose matrix."""
    pose_44 = ensure_pose_matrix(pose)
    rot = pose_44[:3, :3]
    trans = pose_44[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def pixel_to_probe_local(
    row: np.ndarray | float,
    col: np.ndarray | float,
    image_shape: Tuple[int, int],
    geometry: ProbeGeometry,
) -> np.ndarray:
    """Map image pixels to probe-local coordinates in millimeters.

    The probe-local frame uses:
    - x: lateral direction centered on the probe face
    - y: beam/depth direction, starting at zero on the probe face
    - z: elevation / out-of-plane, fixed at zero for a 2D scan plane

    Pixel centers are used for the conversion.
    """
    height, width = image_shape
    row_arr = np.asarray(row, dtype=np.float32)
    col_arr = np.asarray(col, dtype=np.float32)
    lateral = ((col_arr + 0.5) / float(width) - 0.5) * geometry.width_mm
    depth = ((row_arr + 0.5) / float(height)) * geometry.depth_mm
    zeros = np.zeros_like(lateral, dtype=np.float32)
    return np.stack([lateral, depth, zeros], axis=-1).astype(np.float32)


def probe_local_to_world(points_mm: np.ndarray, pose_probe_to_world: np.ndarray) -> np.ndarray:
    """Transform probe-local points in millimeters into world coordinates."""
    pose = ensure_pose_matrix(pose_probe_to_world)
    pts_h = as_homogeneous(points_mm)
    world = (pose @ pts_h.T).T
    return world[:, :3].astype(np.float32)


def world_to_probe_local(points_mm: np.ndarray, pose_probe_to_world: np.ndarray) -> np.ndarray:
    """Transform world points in millimeters into probe-local coordinates."""
    inv_pose = invert_pose(pose_probe_to_world)
    pts_h = as_homogeneous(points_mm)
    probe_local = (inv_pose @ pts_h.T).T
    return probe_local[:, :3].astype(np.float32)


def world_to_voxel(points_mm: np.ndarray, geometry: VolumeGeometry) -> np.ndarray:
    """Map world coordinates in millimeters into floating voxel coordinates."""
    points = np.asarray(points_mm, dtype=np.float32)
    points = np.atleast_2d(points)
    if points.shape[-1] != 3:
        raise ValueError("points_mm must have shape (..., 3)")
    return ((points - geometry.origin_mm) / geometry.spacing_mm).astype(np.float32)


def voxel_to_world(voxel_coords: np.ndarray, geometry: VolumeGeometry) -> np.ndarray:
    """Map floating voxel coordinates into world coordinates in millimeters."""
    voxels = np.asarray(voxel_coords, dtype=np.float32)
    voxels = np.atleast_2d(voxels)
    if voxels.shape[-1] != 3:
        raise ValueError("voxel_coords must have shape (..., 3)")
    return (geometry.origin_mm + voxels * geometry.spacing_mm).astype(np.float32)


def probe_plane_corners(
    pose_probe_to_world: np.ndarray,
    geometry: ProbeGeometry,
) -> np.ndarray:
    """Return world-space corners of the scan plane rectangle.

    Corner order:
    - top-left
    - top-right
    - bottom-right
    - bottom-left
    """
    half_width = geometry.width_mm / 2.0
    local_corners = np.array(
        [
            [-half_width, 0.0, 0.0],
            [half_width, 0.0, 0.0],
            [half_width, geometry.depth_mm, 0.0],
            [-half_width, geometry.depth_mm, 0.0],
        ],
        dtype=np.float32,
    )
    return probe_local_to_world(local_corners, pose_probe_to_world)


def pose_to_axes(pose_probe_to_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return probe origin and local world-space axes from a pose.

    Returns:
    - origin_mm
    - x_axis_world
    - y_axis_world
    - z_axis_world
    """
    pose = ensure_pose_matrix(pose_probe_to_world)
    origin = pose[:3, 3].astype(np.float32)
    x_axis = pose[:3, 0].astype(np.float32)
    y_axis = pose[:3, 1].astype(np.float32)
    z_axis = pose[:3, 2].astype(np.float32)
    return origin, x_axis, y_axis, z_axis
