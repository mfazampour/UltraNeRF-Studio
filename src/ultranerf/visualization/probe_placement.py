"""Probe placement helpers driven by MPR selections."""

from __future__ import annotations

import numpy as np

from ultranerf.visualization.mpr import MPRSelection
from ultranerf.visualization.transforms import ensure_pose_matrix


def pose_from_rotation_and_origin(rotation: np.ndarray, origin_mm: np.ndarray) -> np.ndarray:
    """Construct a 4x4 probe-to-world pose from rotation and origin."""
    rot = np.asarray(rotation, dtype=np.float32)
    origin = np.asarray(origin_mm, dtype=np.float32).reshape(3)
    if rot.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = origin
    return pose


def set_pose_origin(pose_probe_to_world: np.ndarray, origin_mm: np.ndarray) -> np.ndarray:
    """Return a copy of `pose_probe_to_world` with an updated origin."""
    pose = ensure_pose_matrix(pose_probe_to_world).copy()
    pose[:3, 3] = np.asarray(origin_mm, dtype=np.float32).reshape(3)
    return pose


def default_probe_rotation() -> np.ndarray:
    """Default probe rotation aligned with the canonical visualization axes."""
    return np.eye(3, dtype=np.float32)


def probe_pose_from_mpr_selection(
    selection: MPRSelection,
    *,
    current_pose: np.ndarray | None = None,
    default_rotation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Create a probe pose from an MPR selection.

    Placement updates the probe center while keeping the current orientation if
    available. If no current pose is provided, a default rotation is used.
    """
    origin = np.asarray(selection.world_point_mm, dtype=np.float32).reshape(3)
    if current_pose is not None:
        return set_pose_origin(current_pose, origin)

    rotation = (
        np.asarray(default_rotation_matrix, dtype=np.float32)
        if default_rotation_matrix is not None
        else default_probe_rotation()
    )
    return pose_from_rotation_and_origin(rotation, origin)
