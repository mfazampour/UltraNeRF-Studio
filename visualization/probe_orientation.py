"""Orientation control helpers for the virtual probe."""

from __future__ import annotations

import numpy as np

from visualization.probe_placement import pose_from_rotation_and_origin
from visualization.transforms import ensure_pose_matrix


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    rot = np.asarray(rotation, dtype=np.float32)
    if rot.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    u, _, vt = np.linalg.svd(rot)
    corrected = u @ vt
    if np.linalg.det(corrected) < 0:
        u[:, -1] *= -1.0
        corrected = u @ vt
    return corrected.astype(np.float32)


def rotation_matrix_from_yaw_pitch_roll(
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Create a rotation matrix from yaw/pitch/roll in Z-Y-X order.

    Conventions:
    - yaw: rotation around world/probe z
    - pitch: rotation around world/probe y
    - roll: rotation around world/probe x
    """
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))
    rotation = rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)
    return orthonormalize_rotation(rotation)


def pose_from_yaw_pitch_roll(
    origin_mm: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Construct a probe pose from an origin and yaw/pitch/roll angles."""
    rotation = rotation_matrix_from_yaw_pitch_roll(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
    return pose_from_rotation_and_origin(rotation, origin_mm)


def yaw_pitch_roll_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, and roll in degrees from a Z-Y-X rotation matrix."""
    rot = orthonormalize_rotation(rotation)
    pitch_rad = float(np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0)))
    cos_pitch = float(np.cos(pitch_rad))
    if abs(cos_pitch) > 1e-6:
        yaw_rad = float(np.arctan2(rot[1, 0], rot[0, 0]))
        roll_rad = float(np.arctan2(rot[2, 1], rot[2, 2]))
    else:
        yaw_rad = float(np.arctan2(-rot[0, 1], rot[1, 1]))
        roll_rad = 0.0
    return (
        float(np.rad2deg(yaw_rad)),
        float(np.rad2deg(pitch_rad)),
        float(np.rad2deg(roll_rad)),
    )


def pose_to_yaw_pitch_roll(pose_probe_to_world: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, and roll in degrees from a probe pose."""
    pose = ensure_pose_matrix(pose_probe_to_world)
    return yaw_pitch_roll_from_rotation_matrix(pose[:3, :3])


def update_probe_pose_orientation(
    pose_probe_to_world: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Update the orientation of a probe pose while preserving its origin."""
    pose = ensure_pose_matrix(pose_probe_to_world)
    origin = pose[:3, 3].copy()
    return pose_from_yaw_pitch_roll(origin, yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
