"""Geometric representation of the probe and scan plane for 3D viewers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultranerf.visualization.transforms import ProbeGeometry, pose_to_axes, probe_local_to_world, probe_plane_corners


@dataclass(frozen=True)
class ProbeRepresentation:
    """Probe overlay geometry in world-space millimeters."""

    origin_mm: np.ndarray
    axes_endpoints_mm: dict[str, np.ndarray]
    scan_plane_corners_mm: np.ndarray
    beam_line_mm: np.ndarray
    probe_face_line_mm: np.ndarray


def build_probe_representation(
    pose_probe_to_world: np.ndarray,
    geometry: ProbeGeometry,
    *,
    axis_length_mm: float | None = None,
) -> ProbeRepresentation:
    """Build a simple world-space representation of the probe and scan plane."""
    origin, x_axis, y_axis, z_axis = pose_to_axes(pose_probe_to_world)
    axis_length = float(axis_length_mm if axis_length_mm is not None else max(geometry.width_mm, geometry.depth_mm) * 0.25)
    scan_plane = probe_plane_corners(pose_probe_to_world, geometry)
    beam_line = probe_local_to_world(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, geometry.depth_mm, 0.0],
            ],
            dtype=np.float32,
        ),
        pose_probe_to_world,
    )
    probe_face_line = probe_local_to_world(
        np.array(
            [
                [-geometry.width_mm / 2.0, 0.0, 0.0],
                [geometry.width_mm / 2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        pose_probe_to_world,
    )
    axes_endpoints = {
        "x": origin + x_axis * axis_length,
        "y": origin + y_axis * axis_length,
        "z": origin + z_axis * axis_length,
    }
    return ProbeRepresentation(
        origin_mm=origin.astype(np.float32),
        axes_endpoints_mm={k: v.astype(np.float32) for k, v in axes_endpoints.items()},
        scan_plane_corners_mm=scan_plane.astype(np.float32),
        beam_line_mm=beam_line.astype(np.float32),
        probe_face_line_mm=probe_face_line.astype(np.float32),
    )
