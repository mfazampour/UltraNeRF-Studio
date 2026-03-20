"""Comparison helpers for matching arbitrary probe poses to recorded frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ultranerf.visualization.transforms import ensure_pose_matrix


@dataclass(frozen=True)
class ComparisonMatch:
    """A nearest recorded frame match for a probe pose."""

    index: int
    translation_distance_mm: float
    rotation_distance_deg: float
    score: float
    matched_pose_mm: np.ndarray


def translation_distance_mm(pose_a_mm: np.ndarray, pose_b_mm: np.ndarray) -> float:
    """Euclidean translation distance in millimeters."""
    pose_a = ensure_pose_matrix(pose_a_mm)
    pose_b = ensure_pose_matrix(pose_b_mm)
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def rotation_distance_deg(pose_a_mm: np.ndarray, pose_b_mm: np.ndarray) -> float:
    """Angular distance between two rotations in degrees."""
    pose_a = ensure_pose_matrix(pose_a_mm)
    pose_b = ensure_pose_matrix(pose_b_mm)
    rel = pose_a[:3, :3].T @ pose_b[:3, :3]
    trace = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = float(np.arccos(trace))
    return float(np.rad2deg(angle_rad))


def pose_match_score(
    pose_a_mm: np.ndarray,
    pose_b_mm: np.ndarray,
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> float:
    """Combined pose match score from translation and rotation distance."""
    return (
        translation_weight * translation_distance_mm(pose_a_mm, pose_b_mm)
        + rotation_weight * rotation_distance_deg(pose_a_mm, pose_b_mm)
    )


def find_nearest_pose_match(
    query_pose_mm: np.ndarray,
    recorded_poses_mm: np.ndarray,
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> ComparisonMatch:
    """Find the nearest recorded pose to an arbitrary query pose."""
    poses = np.asarray(recorded_poses_mm, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[0] == 0:
        raise ValueError("recorded_poses_mm must have shape (N, 3, 4) or (N, 4, 4)")

    best_index = -1
    best_score = float("inf")
    best_translation = 0.0
    best_rotation = 0.0

    for idx, pose in enumerate(poses):
        t_dist = translation_distance_mm(query_pose_mm, pose)
        r_dist = rotation_distance_deg(query_pose_mm, pose)
        score = translation_weight * t_dist + rotation_weight * r_dist
        if score < best_score:
            best_index = idx
            best_score = score
            best_translation = t_dist
            best_rotation = r_dist

    matched_pose = ensure_pose_matrix(poses[best_index]).astype(np.float32)
    return ComparisonMatch(
        index=int(best_index),
        translation_distance_mm=float(best_translation),
        rotation_distance_deg=float(best_rotation),
        score=float(best_score),
        matched_pose_mm=matched_pose,
    )


def build_comparison_payload(
    *,
    rendered_output: dict[str, Any],
    query_pose_mm: np.ndarray,
    recorded_images: np.ndarray,
    recorded_poses_mm: np.ndarray,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> dict[str, Any]:
    """Create a comparison payload for viewer use."""
    match = find_nearest_pose_match(
        query_pose_mm,
        recorded_poses_mm,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
    )
    return {
        "rendered_output": rendered_output,
        "query_pose_mm": ensure_pose_matrix(query_pose_mm).astype(np.float32),
        "matched_index": match.index,
        "matched_pose_mm": match.matched_pose_mm,
        "matched_image": np.asarray(recorded_images[match.index]),
        "translation_distance_mm": match.translation_distance_mm,
        "rotation_distance_deg": match.rotation_distance_deg,
        "score": match.score,
    }
