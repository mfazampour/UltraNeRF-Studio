"""Sweep-aware nearest-frame matching for multi-sweep scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ultranerf.visualization.comparison import rotation_distance_deg, translation_distance_mm
from ultranerf.visualization.multi_sweep import MultiSweepScene, SweepRecord
from ultranerf.visualization.transforms import ensure_pose_matrix


MultiSweepComparisonPolicy = str


@dataclass(frozen=True)
class MultiSweepComparisonMatch:
    """Nearest recorded frame match across one or more sweeps."""

    sweep_id: str
    display_name: str
    frame_index: int
    translation_distance_mm: float
    rotation_distance_deg: float
    score: float
    matched_pose_mm: np.ndarray
    matched_image: np.ndarray
    from_active_sweep: bool


def resolve_candidate_sweeps(
    scene: MultiSweepScene,
    *,
    active_sweep_id: str | None = None,
    comparison_policy: MultiSweepComparisonPolicy | None = None,
    allowed_sweep_ids: tuple[str, ...] | None = None,
) -> tuple[SweepRecord, ...]:
    """Resolve which sweeps participate in comparison."""
    policy = comparison_policy or scene.comparison_policy
    active_id = active_sweep_id or scene.active_sweep_id

    if allowed_sweep_ids is not None:
        allowed = set(allowed_sweep_ids)
        sweeps = tuple(sweep for sweep in scene.sweeps if sweep.sweep_id in allowed and sweep.enabled)
    elif policy == "active_only":
        sweeps = (scene.get_sweep(active_id),)
    else:
        sweeps = scene.enabled_sweeps or scene.sweeps

    if not sweeps:
        raise ValueError("No sweeps are available for comparison")
    return sweeps


def find_multi_sweep_pose_match(
    query_pose_mm: np.ndarray,
    scene: MultiSweepScene,
    *,
    active_sweep_id: str | None = None,
    comparison_policy: MultiSweepComparisonPolicy | None = None,
    allowed_sweep_ids: tuple[str, ...] | None = None,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> MultiSweepComparisonMatch:
    """Find the nearest recorded frame across the chosen sweep set."""
    query_pose = ensure_pose_matrix(query_pose_mm)
    active_id = active_sweep_id or scene.active_sweep_id
    candidate_sweeps = resolve_candidate_sweeps(
        scene,
        active_sweep_id=active_id,
        comparison_policy=comparison_policy,
        allowed_sweep_ids=allowed_sweep_ids,
    )

    best_match: MultiSweepComparisonMatch | None = None
    for sweep in candidate_sweeps:
        for frame_index, pose in enumerate(sweep.poses_mm):
            translation = translation_distance_mm(query_pose, pose)
            rotation = rotation_distance_deg(query_pose, pose)
            score = translation_weight * translation + rotation_weight * rotation
            if best_match is None or score < best_match.score:
                best_match = MultiSweepComparisonMatch(
                    sweep_id=sweep.sweep_id,
                    display_name=sweep.display_name or sweep.sweep_id,
                    frame_index=int(frame_index),
                    translation_distance_mm=float(translation),
                    rotation_distance_deg=float(rotation),
                    score=float(score),
                    matched_pose_mm=ensure_pose_matrix(pose).astype(np.float32),
                    matched_image=np.asarray(sweep.images[frame_index], dtype=np.float32),
                    from_active_sweep=(sweep.sweep_id == active_id),
                )

    if best_match is None:
        raise ValueError("No recorded poses were available for multi-sweep comparison")
    return best_match


def build_multi_sweep_comparison_payload(
    *,
    rendered_output: dict[str, Any],
    query_pose_mm: np.ndarray,
    scene: MultiSweepScene,
    active_sweep_id: str | None = None,
    comparison_policy: MultiSweepComparisonPolicy | None = None,
    allowed_sweep_ids: tuple[str, ...] | None = None,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> dict[str, Any]:
    """Build a viewer-ready comparison payload for a multi-sweep scene."""
    match = find_multi_sweep_pose_match(
        query_pose_mm,
        scene,
        active_sweep_id=active_sweep_id,
        comparison_policy=comparison_policy,
        allowed_sweep_ids=allowed_sweep_ids,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
    )
    return {
        "rendered_output": rendered_output,
        "query_pose_mm": ensure_pose_matrix(query_pose_mm).astype(np.float32),
        "matched_sweep_id": match.sweep_id,
        "matched_sweep_name": match.display_name,
        "matched_index": match.frame_index,
        "matched_pose_mm": match.matched_pose_mm,
        "matched_image": match.matched_image,
        "translation_distance_mm": match.translation_distance_mm,
        "rotation_distance_deg": match.rotation_distance_deg,
        "score": match.score,
        "from_active_sweep": match.from_active_sweep,
    }
