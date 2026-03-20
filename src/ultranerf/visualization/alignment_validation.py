"""Cross-sweep alignment validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultranerf.visualization.multi_sweep import MultiSweepScene, SweepRecord
from ultranerf.visualization.sweep_volume import compute_sweep_bounds_mm
from ultranerf.visualization.trajectory import trajectory_centers_from_poses


@dataclass(frozen=True)
class SweepAlignmentSummary:
    """Per-sweep geometric summary used for alignment inspection."""

    sweep_id: str
    frame_count: int
    bounds_min_mm: np.ndarray
    bounds_max_mm: np.ndarray
    center_mean_mm: np.ndarray
    center_std_mm: np.ndarray
    alignment_source: str
    enabled: bool


@dataclass(frozen=True)
class PairwiseAlignmentResult:
    """Pairwise geometric alignment summary between two sweeps."""

    sweep_a_id: str
    sweep_b_id: str
    centroid_distance_mm: float
    nearest_center_distance_mm: float
    support_overlap_fraction: float
    plausible_alignment: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentValidationResult:
    """Structured output from multi-sweep alignment validation."""

    per_sweep: tuple[SweepAlignmentSummary, ...]
    pairwise: tuple[PairwiseAlignmentResult, ...]
    warnings: tuple[str, ...]
    is_plausibly_aligned: bool


def summarize_sweep_alignment(sweep: SweepRecord) -> SweepAlignmentSummary:
    """Summarize one sweep in world-space millimeters."""
    centers = trajectory_centers_from_poses(sweep.poses_mm)
    bounds_min_mm, bounds_max_mm = compute_sweep_bounds_mm(sweep.poses_mm, sweep.probe_geometry)
    return SweepAlignmentSummary(
        sweep_id=sweep.sweep_id,
        frame_count=sweep.frame_count,
        bounds_min_mm=bounds_min_mm.astype(np.float32),
        bounds_max_mm=bounds_max_mm.astype(np.float32),
        center_mean_mm=centers.mean(axis=0).astype(np.float32),
        center_std_mm=centers.std(axis=0).astype(np.float32),
        alignment_source=sweep.alignment_source,
        enabled=bool(sweep.enabled),
    )


def _bounds_volume(bounds_min_mm: np.ndarray, bounds_max_mm: np.ndarray) -> float:
    extents = np.maximum(np.asarray(bounds_max_mm, dtype=np.float32) - np.asarray(bounds_min_mm, dtype=np.float32), 0.0)
    positive_extents = extents[extents > 0.0]
    if positive_extents.size == 0:
        return 0.0
    return float(np.prod(positive_extents))


def support_overlap_fraction(
    bounds_a_min_mm: np.ndarray,
    bounds_a_max_mm: np.ndarray,
    bounds_b_min_mm: np.ndarray,
    bounds_b_max_mm: np.ndarray,
) -> float:
    """Compute the overlap fraction of two AABBs relative to the smaller box."""
    bounds_a_min = np.asarray(bounds_a_min_mm, dtype=np.float32)
    bounds_a_max = np.asarray(bounds_a_max_mm, dtype=np.float32)
    bounds_b_min = np.asarray(bounds_b_min_mm, dtype=np.float32)
    bounds_b_max = np.asarray(bounds_b_max_mm, dtype=np.float32)
    extents_a = np.maximum(bounds_a_max - bounds_a_min, 0.0)
    extents_b = np.maximum(bounds_b_max - bounds_b_min, 0.0)
    shared_active_dims = np.logical_and(extents_a > 0.0, extents_b > 0.0)
    if not np.any(shared_active_dims):
        return 0.0

    intersection_min = np.maximum(bounds_a_min, bounds_b_min)
    intersection_max = np.minimum(bounds_a_max, bounds_b_max)
    intersection_extents = np.maximum(intersection_max - intersection_min, 0.0)
    if np.any(intersection_extents[shared_active_dims] <= 0.0):
        return 0.0

    intersection_volume = float(np.prod(intersection_extents[shared_active_dims]))
    volume_a = float(np.prod(extents_a[shared_active_dims]))
    volume_b = float(np.prod(extents_b[shared_active_dims]))
    min_volume = min(volume_a, volume_b)
    if min_volume <= 0.0 or intersection_volume <= 0.0:
        return 0.0
    return float(intersection_volume / min_volume)


def nearest_center_distance_mm(centers_a_mm: np.ndarray, centers_b_mm: np.ndarray) -> float:
    """Return the nearest-center distance between two trajectory point sets."""
    a = np.asarray(centers_a_mm, dtype=np.float32)
    b = np.asarray(centers_b_mm, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] == 0:
        raise ValueError("centers_a_mm must have shape (N, 3)")
    if b.ndim != 2 or b.shape[1] != 3 or b.shape[0] == 0:
        raise ValueError("centers_b_mm must have shape (M, 3)")
    deltas = a[:, None, :] - b[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return float(np.min(distances))


def compare_sweeps_alignment(
    sweep_a: SweepRecord,
    sweep_b: SweepRecord,
    *,
    centroid_warn_mm: float = 30.0,
    nearest_warn_mm: float = 15.0,
    overlap_warn_fraction: float = 0.01,
) -> PairwiseAlignmentResult:
    """Compare two sweeps using simple geometric overlap heuristics."""
    summary_a = summarize_sweep_alignment(sweep_a)
    summary_b = summarize_sweep_alignment(sweep_b)
    centers_a = trajectory_centers_from_poses(sweep_a.poses_mm)
    centers_b = trajectory_centers_from_poses(sweep_b.poses_mm)

    centroid_distance = float(np.linalg.norm(summary_a.center_mean_mm - summary_b.center_mean_mm))
    nearest_distance = nearest_center_distance_mm(centers_a, centers_b)
    overlap_fraction = support_overlap_fraction(
        summary_a.bounds_min_mm,
        summary_a.bounds_max_mm,
        summary_b.bounds_min_mm,
        summary_b.bounds_max_mm,
    )

    warnings: list[str] = []
    if centroid_distance > float(centroid_warn_mm):
        warnings.append(
            f"centroid distance {centroid_distance:.2f} mm exceeds {float(centroid_warn_mm):.2f} mm"
        )
    if nearest_distance > float(nearest_warn_mm):
        warnings.append(
            f"nearest center distance {nearest_distance:.2f} mm exceeds {float(nearest_warn_mm):.2f} mm"
        )
    if overlap_fraction < float(overlap_warn_fraction):
        warnings.append(
            f"support overlap fraction {overlap_fraction:.4f} is below {float(overlap_warn_fraction):.4f}"
        )

    return PairwiseAlignmentResult(
        sweep_a_id=sweep_a.sweep_id,
        sweep_b_id=sweep_b.sweep_id,
        centroid_distance_mm=centroid_distance,
        nearest_center_distance_mm=nearest_distance,
        support_overlap_fraction=overlap_fraction,
        plausible_alignment=not warnings,
        warnings=tuple(warnings),
    )


def validate_multi_sweep_alignment(
    scene: MultiSweepScene,
    *,
    centroid_warn_mm: float = 30.0,
    nearest_warn_mm: float = 15.0,
    overlap_warn_fraction: float = 0.01,
) -> AlignmentValidationResult:
    """Validate cross-sweep alignment heuristically before visualization."""
    sweeps = scene.enabled_sweeps or scene.sweeps
    per_sweep = tuple(summarize_sweep_alignment(sweep) for sweep in sweeps)
    pairwise_results = []
    all_warnings: list[str] = []

    for idx, sweep_a in enumerate(sweeps):
        for sweep_b in sweeps[idx + 1 :]:
            result = compare_sweeps_alignment(
                sweep_a,
                sweep_b,
                centroid_warn_mm=centroid_warn_mm,
                nearest_warn_mm=nearest_warn_mm,
                overlap_warn_fraction=overlap_warn_fraction,
            )
            pairwise_results.append(result)
            for warning in result.warnings:
                all_warnings.append(f"{result.sweep_a_id} vs {result.sweep_b_id}: {warning}")

    return AlignmentValidationResult(
        per_sweep=per_sweep,
        pairwise=tuple(pairwise_results),
        warnings=tuple(all_warnings),
        is_plausibly_aligned=not all_warnings,
    )
