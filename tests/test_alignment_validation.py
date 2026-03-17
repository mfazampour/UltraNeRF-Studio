import numpy as np

from visualization.alignment_validation import (
    compare_sweeps_alignment,
    nearest_center_distance_mm,
    support_overlap_fraction,
    validate_multi_sweep_alignment,
)
from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.transforms import ProbeGeometry


def make_images() -> np.ndarray:
    return np.ones((3, 4, 5), dtype=np.float32)


def make_poses(offset_mm: float) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    for idx in range(3):
        poses[idx, 0, 3] = offset_mm + idx * 2.0
    return poses


def make_sweep(sweep_id: str, offset_mm: float) -> SweepRecord:
    return SweepRecord(
        sweep_id=sweep_id,
        images=make_images(),
        poses_mm=make_poses(offset_mm),
        probe_geometry=ProbeGeometry(width_mm=20.0, depth_mm=30.0),
    )


def test_support_overlap_fraction_is_zero_for_disjoint_boxes() -> None:
    overlap = support_overlap_fraction(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([10.0, 10.0, 10.0], dtype=np.float32),
        np.array([20.0, 0.0, 0.0], dtype=np.float32),
        np.array([30.0, 10.0, 10.0], dtype=np.float32),
    )

    assert overlap == 0.0


def test_nearest_center_distance_uses_minimum_pairwise_distance() -> None:
    centers_a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    centers_b = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)

    assert nearest_center_distance_mm(centers_a, centers_b) == 3.0


def test_compare_sweeps_alignment_marks_nearby_sweeps_as_plausible() -> None:
    result = compare_sweeps_alignment(make_sweep("a", 0.0), make_sweep("b", 1.0))

    assert result.plausible_alignment is True
    assert result.support_overlap_fraction > 0.0
    assert result.nearest_center_distance_mm <= 1.0


def test_compare_sweeps_alignment_warns_for_large_offsets() -> None:
    result = compare_sweeps_alignment(
        make_sweep("a", 0.0),
        make_sweep("b", 200.0),
        centroid_warn_mm=30.0,
        nearest_warn_mm=15.0,
        overlap_warn_fraction=0.01,
    )

    assert result.plausible_alignment is False
    assert result.support_overlap_fraction == 0.0
    assert any("centroid distance" in warning for warning in result.warnings)


def test_validate_multi_sweep_alignment_returns_pairwise_warnings() -> None:
    scene = MultiSweepScene(sweeps=(make_sweep("a", 0.0), make_sweep("b", 200.0)))

    result = validate_multi_sweep_alignment(scene)

    assert result.is_plausibly_aligned is False
    assert len(result.per_sweep) == 2
    assert len(result.pairwise) == 1
    assert result.warnings
