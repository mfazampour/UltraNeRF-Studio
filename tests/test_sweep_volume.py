import numpy as np

from visualization.sweep_volume import (
    compute_sweep_bounds_mm,
    fuse_sweeps_to_volume,
    volume_geometry_from_bounds_mm,
)
from visualization.transforms import ProbeGeometry, VolumeGeometry


def translation_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def test_compute_sweep_bounds_mm_covers_multiple_scan_planes():
    geometry = ProbeGeometry(width_mm=4.0, depth_mm=6.0)
    poses = np.stack(
        [
            translation_pose(0.0, 0.0, 0.0),
            translation_pose(10.0, 0.0, 2.0),
        ],
        axis=0,
    )

    bounds_min, bounds_max = compute_sweep_bounds_mm(poses, geometry)

    assert np.allclose(bounds_min, np.array([-2.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(bounds_max, np.array([12.0, 6.0, 2.0], dtype=np.float32))


def test_volume_geometry_from_bounds_mm_creates_expected_shape():
    geometry, shape = volume_geometry_from_bounds_mm(
        bounds_min_mm=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_mm=np.array([4.0, 6.0, 2.0], dtype=np.float32),
        spacing_mm=(2.0, 3.0, 1.0),
    )

    assert np.allclose(geometry.origin_mm, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(geometry.spacing_mm, np.array([2.0, 3.0, 1.0], dtype=np.float32))
    assert shape == (3, 3, 3)


def test_fuse_sweeps_to_volume_accumulates_synthetic_slice_into_expected_voxels():
    images = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32)], axis=0)
    probe_geometry = ProbeGeometry(width_mm=4.0, depth_mm=4.0)
    volume_geometry = VolumeGeometry(origin_mm=np.array([-1.0, 1.0, 0.0]), spacing_mm=np.array([2.0, 2.0, 1.0]))

    fused = fuse_sweeps_to_volume(
        images=images,
        poses_probe_to_world=poses,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=(2, 2, 1),
    )

    expected = np.array(
        [
            [[1.0], [3.0]],
            [[2.0], [4.0]],
        ],
        dtype=np.float32,
    )

    assert np.allclose(fused.scalar_volume, expected)
    assert np.allclose(fused.weight_volume, np.ones((2, 2, 1), dtype=np.float32))
    assert np.allclose(fused.bounds_min_mm, np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(fused.bounds_max_mm, np.array([1.0, 3.0, 0.0], dtype=np.float32))


def test_fuse_sweeps_to_volume_averages_overlapping_samples():
    images = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    probe_geometry = ProbeGeometry(width_mm=4.0, depth_mm=4.0)
    volume_geometry = VolumeGeometry(origin_mm=np.array([-1.0, 1.0, 0.0]), spacing_mm=np.array([2.0, 2.0, 1.0]))

    fused = fuse_sweeps_to_volume(
        images=images,
        poses_probe_to_world=poses,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=(2, 2, 1),
        reduction_mode="mean",
    )

    assert np.allclose(fused.scalar_volume, np.full((2, 2, 1), 2.0, dtype=np.float32))
    assert np.allclose(fused.weight_volume, np.full((2, 2, 1), 2.0, dtype=np.float32))


def test_fuse_sweeps_to_volume_uses_max_for_overlapping_samples_when_requested():
    images = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    probe_geometry = ProbeGeometry(width_mm=4.0, depth_mm=4.0)
    volume_geometry = VolumeGeometry(origin_mm=np.array([-1.0, 1.0, 0.0]), spacing_mm=np.array([2.0, 2.0, 1.0]))

    fused = fuse_sweeps_to_volume(
        images=images,
        poses_probe_to_world=poses,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=(2, 2, 1),
        reduction_mode="max",
    )

    assert np.allclose(fused.scalar_volume, np.full((2, 2, 1), 3.0, dtype=np.float32))
    assert np.allclose(fused.weight_volume, np.full((2, 2, 1), 2.0, dtype=np.float32))


def test_fuse_sweeps_to_volume_skips_nonfinite_samples():
    images = np.array(
        [
            [
                [1.0, np.nan],
                [np.inf, 4.0],
            ]
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32)], axis=0)
    probe_geometry = ProbeGeometry(width_mm=4.0, depth_mm=4.0)
    volume_geometry = VolumeGeometry(origin_mm=np.array([-1.0, 1.0, 0.0]), spacing_mm=np.array([2.0, 2.0, 1.0]))

    fused = fuse_sweeps_to_volume(
        images=images,
        poses_probe_to_world=poses,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=(2, 2, 1),
    )

    expected_scalar = np.array(
        [
            [[1.0], [0.0]],
            [[0.0], [4.0]],
        ],
        dtype=np.float32,
    )
    expected_weights = np.array(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ],
        dtype=np.float32,
    )

    assert np.allclose(fused.scalar_volume, expected_scalar)
    assert np.allclose(fused.weight_volume, expected_weights)
