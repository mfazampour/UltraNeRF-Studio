import numpy as np

from visualization.transforms import (
    ProbeGeometry,
    VolumeGeometry,
    ensure_pose_matrix,
    invert_pose,
    pixel_to_probe_local,
    pose_to_axes,
    probe_local_to_world,
    probe_plane_corners,
    voxel_to_world,
    world_to_probe_local,
    world_to_voxel,
)


def translation_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def test_ensure_pose_matrix_accepts_3x4_and_4x4():
    pose_34 = np.concatenate([np.eye(3, dtype=np.float32), np.array([[1.0], [2.0], [3.0]], dtype=np.float32)], axis=1)
    pose_44 = ensure_pose_matrix(pose_34)

    assert pose_44.shape == (4, 4)
    assert np.allclose(pose_44[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_invert_pose_round_trip_identity():
    pose = translation_pose(10.0, 20.0, 30.0)

    inv = invert_pose(pose)
    identity = pose @ inv

    assert np.allclose(identity, np.eye(4), atol=1e-6)


def test_pixel_to_probe_local_uses_centered_lateral_and_forward_depth():
    geometry = ProbeGeometry(width_mm=80.0, depth_mm=140.0)
    pts = pixel_to_probe_local(
        row=np.array([0.0, 139.0], dtype=np.float32),
        col=np.array([0.0, 79.0], dtype=np.float32),
        image_shape=(140, 80),
        geometry=geometry,
    )

    assert np.allclose(pts[0], np.array([-39.5, 0.5, 0.0], dtype=np.float32))
    assert np.allclose(pts[1], np.array([39.5, 139.5, 0.0], dtype=np.float32))


def test_probe_local_to_world_and_back_are_consistent():
    pose = translation_pose(10.0, 20.0, 30.0)
    local_points = np.array([[0.0, 0.0, 0.0], [5.0, 15.0, 0.0]], dtype=np.float32)

    world_points = probe_local_to_world(local_points, pose)
    recovered_local = world_to_probe_local(world_points, pose)

    assert np.allclose(world_points, np.array([[10.0, 20.0, 30.0], [15.0, 35.0, 30.0]], dtype=np.float32))
    assert np.allclose(recovered_local, local_points, atol=1e-6)


def test_world_and_voxel_conversion_are_consistent():
    geometry = VolumeGeometry(origin_mm=np.array([10.0, 20.0, 30.0]), spacing_mm=np.array([0.5, 1.0, 2.0]))
    world_points = np.array([[10.0, 20.0, 30.0], [11.0, 24.0, 34.0]], dtype=np.float32)

    voxel_points = world_to_voxel(world_points, geometry)
    recovered_world = voxel_to_world(voxel_points, geometry)

    assert np.allclose(voxel_points, np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 2.0]], dtype=np.float32))
    assert np.allclose(recovered_world, world_points, atol=1e-6)


def test_probe_plane_corners_match_identity_pose_geometry():
    geometry = ProbeGeometry(width_mm=80.0, depth_mm=140.0)
    corners = probe_plane_corners(np.eye(4, dtype=np.float32), geometry)

    expected = np.array(
        [
            [-40.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [40.0, 140.0, 0.0],
            [-40.0, 140.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(corners, expected)


def test_pose_to_axes_returns_origin_and_basis_vectors():
    pose = translation_pose(1.0, 2.0, 3.0)

    origin, x_axis, y_axis, z_axis = pose_to_axes(pose)

    assert np.allclose(origin, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(z_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
