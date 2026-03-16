import numpy as np

from visualization.probe_orientation import (
    orthonormalize_rotation,
    pose_from_yaw_pitch_roll,
    rotation_matrix_from_yaw_pitch_roll,
    update_probe_pose_orientation,
)


def test_rotation_matrix_from_yaw_pitch_roll_identity_case():
    rotation = rotation_matrix_from_yaw_pitch_roll(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)

    assert np.allclose(rotation, np.eye(3, dtype=np.float32))


def test_rotation_matrix_from_yaw_pitch_roll_rotates_x_into_y_for_90deg_yaw():
    rotation = rotation_matrix_from_yaw_pitch_roll(yaw_deg=90.0, pitch_deg=0.0, roll_deg=0.0)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rotated = rotation @ x_axis

    assert np.allclose(rotated, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_orthonormalize_rotation_projects_noisy_matrix_to_valid_rotation():
    noisy = np.array(
        [
            [1.0, 0.01, 0.0],
            [0.0, 0.99, -0.02],
            [0.0, 0.03, 1.01],
        ],
        dtype=np.float32,
    )

    rotation = orthonormalize_rotation(noisy)

    assert np.allclose(rotation.T @ rotation, np.eye(3, dtype=np.float32), atol=1e-5)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)


def test_pose_from_yaw_pitch_roll_preserves_origin_and_sets_rotation():
    pose = pose_from_yaw_pitch_roll(np.array([1.0, 2.0, 3.0], dtype=np.float32), yaw_deg=90.0, pitch_deg=0.0, roll_deg=0.0)

    assert np.allclose(pose[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_update_probe_pose_orientation_preserves_origin():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    updated = update_probe_pose_orientation(pose, yaw_deg=0.0, pitch_deg=90.0, roll_deg=0.0)

    assert np.allclose(updated[:3, 3], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert np.allclose(updated[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-6)
