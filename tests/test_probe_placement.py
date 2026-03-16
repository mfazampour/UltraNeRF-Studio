import numpy as np

from visualization.mpr import selection_from_world_point
from visualization.probe_placement import default_probe_rotation, probe_pose_from_mpr_selection
from visualization.transforms import VolumeGeometry


def rotation_z_90() -> np.ndarray:
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_probe_pose_from_mpr_selection_uses_identity_rotation_by_default():
    geometry = VolumeGeometry(origin_mm=np.array([0.0, 0.0, 0.0]), spacing_mm=np.array([1.0, 1.0, 1.0]))
    selection = selection_from_world_point(np.array([10.0, 20.0, 30.0], dtype=np.float32), geometry, (100, 100, 100))

    pose = probe_pose_from_mpr_selection(selection)

    assert np.allclose(pose[:3, :3], default_probe_rotation())
    assert np.allclose(pose[:3, 3], np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_probe_pose_from_mpr_selection_preserves_current_orientation():
    geometry = VolumeGeometry(origin_mm=np.array([0.0, 0.0, 0.0]), spacing_mm=np.array([1.0, 1.0, 1.0]))
    selection = selection_from_world_point(np.array([5.0, 6.0, 7.0], dtype=np.float32), geometry, (100, 100, 100))
    current_pose = np.eye(4, dtype=np.float32)
    current_pose[:3, :3] = rotation_z_90()
    current_pose[:3, 3] = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    pose = probe_pose_from_mpr_selection(selection, current_pose=current_pose)

    assert np.allclose(pose[:3, :3], rotation_z_90())
    assert np.allclose(pose[:3, 3], np.array([5.0, 6.0, 7.0], dtype=np.float32))


def test_probe_pose_from_mpr_selection_accepts_custom_default_rotation():
    geometry = VolumeGeometry(origin_mm=np.array([0.0, 0.0, 0.0]), spacing_mm=np.array([1.0, 1.0, 1.0]))
    selection = selection_from_world_point(np.array([1.0, 2.0, 3.0], dtype=np.float32), geometry, (100, 100, 100))

    pose = probe_pose_from_mpr_selection(selection, default_rotation_matrix=rotation_z_90())

    assert np.allclose(pose[:3, :3], rotation_z_90())
    assert np.allclose(pose[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
