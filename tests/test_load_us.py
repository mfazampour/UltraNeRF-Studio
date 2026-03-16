import numpy as np

from load_us import load_rec_data, load_us_data


def make_pose(tx_mm: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = tx_mm
    return pose


def test_load_us_data_normalizes_images_scales_translations_and_selects_holdout(tmp_path):
    images = np.array(
        [
            np.full((4, 5), 0, dtype=np.uint8),
            np.full((4, 5), 128, dtype=np.uint8),
            np.full((4, 5), 255, dtype=np.uint8),
        ]
    )
    poses = np.stack([make_pose(0.0), make_pose(1.0), make_pose(3.0)], axis=0)

    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "poses.npy", poses)

    loaded_images, loaded_poses, i_test = load_us_data(str(tmp_path))

    assert loaded_images.dtype == np.float32
    assert loaded_poses.dtype == np.float32
    assert np.isclose(loaded_images[1, 0, 0], 128.0 / 255.0)
    assert np.allclose(loaded_poses[:, 0, 3], np.array([0.0, 0.001, 0.003], dtype=np.float32))
    assert i_test == 1


def test_load_us_data_reconstruction_returns_labels_and_pose_labels(tmp_path):
    images = np.full((2, 3, 4), 64, dtype=np.uint8)
    labels = np.full((2, 3, 4), 255, dtype=np.uint8)
    poses = np.stack([make_pose(0.0), make_pose(2.0)], axis=0)
    poses_labels = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "poses.npy", poses)
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "poses_labels.npy", poses_labels)

    loaded_images, loaded_poses, loaded_labels, loaded_pose_labels, i_test = load_us_data(
        str(tmp_path), reconstruction=True
    )

    assert loaded_images.shape == images.shape
    assert loaded_labels.shape == labels.shape
    assert loaded_pose_labels.shape == poses_labels.shape
    assert np.allclose(loaded_labels, 1.0)
    assert i_test in (0, 1)


def test_load_rec_data_normalizes_labels(tmp_path):
    labels = np.array([[[0, 255]]], dtype=np.uint8)
    poses_labels = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)

    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "poses_labels.npy", poses_labels)

    loaded_labels, loaded_pose_labels = load_rec_data(str(tmp_path))

    assert np.allclose(loaded_labels, np.array([[[0.0, 1.0]]], dtype=np.float32))
    assert np.array_equal(loaded_pose_labels, poses_labels)
