import numpy as np

from visualization.comparison import (
    build_comparison_payload,
    find_nearest_pose_match,
    pose_match_score,
    rotation_distance_deg,
    translation_distance_mm,
)


def translation_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def rotation_z_90_pose() -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return pose


def test_translation_distance_mm_uses_pose_translation():
    assert translation_distance_mm(translation_pose(0.0, 0.0, 0.0), translation_pose(3.0, 4.0, 0.0)) == 5.0


def test_rotation_distance_deg_detects_quarter_turn():
    assert np.isclose(rotation_distance_deg(np.eye(4, dtype=np.float32), rotation_z_90_pose()), 90.0, atol=1e-5)


def test_pose_match_score_combines_translation_and_rotation():
    score = pose_match_score(
        np.eye(4, dtype=np.float32),
        rotation_z_90_pose(),
        translation_weight=2.0,
        rotation_weight=0.5,
    )

    assert np.isclose(score, 45.0, atol=1e-5)


def test_find_nearest_pose_match_returns_best_recorded_pose():
    recorded = np.stack(
        [
            translation_pose(0.0, 0.0, 0.0),
            translation_pose(5.0, 0.0, 0.0),
            translation_pose(20.0, 0.0, 0.0),
        ],
        axis=0,
    )
    query = translation_pose(6.0, 0.0, 0.0)

    match = find_nearest_pose_match(query, recorded)

    assert match.index == 1
    assert np.isclose(match.translation_distance_mm, 1.0)
    assert np.isclose(match.rotation_distance_deg, 0.0)


def test_build_comparison_payload_returns_matched_image_and_pose_metadata():
    recorded_images = np.array(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ]
    )
    recorded_poses = np.stack([translation_pose(0.0, 0.0, 0.0), translation_pose(10.0, 0.0, 0.0)], axis=0)
    query = translation_pose(9.0, 0.0, 0.0)
    rendered_output = {"intensity_map": np.ones((1, 1, 2, 2), dtype=np.float32)}

    payload = build_comparison_payload(
        rendered_output=rendered_output,
        query_pose_mm=query,
        recorded_images=recorded_images,
        recorded_poses_mm=recorded_poses,
    )

    assert payload["matched_index"] == 1
    assert np.allclose(payload["matched_image"], np.ones((2, 2), dtype=np.float32))
    assert np.isclose(payload["translation_distance_mm"], 1.0)
    assert np.isclose(payload["rotation_distance_deg"], 0.0)
