import numpy as np

from ultranerf.visualization.multi_sweep import MultiSweepScene, SweepRecord
from ultranerf.visualization.multi_sweep_comparison import (
    build_multi_sweep_comparison_payload,
    find_multi_sweep_pose_match,
    resolve_candidate_sweeps,
)
from ultranerf.visualization.transforms import ProbeGeometry


def make_images(value: float) -> np.ndarray:
    return np.full((2, 3, 4), value, dtype=np.float32)


def make_poses(offset_mm: float) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = np.array([offset_mm, offset_mm + 1.0], dtype=np.float32)
    return poses


def make_scene() -> MultiSweepScene:
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=20.0)
    return MultiSweepScene(
        sweeps=(
            SweepRecord("a", make_images(1.0), make_poses(0.0), geometry, display_name="Sweep A"),
            SweepRecord("b", make_images(2.0), make_poses(50.0), geometry, display_name="Sweep B"),
        ),
        active_sweep_id="a",
    )


def test_resolve_candidate_sweeps_defaults_to_enabled_sweeps() -> None:
    scene = make_scene()

    sweeps = resolve_candidate_sweeps(scene)

    assert tuple(sweep.sweep_id for sweep in sweeps) == ("a", "b")


def test_find_multi_sweep_pose_match_can_restrict_to_active_sweep() -> None:
    scene = make_scene()
    query_pose = np.eye(4, dtype=np.float32)
    query_pose[0, 3] = 51.0

    match = find_multi_sweep_pose_match(query_pose, scene, comparison_policy="active_only")

    assert match.sweep_id == "a"
    assert match.from_active_sweep is True


def test_find_multi_sweep_pose_match_searches_across_all_enabled_sweeps() -> None:
    scene = make_scene()
    query_pose = np.eye(4, dtype=np.float32)
    query_pose[0, 3] = 51.0

    match = find_multi_sweep_pose_match(query_pose, scene, comparison_policy="all_enabled")

    assert match.sweep_id == "b"
    assert match.frame_index == 1
    assert match.from_active_sweep is False


def test_build_multi_sweep_comparison_payload_reports_sweep_metadata() -> None:
    scene = make_scene()
    query_pose = np.eye(4, dtype=np.float32)
    query_pose[0, 3] = 50.0

    payload = build_multi_sweep_comparison_payload(
        rendered_output={"intensity_map": np.ones((1, 1, 2, 2), dtype=np.float32)},
        query_pose_mm=query_pose,
        scene=scene,
    )

    assert payload["matched_sweep_id"] == "b"
    assert payload["matched_sweep_name"] == "Sweep B"
    assert payload["matched_index"] == 0
    assert payload["from_active_sweep"] is False
