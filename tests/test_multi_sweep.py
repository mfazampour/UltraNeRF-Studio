from pathlib import Path

import numpy as np
import pytest

from ultranerf.visualization.multi_sweep import (
    MultiSweepScene,
    SweepRecord,
    apply_world_transform_to_poses,
    validate_sweep_images_and_poses,
)
from ultranerf.visualization.transforms import ProbeGeometry


def make_images(num_frames: int = 3, height: int = 4, width: int = 5) -> np.ndarray:
    return np.arange(num_frames * height * width, dtype=np.float32).reshape(num_frames, height, width)


def make_poses(num_frames: int = 3) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_frames, axis=0)
    for idx in range(num_frames):
        poses[idx, 0, 3] = float(idx)
    return poses


def test_validate_sweep_images_and_poses_normalizes_poses_to_4x4() -> None:
    images = make_images()
    poses = make_poses()[:, :3, :]

    validated_images, validated_poses = validate_sweep_images_and_poses(images, poses)

    assert validated_images.shape == (3, 4, 5)
    assert validated_poses.shape == (3, 4, 4)
    assert np.allclose(validated_poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_sweep_record_normalizes_metadata_and_defaults() -> None:
    sweep = SweepRecord(
        sweep_id="sweep_a",
        images=make_images(),
        poses_mm=make_poses()[:, :3, :],
        probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
        dataset_dir="/tmp/example",
        color_rgb=(0.1, 0.2, 0.3),
    )

    assert sweep.display_name == "sweep_a"
    assert sweep.frame_count == 3
    assert sweep.image_shape == (4, 5)
    assert sweep.dataset_dir == Path("/tmp/example")
    assert sweep.color_rgb == (0.1, 0.2, 0.3)
    assert sweep.alignment_source == "assumed_from_training"
    assert np.allclose(sweep.raw_poses_mm, make_poses())
    assert np.allclose(sweep.poses_mm, make_poses())


def test_multi_sweep_scene_tracks_active_and_enabled_sweeps() -> None:
    geometry = ProbeGeometry(width_mm=80.0, depth_mm=139.0)
    sweep_a = SweepRecord("a", make_images(), make_poses(), geometry)
    sweep_b = SweepRecord("b", make_images(), make_poses(), geometry, enabled=False)

    scene = MultiSweepScene(sweeps=(sweep_a, sweep_b), active_sweep_id="a")

    assert scene.world_unit == "mm"
    assert scene.sweep_ids == ("a", "b")
    assert scene.active_sweep.sweep_id == "a"
    assert tuple(sweep.sweep_id for sweep in scene.enabled_sweeps) == ("a",)
    assert scene.with_active_sweep("b").active_sweep.sweep_id == "b"


def test_multi_sweep_scene_requires_unique_ids() -> None:
    geometry = ProbeGeometry(width_mm=80.0, depth_mm=139.0)
    sweep_a = SweepRecord("dup", make_images(), make_poses(), geometry)
    sweep_b = SweepRecord("dup", make_images(), make_poses(), geometry)

    with pytest.raises(ValueError, match="unique"):
        MultiSweepScene(sweeps=(sweep_a, sweep_b))


def test_sweep_record_rejects_invalid_color_range() -> None:
    with pytest.raises(ValueError, match="range"):
        SweepRecord(
            sweep_id="bad",
            images=make_images(),
            poses_mm=make_poses(),
            probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
            color_rgb=(1.5, 0.0, 0.0),
        )


def test_apply_world_transform_to_poses_left_multiplies_pose_batch() -> None:
    poses = make_poses()
    transform = np.eye(4, dtype=np.float32)
    transform[1, 3] = 5.0

    transformed = apply_world_transform_to_poses(poses, transform)

    assert np.allclose(transformed[:, 0, 3], poses[:, 0, 3])
    assert np.allclose(transformed[:, 1, 3], np.full(poses.shape[0], 5.0, dtype=np.float32))


def test_sweep_record_applies_world_transform_and_preserves_raw_poses() -> None:
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = 12.0

    sweep = SweepRecord(
        sweep_id="registered",
        images=make_images(),
        poses_mm=make_poses(),
        probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
        world_transform_mm=transform,
    )

    assert sweep.has_identity_world_transform is False
    assert np.allclose(sweep.raw_poses_mm[:, 0, 3], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(sweep.poses_mm[:, 2, 3], np.full(3, 12.0, dtype=np.float32))


def test_with_world_transform_rebuilds_transformed_pose_batch() -> None:
    sweep = SweepRecord(
        sweep_id="base",
        images=make_images(),
        poses_mm=make_poses(),
        probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
    )
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 100.0

    transformed = sweep.with_world_transform(transform)

    assert transformed.alignment_source == "externally_registered"
    assert np.allclose(transformed.raw_poses_mm, sweep.raw_poses_mm)
    assert np.allclose(transformed.poses_mm[:, 0, 3], np.array([100.0, 101.0, 102.0], dtype=np.float32))
