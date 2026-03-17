import numpy as np

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.multi_sweep_volume import build_sweep_overlay, fuse_multi_sweep_scene
from visualization.transforms import ProbeGeometry


def make_images(value: float) -> np.ndarray:
    return np.full((2, 4, 4), value, dtype=np.float32)


def make_poses(offset_mm: float) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = offset_mm
    return poses


def make_sweep(sweep_id: str, value: float, offset_mm: float, *, enabled: bool = True) -> SweepRecord:
    return SweepRecord(
        sweep_id=sweep_id,
        images=make_images(value),
        poses_mm=make_poses(offset_mm),
        probe_geometry=ProbeGeometry(width_mm=20.0, depth_mm=20.0),
        enabled=enabled,
    )


def test_build_sweep_overlay_returns_per_sweep_volume_and_trajectory() -> None:
    overlay = build_sweep_overlay(
        make_sweep("a", 1.0, 0.0),
        spacing_mm=(5.0, 5.0, 5.0),
        pixel_stride=(2, 2),
        axis_stride=1,
    )

    assert overlay.sweep_id == "a"
    assert overlay.fused_volume.scalar_volume.ndim == 3
    assert overlay.trajectory.centers_mm.shape[0] == 2


def test_build_sweep_overlay_can_skip_volume_generation() -> None:
    overlay = build_sweep_overlay(
        make_sweep("a", 1.0, 0.0),
        spacing_mm=(5.0, 5.0, 5.0),
        pixel_stride=(2, 2),
        axis_stride=1,
        include_volume=False,
    )

    assert overlay.sweep_id == "a"
    assert overlay.fused_volume is None
    assert overlay.trajectory.centers_mm.shape[0] == 2


def test_fuse_multi_sweep_scene_combines_enabled_sweeps_only() -> None:
    scene = MultiSweepScene(
        sweeps=(
            make_sweep("a", 1.0, 0.0, enabled=True),
            make_sweep("b", 5.0, 40.0, enabled=False),
        )
    )

    result = fuse_multi_sweep_scene(scene, spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    assert result.enabled_sweep_ids == ("a",)
    assert tuple(overlay.sweep_id for overlay in result.sweep_overlays) == ("a",)


def test_fuse_multi_sweep_scene_respects_explicit_enabled_id_subset() -> None:
    scene = MultiSweepScene(
        sweeps=(
            make_sweep("a", 1.0, 0.0),
            make_sweep("b", 5.0, 40.0),
        )
    )

    result = fuse_multi_sweep_scene(
        scene,
        spacing_mm=(5.0, 5.0, 5.0),
        pixel_stride=(2, 2),
        enabled_sweep_ids=("b",),
    )

    assert result.enabled_sweep_ids == ("b",)
    assert tuple(overlay.sweep_id for overlay in result.sweep_overlays) == ("b",)


def test_fuse_multi_sweep_scene_uses_union_bounds_for_selected_sweeps() -> None:
    scene = MultiSweepScene(
        sweeps=(
            make_sweep("a", 1.0, 0.0),
            make_sweep("b", 5.0, 40.0),
        )
    )

    result = fuse_multi_sweep_scene(scene, spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    assert result.bounds_min_mm[0] <= -10.0
    assert result.bounds_max_mm[0] >= 50.0


def test_fuse_multi_sweep_scene_uses_transformed_world_poses() -> None:
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = 25.0
    scene = MultiSweepScene(
        sweeps=(
            SweepRecord(
                sweep_id="registered",
                images=make_images(2.0),
                poses_mm=make_poses(0.0),
                probe_geometry=ProbeGeometry(width_mm=20.0, depth_mm=20.0),
                world_transform_mm=transform,
            ),
        )
    )

    result = fuse_multi_sweep_scene(scene, spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    assert result.bounds_min_mm[2] >= 25.0


def test_fuse_multi_sweep_scene_can_skip_per_sweep_volumes() -> None:
    scene = MultiSweepScene(
        sweeps=(
            make_sweep("a", 1.0, 0.0),
            make_sweep("b", 5.0, 40.0),
        )
    )

    result = fuse_multi_sweep_scene(
        scene,
        spacing_mm=(5.0, 5.0, 5.0),
        pixel_stride=(2, 2),
        include_per_sweep_volumes=False,
    )

    assert result.aggregate_volume.scalar_volume.ndim == 3
    assert all(overlay.fused_volume is None for overlay in result.sweep_overlays)
