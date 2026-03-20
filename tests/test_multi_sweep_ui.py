import numpy as np
import pytest

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.multi_sweep_ui import MultiSweepSceneController
from visualization.transforms import ProbeGeometry


def make_images(value: float) -> np.ndarray:
    return np.full((2, 3, 4), value, dtype=np.float32)


def make_poses(offset_mm: float) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = offset_mm
    return poses


def make_scene() -> MultiSweepScene:
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=20.0)
    return MultiSweepScene(
        sweeps=(
            SweepRecord("a", make_images(1.0), make_poses(0.0), geometry),
            SweepRecord("b", make_images(2.0), make_poses(40.0), geometry),
        )
    )


def test_controller_tracks_active_enabled_and_policy_state() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    assert controller.state.visible_sweep_ids == ("a",)

    state = controller.set_active_sweep("b")
    assert state.active_sweep_id == "b"
    assert state.visible_sweep_ids == ("a",)

    state = controller.set_comparison_policy("active_only")
    assert state.comparison_policy == "active_only"

    state = controller.set_show_aggregate_volume(False)
    assert state.show_aggregate_volume is False


def test_controller_tracks_visible_sweeps_separately_from_enabled_sweeps() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    state = controller.set_enabled_sweeps(("a", "b"))
    assert state.enabled_sweep_ids == ("a", "b")

    state = controller.set_visible_sweeps(("b",))
    assert state.visible_sweep_ids == ("b",)


def test_controller_requires_at_least_one_enabled_sweep() -> None:
    controller = MultiSweepSceneController(make_scene())

    with pytest.raises(ValueError, match="At least one sweep"):
        controller.set_enabled_sweeps(tuple())


def test_controller_builds_fusion_result_for_enabled_subset() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))
    controller.set_enabled_sweeps(("b",))
    state = controller.state

    assert state.enabled_sweep_ids == ("b",)
    assert state.active_sweep_id == "b"


def test_controller_builds_fusion_result_for_visible_subset() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))
    controller.set_visible_sweeps(("b",))
    controller.set_show_aggregate_volume(False)

    result = controller.build_fusion_result()

    assert result.enabled_sweep_ids == ("b",)
    assert tuple(overlay.sweep_id for overlay in result.sweep_overlays) == ("b",)


def test_controller_skips_per_sweep_volumes_while_aggregate_mode_is_enabled() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))

    result = controller.build_fusion_result()

    assert result.aggregate_volume.scalar_volume.ndim == 3
    assert all(overlay.fused_volume is None for overlay in result.sweep_overlays)
    assert result.enabled_sweep_ids == ("a", "b")


def test_non_aggregate_mode_defaults_to_active_sweep_visibility() -> None:
    controller = MultiSweepSceneController(make_scene(), spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))
    controller.set_show_aggregate_volume(False)

    result = controller.build_fusion_result()

    assert controller.state.visible_sweep_ids == ("a",)
    assert tuple(overlay.sweep_id for overlay in result.sweep_overlays) == ("a",)
