import numpy as np

from visualization.app import VisualizationAppState
from visualization.napari_ui import VisualizationUIController
from visualization.render_controller import RenderController
from visualization.sweep_volume import FusedSweepVolume
from visualization.trajectory import build_trajectory_overlay
from visualization.transforms import ProbeGeometry


class FakeLayer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs


class FakeViewer:
    def __init__(self):
        self.layers = {}

    def add_points(self, data, **kwargs):
        layer = FakeLayer(np.asarray(data), **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer

    def add_vectors(self, data, **kwargs):
        layer = FakeLayer(np.asarray(data), **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer

    def add_shapes(self, data, **kwargs):
        normalized = [np.asarray(item) for item in data]
        layer = FakeLayer(normalized, **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer


class FakeNerfSession:
    def __init__(self):
        self.calls = []

    def render_pose(self, pose_mm, **kwargs):
        self.calls.append(np.asarray(pose_mm))
        return {"intensity_map": np.full((4, 5), float(len(self.calls)), dtype=np.float32)}


def make_app_state():
    images = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(3)], axis=0)
    poses[1, 0, 3] = 10.0
    poses[2, 1, 3] = 5.0
    fused = FusedSweepVolume(
        scalar_volume=np.ones((6, 7, 8), dtype=np.float32),
        weight_volume=np.ones((6, 7, 8), dtype=np.float32),
        spacing_mm=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin_mm=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_min_mm=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_mm=np.array([5.0, 6.0, 7.0], dtype=np.float32),
    )
    return VisualizationAppState(
        dataset_dir=None,
        fused_volume=fused,
        trajectory=build_trajectory_overlay(poses, axis_stride=1, axis_length_mm=4.0),
        cache_path=None,
        cache_used=False,
        images=images,
        poses_mm=poses,
        probe_geometry=ProbeGeometry(width_mm=20.0, depth_mm=40.0),
        preset_name="soft_tissue",
    )


def test_initialize_adds_trajectory_and_probe_layers():
    state = make_app_state()
    viewer = FakeViewer()

    controller = VisualizationUIController(viewer, state)
    scene_state = controller.initialize()

    assert "trajectory_path" in viewer.layers
    assert "trajectory_centers" in viewer.layers
    assert "trajectory_axes" in viewer.layers
    assert "probe_origin" in viewer.layers
    assert "probe_axes" in viewer.layers
    assert "probe_scan_plane" in viewer.layers
    assert "probe_beam_line" in viewer.layers
    assert "probe_face_line" in viewer.layers
    assert scene_state.comparison_payload["matched_index"] == 0


def test_set_probe_pose_updates_probe_layers_and_comparison_match():
    state = make_app_state()
    viewer = FakeViewer()
    controller = VisualizationUIController(viewer, state)
    controller.initialize()

    updated_state = controller.set_probe_to_recorded_pose(1)

    probe_origin = viewer.layers["probe_origin"].data
    assert np.allclose(probe_origin[0], [10.0, 0.0, 0.0])
    assert updated_state.comparison_payload["matched_index"] == 1


def test_render_now_uses_render_controller_and_updates_scene_state():
    state = make_app_state()
    viewer = FakeViewer()
    nerf_session = FakeNerfSession()
    render_controller = RenderController(nerf_session=nerf_session, trigger_mode="manual")
    controller = VisualizationUIController(viewer, state, render_controller=render_controller)
    controller.initialize()

    output = controller.render_now()

    assert output["intensity_map"].shape == (4, 5)
    assert len(nerf_session.calls) == 1
    assert np.allclose(controller.state.rendered_output["intensity_map"], output["intensity_map"])
