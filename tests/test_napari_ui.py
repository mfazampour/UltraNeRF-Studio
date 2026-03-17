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


class FakeRenderPanel:
    def __init__(self):
        self.status = None
        self.metadata = None
        self.image = None

    def set_status(self, text):
        self.status = text

    def set_metadata(self, text):
        self.metadata = text

    def set_image(self, image):
        self.image = np.asarray(image)


class FakeProbeControls:
    def __init__(self):
        self.values = None

    def set_pose_values(self, **kwargs):
        self.values = dict(kwargs)


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


def test_render_panel_receives_status_and_image_updates():
    state = make_app_state()
    viewer = FakeViewer()
    nerf_session = FakeNerfSession()
    render_controller = RenderController(nerf_session=nerf_session, trigger_mode="manual")
    controller = VisualizationUIController(viewer, state, render_controller=render_controller)
    render_panel = FakeRenderPanel()
    controller.attach_render_panel(render_panel)
    controller.initialize()

    assert render_panel.status == "Ready"
    assert render_panel.metadata == "No render available"

    controller.render_now()

    assert render_panel.status == "Rendered"
    assert render_panel.metadata == "Image shape: (4, 5)"
    assert render_panel.image.shape == (4, 5)


def test_probe_controls_receive_pose_updates_and_can_snap_to_recorded_pose():
    state = make_app_state()
    viewer = FakeViewer()
    controller = VisualizationUIController(viewer, state)
    probe_controls = FakeProbeControls()
    controller.attach_probe_controls(probe_controls)
    controller.initialize()

    assert np.allclose(probe_controls.values["origin_mm"], [0.0, 0.0, 0.0])
    assert probe_controls.values["recorded_index"] == 0

    controller.set_probe_pose_from_components(
        origin_mm=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert np.allclose(controller.state.probe_pose_mm[:3, 3], [10.0, 0.0, 0.0])

    controller.snap_probe_to_nearest_recorded_pose()
    assert np.allclose(controller.state.probe_pose_mm[:3, 3], [10.0, 0.0, 0.0])
