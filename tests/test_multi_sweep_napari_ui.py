import numpy as np

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.multi_sweep_app import MultiSweepVisualizationAppState
from visualization.multi_sweep_napari_ui import (
    MultiSweepVisualizationUIController,
    _color_to_hex,
    _compute_aggregate_contrast_limits,
)
from visualization.multi_sweep_ui import MultiSweepSceneController
from visualization.multi_sweep_volume import fuse_multi_sweep_scene
from visualization.render_controller import RenderController
from visualization.transforms import ProbeGeometry
from visualization.alignment_validation import validate_multi_sweep_alignment


class FakeLayer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs
        self.visible = True


class FakeViewer:
    def __init__(self):
        self.layers = {}

    def add_image(self, data, **kwargs):
        layer = FakeLayer(data, **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer

    def add_points(self, data, **kwargs):
        layer = FakeLayer(np.asarray(data), **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer

    def add_vectors(self, data, **kwargs):
        layer = FakeLayer(np.asarray(data), **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer

    def add_shapes(self, data, **kwargs):
        layer = FakeLayer(data, **kwargs)
        self.layers[kwargs["name"]] = layer
        return layer


class FakeNerfSession:
    def __init__(self):
        self.calls = []

    def render_pose(self, pose_mm, **kwargs):
        self.calls.append(np.asarray(pose_mm))
        return {"intensity_map": np.full((4, 5), float(len(self.calls)), dtype=np.float32)}


class FakeProbeControls:
    def __init__(self):
        self.values = None
        self.num_frames = None

    def set_pose_values(self, **kwargs):
        self.values = dict(kwargs)

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames


class FakeComparisonPanel:
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


class FakeScaledPanel:
    def __init__(self):
        self.image = None
        self.scale_mm = None

    def set_status(self, text):
        self.status = text

    def set_metadata(self, text):
        self.metadata = text

    def set_image(self, image, *, scale_mm=None):
        self.image = np.asarray(image)
        self.scale_mm = tuple(scale_mm) if scale_mm is not None else None


def make_state():
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=20.0)
    images_a = np.ones((2, 4, 5), dtype=np.float32)
    images_b = np.full((2, 4, 5), 2.0, dtype=np.float32)
    poses_a = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses_b = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses_b[:, 0, 3] = 30.0
    scene = MultiSweepScene(
        sweeps=(
            SweepRecord("a", images_a, poses_a, geometry, display_name="Sweep A", color_rgb=(0.9, 0.5, 0.2)),
            SweepRecord("b", images_b, poses_b, geometry, display_name="Sweep B", color_rgb=(0.2, 0.6, 0.9)),
        )
    )
    controller = MultiSweepSceneController(scene, spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))
    fusion_result = fuse_multi_sweep_scene(scene, spacing_mm=(5.0, 5.0, 5.0), pixel_stride=(2, 2))
    return MultiSweepVisualizationAppState(
        manifest_path=None,
        scene=scene,
        scene_controller=controller,
        fusion_result=fusion_result,
        alignment_validation=validate_multi_sweep_alignment(scene),
        preset_name="soft_tissue",
        fusion_device="cpu",
        reduction_mode="max",
    )


def test_color_to_hex_converts_normalized_rgb_tuple() -> None:
    assert _color_to_hex((1.0, 0.5, 0.0), default="#000000") == "#ff8000"


def test_compute_aggregate_contrast_limits_suppresses_low_signal_background() -> None:
    data = np.concatenate(
        [
            np.zeros(500, dtype=np.float32),
            np.full(400, 0.01, dtype=np.float32),
            np.linspace(0.5, 2.0, 100, dtype=np.float32),
        ]
    ).reshape(20, 10, 5)

    lower, upper = _compute_aggregate_contrast_limits(data)

    assert lower > 0.0
    assert upper > lower


def test_initialize_adds_multi_sweep_layers_and_probe() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)

    scene_state = controller.initialize()

    assert "sweep_volume__aggregate" in viewer.layers
    assert "trajectory_path__a" in viewer.layers
    assert "trajectory_path__b" not in viewer.layers
    assert "sweep_volume__a" not in viewer.layers
    assert "probe_origin" in viewer.layers
    assert scene_state.comparison_payload["matched_sweep_id"] in ("a", "b")
    assert "trajectory_axes__a" not in viewer.layers
    assert "trajectory_axes__b" not in viewer.layers


def test_initialize_reuses_existing_aggregate_layer() -> None:
    state = make_state()
    viewer = FakeViewer()
    preexisting_layer = viewer.add_image(np.zeros((2, 2, 2), dtype=np.float32), name="sweep_volume__aggregate")
    controller = MultiSweepVisualizationUIController(viewer, state)

    controller.initialize()

    assert controller._layers["sweep_volume__aggregate"] is preexisting_layer
    assert list(viewer.layers).count("sweep_volume__aggregate") == 1


def test_handle_multi_sweep_state_change_updates_visibility() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    controller.initialize()

    state.scene_controller.set_show_aggregate_volume(False)
    controller.handle_multi_sweep_state_change(state.scene_controller.state)

    assert viewer.layers["sweep_volume__aggregate"].visible is False
    assert viewer.layers["sweep_volume__a"].visible is True


def test_aggregate_mode_shows_only_active_sweep_trajectories() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    controller.initialize()

    assert viewer.layers["trajectory_path__a"].visible is True
    assert "trajectory_path__b" not in viewer.layers

    controller.set_active_sweep("b")

    assert viewer.layers["trajectory_path__a"].visible is False
    assert viewer.layers["trajectory_path__b"].visible is True


def test_probe_controls_and_comparison_panel_refresh_with_active_sweep() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    probe_controls = FakeProbeControls()
    comparison_panel = FakeComparisonPanel()
    controller.attach_probe_controls(probe_controls)
    controller.attach_comparison_panel(comparison_panel)
    controller.initialize()

    controller.set_active_sweep("b")

    assert probe_controls.num_frames == 2
    assert "Sweep" in comparison_panel.metadata


def test_render_now_updates_multi_sweep_comparison_payload() -> None:
    state = make_state()
    viewer = FakeViewer()
    render_controller = RenderController(nerf_session=FakeNerfSession(), trigger_mode="manual")
    controller = MultiSweepVisualizationUIController(viewer, state, render_controller=render_controller)
    controller.initialize()

    output = controller.render_now()

    assert output["intensity_map"].shape == (4, 5)
    assert controller.state.comparison_payload["matched_sweep_id"] in ("a", "b")


def test_snap_probe_to_nearest_uses_all_enabled_sweeps_when_requested() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    controller.initialize()
    state.scene_controller.set_comparison_policy("all_enabled")

    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 29.0
    controller.set_probe_pose(pose)
    updated = controller.snap_probe_to_nearest_recorded_pose()

    assert state.scene_controller.state.active_sweep_id == "b"
    assert np.allclose(updated.probe_pose_mm[:3, 3], np.array([30.0, 0.0, 0.0], dtype=np.float32))


def test_embedded_panels_receive_physical_pixel_scale() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    comparison_panel = FakeScaledPanel()
    render_panel = FakeScaledPanel()
    controller.attach_comparison_panel(comparison_panel)
    controller.attach_render_panel(render_panel)
    render_controller = RenderController(nerf_session=FakeNerfSession(), trigger_mode="manual")
    controller.render_controller = render_controller

    controller.initialize()
    controller.render_now()

    assert comparison_panel.scale_mm == (5.0, 4.0)
    assert render_panel.scale_mm == (5.0, 4.0)
