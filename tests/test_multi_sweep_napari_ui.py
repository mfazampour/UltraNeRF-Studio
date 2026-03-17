import numpy as np

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.multi_sweep_app import MultiSweepVisualizationAppState
from visualization.multi_sweep_napari_ui import MultiSweepVisualizationUIController, _color_to_hex
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
    )


def test_color_to_hex_converts_normalized_rgb_tuple() -> None:
    assert _color_to_hex((1.0, 0.5, 0.0), default="#000000") == "#ff8000"


def test_initialize_adds_multi_sweep_layers_and_probe() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)

    scene_state = controller.initialize()

    assert "sweep_volume__aggregate" in viewer.layers
    assert "trajectory_path__a" in viewer.layers
    assert "trajectory_path__b" in viewer.layers
    assert "probe_origin" in viewer.layers
    assert scene_state.comparison_payload["matched_sweep_id"] in ("a", "b")


def test_handle_multi_sweep_state_change_updates_visibility() -> None:
    state = make_state()
    viewer = FakeViewer()
    controller = MultiSweepVisualizationUIController(viewer, state)
    controller.initialize()

    state.scene_controller.set_show_aggregate_volume(False)
    controller.handle_multi_sweep_state_change(state.scene_controller.state)

    assert viewer.layers["sweep_volume__aggregate"].visible is False
    assert viewer.layers["sweep_volume__a"].visible is True


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
