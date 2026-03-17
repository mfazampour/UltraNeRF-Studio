import json
from pathlib import Path

import numpy as np

from visualization.app import (
    NerfLaunchConfig,
    build_or_load_fused_volume,
    build_render_controller,
    launch_visualization_app,
    prepare_visualization_app,
    resolve_render_image_shape,
)


def make_dataset(tmp_path: Path):
    images = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    poses[1, 0, 3] = 10.0
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "poses.npy", poses)
    return images, poses


def test_build_or_load_fused_volume_writes_and_reuses_cache(tmp_path):
    make_dataset(tmp_path)
    cache_path = tmp_path / "cache" / "volume.npz"

    fused_a, cache_a, cache_used_a, _, _ = build_or_load_fused_volume(
        dataset_dir=tmp_path,
        probe_geometry=type("Probe", (), {"width_mm": 4.0, "depth_mm": 4.0})(),
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=cache_path,
    )
    fused_b, cache_b, cache_used_b, _, _ = build_or_load_fused_volume(
        dataset_dir=tmp_path,
        probe_geometry=type("Probe", (), {"width_mm": 4.0, "depth_mm": 4.0})(),
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=cache_path,
    )

    assert cache_a == cache_path
    assert cache_b == cache_path
    assert not cache_used_a
    assert cache_used_b
    assert np.allclose(fused_a.scalar_volume, fused_b.scalar_volume)


def test_prepare_visualization_app_returns_volume_and_trajectory(tmp_path):
    images, poses = make_dataset(tmp_path)

    state = prepare_visualization_app(
        dataset_dir=tmp_path,
        probe_width_mm=4.0,
        probe_depth_mm=4.0,
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=None,
        preset_name="soft_tissue",
    )

    assert state.images.shape == images.shape
    assert state.poses_mm.shape == poses.shape
    assert state.fused_volume.scalar_volume.ndim == 3
    assert state.trajectory.centers_mm.shape[0] == poses.shape[0]
    assert state.preset_name == "soft_tissue"


def test_resolve_render_image_shape_uses_dataset_shape_and_overrides():
    images = np.zeros((2, 4, 5), dtype=np.float32)

    assert resolve_render_image_shape(images) == (4, 5)
    assert resolve_render_image_shape(images, render_height=6) == (6, 5)
    assert resolve_render_image_shape(images, render_width=7) == (4, 7)
    assert resolve_render_image_shape(images, render_height=8, render_width=9) == (8, 9)


def test_build_render_controller_uses_nerf_session_factory_and_trigger_mode(tmp_path):
    make_dataset(tmp_path)
    state = prepare_visualization_app(
        dataset_dir=tmp_path,
        probe_width_mm=4.0,
        probe_depth_mm=4.0,
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=None,
        preset_name="soft_tissue",
    )
    checkpoint_path = tmp_path / "model.tar"
    config_path = tmp_path / "config.txt"
    checkpoint_path.write_text("checkpoint")
    config_path.write_text("config")
    call_log = {}

    def fake_nerf_session_factory(**kwargs):
        call_log.update(kwargs)
        return object()

    controller = build_render_controller(
        state,
        NerfLaunchConfig(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            trigger_mode="on_pose_change",
            min_render_interval_ms=250.0,
            render_image_shape=(6, 7),
            device="cpu",
        ),
        nerf_session_factory=fake_nerf_session_factory,
    )

    assert controller.trigger_mode == "on_pose_change"
    assert controller.min_render_interval_s == 0.25
    assert call_log["image_shape"] == (6, 7)
    assert call_log["probe_width_mm"] == 4.0
    assert call_log["probe_depth_mm"] == 4.0
    assert call_log["device"] == "cpu"


def test_launch_visualization_app_initializes_ui_controller(monkeypatch, tmp_path):
    make_dataset(tmp_path)
    state = prepare_visualization_app(
        dataset_dir=tmp_path,
        probe_width_mm=4.0,
        probe_depth_mm=4.0,
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=None,
        preset_name="soft_tissue",
    )

    class FakeLayer:
        def __init__(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs

    class FakeAxes:
        visible = False

    class FakeScaleBar:
        visible = False
        unit = None

    class FakeWindow:
        def __init__(self):
            self.calls = []

        def add_dock_widget(self, widget, area=None, name=None):
            self.calls.append({"widget": widget, "area": area, "name": name})

    class FakeViewer:
        def __init__(self, *args, **kwargs):
            self.layers = {}
            self.axes = FakeAxes()
            self.scale_bar = FakeScaleBar()
            self.window = FakeWindow()

        def add_image(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_points(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_vectors(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_shapes(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

    class FakeNapari:
        Viewer = FakeViewer

    class FakeProbeControls:
        def __init__(self, ui_controller, num_frames):
            self.ui_controller = ui_controller
            self.num_frames = num_frames
            self.widget = object()

        def set_pose_values(self, **kwargs):
            self.values = kwargs

    import sys

    monkeypatch.setitem(sys.modules, "napari", FakeNapari)
    monkeypatch.setattr("visualization.probe_controls.create_probe_controls", lambda ui_controller, num_frames: FakeProbeControls(ui_controller, num_frames))

    session = launch_visualization_app(state)

    assert "sweep_volume" in session.viewer.layers
    assert "trajectory_path" in session.viewer.layers
    assert "probe_origin" in session.viewer.layers
    assert session.ui_controller.state.comparison_payload["matched_index"] == 0
    assert session.render_controller is None
    assert session.ui_controller.probe_controls is not None
    assert session.viewer.window.calls[0]["name"] == "Probe Controls"


def test_launch_visualization_app_builds_render_controller_when_nerf_enabled(monkeypatch, tmp_path):
    make_dataset(tmp_path)
    state = prepare_visualization_app(
        dataset_dir=tmp_path,
        probe_width_mm=4.0,
        probe_depth_mm=4.0,
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=None,
        preset_name="soft_tissue",
    )
    checkpoint_path = tmp_path / "model.tar"
    config_path = tmp_path / "config.txt"
    checkpoint_path.write_text("checkpoint")
    config_path.write_text("config")

    class FakeLayer:
        def __init__(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs

    class FakeAxes:
        visible = False

    class FakeScaleBar:
        visible = False
        unit = None

    class FakeWindow:
        def __init__(self):
            self.calls = []

        def add_dock_widget(self, widget, area=None, name=None):
            self.calls.append({"widget": widget, "area": area, "name": name})

    class FakeViewer:
        def __init__(self, *args, **kwargs):
            self.layers = {}
            self.axes = FakeAxes()
            self.scale_bar = FakeScaleBar()
            self.window = FakeWindow()

        def add_image(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_points(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_vectors(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        def add_shapes(self, data, **kwargs):
            layer = FakeLayer(data, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

    class FakeNapari:
        Viewer = FakeViewer

    class FakeNerfSession:
        def render_pose(self, pose_probe_to_world_mm, **kwargs):
            return {"intensity_map": np.zeros((4, 5), dtype=np.float32)}

    class FakeRenderPanel:
        def __init__(self, ui_controller):
            self.ui_controller = ui_controller
            self.widget = object()

        def set_status(self, text):
            self.status = text

        def set_metadata(self, text):
            self.metadata = text

        def set_image(self, image):
            self.image = image

    class FakeProbeControls:
        def __init__(self, ui_controller, num_frames):
            self.ui_controller = ui_controller
            self.num_frames = num_frames
            self.widget = object()

        def set_pose_values(self, **kwargs):
            self.values = kwargs

    class FakeComparisonPanel:
        def __init__(self):
            self.widget = object()

        def set_status(self, text):
            self.status = text

        def set_metadata(self, text):
            self.metadata = text

        def set_image(self, image):
            self.image = image

    import sys

    monkeypatch.setitem(sys.modules, "napari", FakeNapari)
    monkeypatch.setattr(
        "visualization.app.build_render_controller",
        lambda state, nerf_config: __import__("visualization.render_controller", fromlist=["RenderController"]).RenderController(
            nerf_session=FakeNerfSession(),
            trigger_mode=nerf_config.trigger_mode,
        ),
    )
    monkeypatch.setattr("visualization.render_panel.create_render_panel", lambda ui_controller: FakeRenderPanel(ui_controller))
    monkeypatch.setattr("visualization.probe_controls.create_probe_controls", lambda ui_controller, num_frames: FakeProbeControls(ui_controller, num_frames))
    monkeypatch.setattr("visualization.comparison_panel.create_comparison_panel", lambda: FakeComparisonPanel())

    session = launch_visualization_app(
        state,
        nerf_config=NerfLaunchConfig(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            trigger_mode="manual",
            render_image_shape=(4, 5),
            device="cpu",
        ),
    )

    assert session.render_controller is not None
    assert session.render_controller.trigger_mode == "manual"
    assert session.ui_controller.render_panel is not None
    assert session.ui_controller.probe_controls is not None
    assert session.ui_controller.comparison_panel is not None
