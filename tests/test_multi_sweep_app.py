import json
from pathlib import Path

import numpy as np

from visualization.app import NerfLaunchConfig
from visualization.multi_sweep_app import (
    build_multi_sweep_render_controller,
    launch_multi_sweep_visualization_app,
    prepare_multi_sweep_visualization_app,
    resolve_multi_sweep_render_image_shape,
)


def write_sweep_dir(path: Path, offset_mm: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = np.ones((2, 3, 4), dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = offset_mm
    np.save(path / "images.npy", images)
    np.save(path / "poses.npy", poses)


def write_manifest(tmp_path: Path) -> Path:
    write_sweep_dir(tmp_path / "sweep_a", 0.0)
    write_sweep_dir(tmp_path / "sweep_b", 30.0)
    manifest = {
        "probe_geometry": {"width_mm": 20.0, "depth_mm": 20.0},
        "active_sweep_id": "sweep_a",
        "comparison_policy": "all_enabled",
        "sweeps": [
            {"sweep_id": "sweep_a", "dataset_dir": "sweep_a", "display_name": "Sweep A"},
            {"sweep_id": "sweep_b", "dataset_dir": "sweep_b", "display_name": "Sweep B"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_prepare_multi_sweep_visualization_app_returns_scene_and_alignment(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)

    state = prepare_multi_sweep_visualization_app(
        manifest_path=manifest_path,
        spacing_mm=(5.0, 5.0, 5.0),
        pixel_stride=(2, 2),
        preset_name="soft_tissue",
    )

    assert state.scene.sweep_ids == ("sweep_a", "sweep_b")
    assert state.fusion_result.aggregate_volume.scalar_volume.ndim == 3
    assert len(state.alignment_validation.per_sweep) == 2
    assert state.fusion_device == "auto"
    assert state.reduction_mode == "max"
    assert state.startup_profile_log_path is not None
    assert state.startup_profile_log_path.exists()
    assert "build_initial_fusion" in state.startup_profile_timings_ms


def test_resolve_multi_sweep_render_image_shape_uses_active_sweep() -> None:
    class FakeSweep:
        image_shape = (10, 20)

    class FakeScene:
        active_sweep_id = "a"

        def get_sweep(self, _sweep_id):
            return FakeSweep()

    assert resolve_multi_sweep_render_image_shape(FakeScene()) == (10, 20)
    assert resolve_multi_sweep_render_image_shape(FakeScene(), render_height=30) == (30, 20)
    assert resolve_multi_sweep_render_image_shape(FakeScene(), render_width=40) == (10, 40)


def test_build_multi_sweep_render_controller_uses_active_sweep_geometry(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    state = prepare_multi_sweep_visualization_app(manifest_path=manifest_path)
    checkpoint_path = tmp_path / "model.tar"
    config_path = tmp_path / "config.txt"
    checkpoint_path.write_text("checkpoint")
    config_path.write_text("config")
    call_log = {}

    def fake_nerf_session_factory(**kwargs):
        call_log.update(kwargs)
        return object()

    controller = build_multi_sweep_render_controller(
        state,
        NerfLaunchConfig(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            trigger_mode="manual",
            render_image_shape=(6, 7),
        ),
        nerf_session_factory=fake_nerf_session_factory,
    )

    assert controller.trigger_mode == "manual"
    assert call_log["probe_width_mm"] == 20.0
    assert call_log["probe_depth_mm"] == 20.0
    assert call_log["image_shape"] == (6, 7)


def test_launch_multi_sweep_visualization_app_initializes_fake_viewer(monkeypatch, tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    state = prepare_multi_sweep_visualization_app(manifest_path=manifest_path)

    class FakeLayer:
        def __init__(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs
            self.visible = True

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

    class FakeNapari:
        Viewer = FakeViewer

    class FakeProbeControls:
        def __init__(self, ui_controller, num_frames):
            self.ui_controller = ui_controller
            self.num_frames = num_frames
            self.widget = object()

        def set_pose_values(self, **kwargs):
            self.values = kwargs

        def set_num_frames(self, num_frames):
            self.num_frames = num_frames

    class FakeMultiSweepControls:
        def __init__(self):
            self.widget = object()
            self.refreshed = False

        def refresh(self):
            self.refreshed = True

    class FakeComparisonPanel:
        def __init__(self):
            self.widget = object()

        def set_status(self, text):
            self.status = text

        def set_metadata(self, text):
            self.metadata = text

        def set_image(self, image):
            self.image = image

    class FakeSweepSelectionControls:
        def __init__(self):
            self.widget = object()
            self.refreshed = False

        def refresh(self):
            self.refreshed = True

    import sys

    monkeypatch.setitem(sys.modules, "napari", FakeNapari)
    monkeypatch.setattr(
        "visualization.probe_controls.create_probe_controls",
        lambda ui_controller, num_frames: FakeProbeControls(ui_controller, num_frames),
    )
    monkeypatch.setattr(
        "visualization.multi_sweep_ui.create_multi_sweep_controls",
        lambda controller, on_state_changed=None: FakeMultiSweepControls(),
    )
    monkeypatch.setattr(
        "visualization.multi_sweep_ui.create_sweep_selection_controls",
        lambda controller, on_apply=None: FakeSweepSelectionControls(),
    )
    monkeypatch.setattr(
        "visualization.comparison_panel.create_comparison_panel",
        lambda: FakeComparisonPanel(),
    )

    session = launch_multi_sweep_visualization_app(state)

    assert "sweep_volume__aggregate" in session.viewer.layers
    assert "trajectory_path__sweep_a" in session.viewer.layers
    assert "trajectory_path__sweep_b" not in session.viewer.layers
    assert "probe_origin" in session.viewer.layers
    assert session.ui_controller.state.comparison_payload["matched_sweep_id"] in ("sweep_a", "sweep_b")
    dock_areas = {call["name"]: call["area"] for call in session.viewer.window.calls}
    assert dock_areas["Multi-Sweep Controls"] == "right"
    assert dock_areas["Probe Controls"] == "left"
