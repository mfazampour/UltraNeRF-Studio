"""Application orchestration for multi-sweep visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import json

import numpy as np

from visualization.alignment_validation import AlignmentValidationResult, validate_multi_sweep_alignment
from visualization.app import NerfLaunchConfig
from visualization.multi_sweep import MultiSweepScene
from visualization.multi_sweep_loader import load_multi_sweep_scene_from_manifest
from visualization.multi_sweep_ui import MultiSweepSceneController
from visualization.multi_sweep_volume import MultiSweepFusionResult
from visualization.render_controller import RenderController
from visualization.sweep_volume import FusionDevice, FusionReductionMode


@dataclass
class MultiSweepVisualizationAppState:
    """Prepared state for a multi-sweep visualization session."""

    manifest_path: Path | None
    scene: MultiSweepScene
    scene_controller: MultiSweepSceneController
    fusion_result: MultiSweepFusionResult
    alignment_validation: AlignmentValidationResult
    preset_name: str
    cache_root: Path | None = None
    fusion_device: FusionDevice = "auto"
    reduction_mode: FusionReductionMode = "max"
    startup_profile_log_path: Path | None = None
    startup_profile_timings_ms: dict[str, float] | None = None

    @property
    def probe_geometry(self):
        return self.scene.active_sweep.probe_geometry


@dataclass
class MultiSweepLaunchSession:
    """Live objects after launching the multi-sweep napari app."""

    viewer: Any
    ui_controller: Any
    scene_controller: MultiSweepSceneController
    render_controller: RenderController | None


class StartupProfiler:
    """Collect simple wall-clock timings for the multi-sweep startup path."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = self._start
        self.timings_ms: dict[str, float] = {}

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings_ms[str(stage)] = float((now - self._last) * 1000.0)
        self._last = now

    def mark_total(self) -> None:
        now = time.perf_counter()
        self.timings_ms["total"] = float((now - self._start) * 1000.0)


def _resolve_startup_profile_log_path(manifest_path: Path | None) -> Path:
    stem = manifest_path.stem if manifest_path is not None else "multi_sweep"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("logs") / "visualization" / "profiling" / f"{stem}_{timestamp}.json"


def _write_startup_profile_log(path: Path, *, manifest_path: Path | None, timings_ms: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o777)
    except OSError:
        pass
    payload = {
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "timings_ms": {str(k): float(v) for k, v in timings_ms.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    try:
        path.chmod(0o666)
    except OSError:
        pass


class _FallbackReviewPanelsWidget:
    """Non-Qt placeholder used in headless tests without a QApplication."""

    def __init__(self, widgets: tuple[Any, ...]) -> None:
        self.widgets = widgets


def _can_build_multi_view_workspace(viewer: Any) -> bool:
    """Return True when the main napari viewer exposes embeddable Qt widgets."""
    window = getattr(viewer, "window", None)
    if window is None:
        return False
    if not hasattr(window, "_qt_window"):
        return False
    if not (hasattr(window, "_qt_viewer") or hasattr(window, "qt_viewer")):
        return False
    try:
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError:
        return False
    return QApplication.instance() is not None


def _build_review_panels_widget(comparison_widget: Any, render_widget: Any | None) -> Any:
    """Create a vertically stacked right-side review panel when Qt is available."""
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QSplitter, QVBoxLayout, QWidget
    except ModuleNotFoundError:
        widgets = (comparison_widget,) if render_widget is None else (comparison_widget, render_widget)
        return _FallbackReviewPanelsWidget(widgets)

    if QApplication.instance() is None:
        widgets = (comparison_widget,) if render_widget is None else (comparison_widget, render_widget)
        return _FallbackReviewPanelsWidget(widgets)

    review_widget = QWidget()
    review_widget.setMinimumWidth(460)
    review_layout = QVBoxLayout(review_widget)
    review_layout.setContentsMargins(0, 0, 0, 0)
    review_splitter = QSplitter()
    review_splitter.setOrientation(Qt.Vertical)
    review_splitter.addWidget(comparison_widget)

    if render_widget is not None:
        review_splitter.addWidget(render_widget)
        review_splitter.setSizes([420, 420])
    else:
        review_splitter.setSizes([840])

    review_layout.addWidget(review_splitter)
    return review_widget


def _hide_main_viewer_side_docks(viewer: Any) -> None:
    """Hide napari's built-in layer list/controls in the main embedded viewer."""
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        qt_viewer = getattr(viewer.window, "qt_viewer", None)
    if qt_viewer is None:
        return
    for name in ("dockLayerList", "_dockLayerList", "dockLayerControls", "_dockLayerControls"):
        dock = getattr(qt_viewer, name, None)
        if dock is not None:
            try:
                dock.hide()
            except Exception:
                pass


def _install_multi_view_workspace(
    viewer: Any,
    *,
    probe_controls_widget: Any,
    multi_sweep_controls_widget: Any,
    sweep_selection_widget: Any,
    comparison_panel: Any,
    render_panel: Any | None,
) -> None:
    """Embed the main 3D viewer, review panels, and control columns into one workspace."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSplitter, QWidget, QVBoxLayout

    _hide_main_viewer_side_docks(viewer)

    main_qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if main_qt_viewer is None:
        main_qt_viewer = getattr(viewer.window, "qt_viewer", None)
    if main_qt_viewer is None:
        raise RuntimeError("Main napari viewer does not expose an embeddable Qt viewer")

    def _build_column(*widgets: Any, minimum_width: int) -> QWidget:
        column = QWidget()
        column.setMinimumWidth(minimum_width)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        return column

    left_controls_column = _build_column(probe_controls_widget, minimum_width=300)

    review_stack = QSplitter()
    review_stack.setOrientation(Qt.Vertical)
    review_stack.addWidget(comparison_panel.widget)
    if render_panel is not None:
        review_stack.addWidget(render_panel.widget)
        review_stack.setSizes([420, 420])
    else:
        review_stack.setSizes([840])
    review_stack.setChildrenCollapsible(False)

    review_column = _build_column(review_stack, minimum_width=500)

    right_controls_stack = QSplitter()
    right_controls_stack.setOrientation(Qt.Vertical)
    right_controls_stack.addWidget(multi_sweep_controls_widget)
    right_controls_stack.addWidget(sweep_selection_widget)
    right_controls_stack.setSizes([280, 420])
    right_controls_stack.setChildrenCollapsible(False)
    right_controls_column = _build_column(right_controls_stack, minimum_width=320)

    root_splitter = QSplitter()
    root_splitter.setOrientation(Qt.Horizontal)
    root_splitter.addWidget(left_controls_column)
    root_splitter.addWidget(main_qt_viewer)
    root_splitter.addWidget(review_column)
    root_splitter.addWidget(right_controls_column)
    root_splitter.setSizes([320, 1180, 520, 340])
    root_splitter.setChildrenCollapsible(False)
    root_splitter.setStretchFactor(0, 0)
    root_splitter.setStretchFactor(1, 1)
    root_splitter.setStretchFactor(2, 0)
    root_splitter.setStretchFactor(3, 0)

    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(root_splitter)
    viewer.window._qt_window.setCentralWidget(central_widget)


def resolve_multi_sweep_render_image_shape(
    scene: MultiSweepScene,
    *,
    render_height: int | None = None,
    render_width: int | None = None,
    active_sweep_id: str | None = None,
) -> tuple[int, int]:
    """Resolve the NeRF render image size from the active sweep."""
    active_sweep = scene.get_sweep(active_sweep_id or scene.active_sweep_id)
    height, width = active_sweep.image_shape
    if render_height is not None:
        height = int(render_height)
    if render_width is not None:
        width = int(render_width)
    if height <= 0 or width <= 0:
        raise ValueError("render_height and render_width must be positive")
    return (height, width)


def build_multi_sweep_render_controller(
    state: MultiSweepVisualizationAppState,
    nerf_config: NerfLaunchConfig,
    *,
    nerf_session_factory: Any | None = None,
) -> RenderController:
    """Build a checkpoint-backed render controller for a multi-sweep session."""
    checkpoint_path = Path(nerf_config.checkpoint_path)
    config_path = Path(nerf_config.config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if nerf_session_factory is None:
        from visualization.nerf_session import NerfSession

        nerf_session_factory = NerfSession.from_checkpoint

    image_shape = nerf_config.render_image_shape or resolve_multi_sweep_render_image_shape(state.scene)
    active_sweep = state.scene.active_sweep
    nerf_session = nerf_session_factory(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        image_shape=image_shape,
        probe_width_mm=active_sweep.probe_geometry.width_mm,
        probe_depth_mm=active_sweep.probe_geometry.depth_mm,
        device=nerf_config.device,
    )
    return RenderController(
        nerf_session=nerf_session,
        trigger_mode=nerf_config.trigger_mode,
        min_render_interval_s=float(nerf_config.min_render_interval_ms) / 1000.0,
    )


def prepare_multi_sweep_visualization_app(
    *,
    manifest_path: str | Path,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pixel_stride: tuple[int, int] = (2, 2),
    preset_name: str = "soft_tissue",
    cache_root: str | Path | None = None,
    fusion_device: FusionDevice = "auto",
    reduction_mode: FusionReductionMode = "max",
) -> MultiSweepVisualizationAppState:
    """Load and validate a multi-sweep scene for visualization."""
    profiler = StartupProfiler()
    manifest = Path(manifest_path)
    scene = load_multi_sweep_scene_from_manifest(manifest)
    profiler.mark("load_manifest_and_sweeps")
    scene_controller = MultiSweepSceneController(
        scene,
        spacing_mm=spacing_mm,
        pixel_stride=pixel_stride,
        fusion_device=fusion_device,
        reduction_mode=reduction_mode,
    )
    fusion_result = scene_controller.build_fusion_result()
    profiler.mark("build_initial_fusion")
    alignment_validation = validate_multi_sweep_alignment(scene)
    profiler.mark("validate_alignment")
    profiler.mark_total()
    startup_profile_log_path = _resolve_startup_profile_log_path(manifest)
    _write_startup_profile_log(
        startup_profile_log_path,
        manifest_path=manifest,
        timings_ms=profiler.timings_ms,
    )
    return MultiSweepVisualizationAppState(
        manifest_path=manifest,
        scene=scene,
        scene_controller=scene_controller,
        fusion_result=fusion_result,
        alignment_validation=alignment_validation,
        preset_name=preset_name,
        cache_root=Path(cache_root) if cache_root is not None else None,
        fusion_device=fusion_device,
        reduction_mode=reduction_mode,
        startup_profile_log_path=startup_profile_log_path,
        startup_profile_timings_ms=dict(profiler.timings_ms),
    )


def launch_multi_sweep_visualization_app(
    state: MultiSweepVisualizationAppState,
    *,
    initial_pose_index: int = 0,
    render_controller: RenderController | None = None,
    nerf_config: NerfLaunchConfig | None = None,
) -> MultiSweepLaunchSession:
    """Launch a napari multi-sweep session."""
    profiler = StartupProfiler()
    from visualization.comparison_panel import create_comparison_panel
    from visualization.multi_sweep_napari_ui import MultiSweepVisualizationUIController
    from visualization.multi_sweep_ui import create_multi_sweep_controls, create_sweep_selection_controls
    from visualization.probe_controls import create_probe_controls
    from visualization.render_panel import create_render_panel
    from visualization.volume_viewer import launch_basic_volume_viewer

    if render_controller is None and nerf_config is not None:
        render_controller = build_multi_sweep_render_controller(state, nerf_config)

    viewer = launch_basic_volume_viewer(
        state.fusion_result.aggregate_volume,
        viewer_title=f"UltraNeRF Multi-Sweep Volume: {state.manifest_path.stem}",
        preset_name=state.preset_name,
        layer_kwargs={"name": "sweep_volume__aggregate"},
    )
    profiler.mark("launch_main_viewer")
    ui_controller = MultiSweepVisualizationUIController(
        viewer,
        state,
        render_controller=render_controller,
    )
    if hasattr(viewer, "window"):
        use_multi_view_workspace = _can_build_multi_view_workspace(viewer)

        multi_sweep_controls = create_multi_sweep_controls(
            state.scene_controller,
            on_state_changed=ui_controller.handle_multi_sweep_state_change,
        )
        ui_controller.attach_multi_sweep_controls(multi_sweep_controls)

        sweep_selection_controls = create_sweep_selection_controls(
            state.scene_controller,
            on_apply=ui_controller.handle_multi_sweep_state_change,
        )
        ui_controller.attach_sweep_selection_controls(sweep_selection_controls)

        probe_controls = create_probe_controls(
            ui_controller,
            num_frames=state.scene.active_sweep.frame_count,
        )
        ui_controller.attach_probe_controls(probe_controls)

        if use_multi_view_workspace:
            from visualization.embedded_napari_panels import (
                create_embedded_comparison_panel,
                create_embedded_render_panel,
            )

            comparison_panel = create_embedded_comparison_panel()
            ui_controller.attach_comparison_panel(comparison_panel)

            render_panel = None
            if render_controller is not None:
                render_panel = create_embedded_render_panel(ui_controller)
                ui_controller.attach_render_panel(render_panel)

            _install_multi_view_workspace(
                viewer,
                probe_controls_widget=probe_controls.widget,
                multi_sweep_controls_widget=multi_sweep_controls.widget,
                sweep_selection_widget=sweep_selection_controls.widget,
                comparison_panel=comparison_panel,
                render_panel=render_panel,
            )
            profiler.mark("build_multi_view_workspace")
        else:
            viewer.window.add_dock_widget(multi_sweep_controls.widget, area="right", name="Multi-Sweep Controls")
            viewer.window.add_dock_widget(sweep_selection_controls.widget, area="right", name="Sweep Selection")
            viewer.window.add_dock_widget(probe_controls.widget, area="left", name="Probe Controls")
            comparison_panel = create_comparison_panel()
            ui_controller.attach_comparison_panel(comparison_panel)

            render_panel = None
            if render_controller is not None:
                render_panel = create_render_panel(ui_controller)
                ui_controller.attach_render_panel(render_panel)

            review_widget = _build_review_panels_widget(
                comparison_panel.widget,
                None if render_panel is None else render_panel.widget,
            )
            viewer.window.add_dock_widget(review_widget, area="right", name="Review Panels")
            profiler.mark("build_dock_review_workspace")

    active_sweep = state.scene.active_sweep
    safe_index = min(max(int(initial_pose_index), 0), active_sweep.frame_count - 1)
    ui_controller.initialize(active_sweep.poses_mm[safe_index])
    profiler.mark("initialize_scene_layers")
    profiler.mark_total()
    merged_timings = dict(state.startup_profile_timings_ms or {})
    merged_timings.update(profiler.timings_ms)
    if state.startup_profile_log_path is not None:
        _write_startup_profile_log(
            state.startup_profile_log_path,
            manifest_path=state.manifest_path,
            timings_ms=merged_timings,
        )
        state.startup_profile_timings_ms = merged_timings
    return MultiSweepLaunchSession(
        viewer=viewer,
        ui_controller=ui_controller,
        scene_controller=state.scene_controller,
        render_controller=render_controller,
    )
