"""Application orchestration for multi-sweep visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visualization.alignment_validation import AlignmentValidationResult, validate_multi_sweep_alignment
from visualization.app import NerfLaunchConfig
from visualization.multi_sweep import MultiSweepScene
from visualization.multi_sweep_loader import load_multi_sweep_scene_from_manifest
from visualization.multi_sweep_ui import MultiSweepSceneController
from visualization.multi_sweep_volume import MultiSweepFusionResult
from visualization.render_controller import RenderController
from visualization.sweep_volume import FusionDevice


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
) -> MultiSweepVisualizationAppState:
    """Load and validate a multi-sweep scene for visualization."""
    manifest = Path(manifest_path)
    scene = load_multi_sweep_scene_from_manifest(manifest)
    scene_controller = MultiSweepSceneController(
        scene,
        spacing_mm=spacing_mm,
        pixel_stride=pixel_stride,
        fusion_device=fusion_device,
    )
    fusion_result = scene_controller.build_fusion_result()
    alignment_validation = validate_multi_sweep_alignment(scene)
    return MultiSweepVisualizationAppState(
        manifest_path=manifest,
        scene=scene,
        scene_controller=scene_controller,
        fusion_result=fusion_result,
        alignment_validation=alignment_validation,
        preset_name=preset_name,
        cache_root=Path(cache_root) if cache_root is not None else None,
        fusion_device=fusion_device,
    )


def launch_multi_sweep_visualization_app(
    state: MultiSweepVisualizationAppState,
    *,
    initial_pose_index: int = 0,
    render_controller: RenderController | None = None,
    nerf_config: NerfLaunchConfig | None = None,
) -> MultiSweepLaunchSession:
    """Launch a napari multi-sweep session."""
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
    ui_controller = MultiSweepVisualizationUIController(
        viewer,
        state,
        render_controller=render_controller,
    )
    if hasattr(viewer, "window"):
        multi_sweep_controls = create_multi_sweep_controls(
            state.scene_controller,
            on_state_changed=ui_controller.handle_multi_sweep_state_change,
        )
        viewer.window.add_dock_widget(multi_sweep_controls.widget, area="left", name="Multi-Sweep Controls")
        ui_controller.attach_multi_sweep_controls(multi_sweep_controls)

        sweep_selection_controls = create_sweep_selection_controls(
            state.scene_controller,
            on_apply=ui_controller.handle_multi_sweep_state_change,
        )
        viewer.window.add_dock_widget(sweep_selection_controls.widget, area="right", name="Sweep Selection")
        ui_controller.attach_sweep_selection_controls(sweep_selection_controls)

        probe_controls = create_probe_controls(
            ui_controller,
            num_frames=state.scene.active_sweep.frame_count,
        )
        viewer.window.add_dock_widget(probe_controls.widget, area="left", name="Probe Controls")
        ui_controller.attach_probe_controls(probe_controls)

        comparison_panel = create_comparison_panel()
        viewer.window.add_dock_widget(comparison_panel.widget, area="right", name="Nearest Recorded Frame")
        ui_controller.attach_comparison_panel(comparison_panel)

        if render_controller is not None:
            render_panel = create_render_panel(ui_controller)
            viewer.window.add_dock_widget(render_panel.widget, area="right", name="NeRF Render")
            ui_controller.attach_render_panel(render_panel)

    active_sweep = state.scene.active_sweep
    safe_index = min(max(int(initial_pose_index), 0), active_sweep.frame_count - 1)
    ui_controller.initialize(active_sweep.poses_mm[safe_index])
    return MultiSweepLaunchSession(
        viewer=viewer,
        ui_controller=ui_controller,
        scene_controller=state.scene_controller,
        render_controller=render_controller,
    )
