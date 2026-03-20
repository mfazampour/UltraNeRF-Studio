"""Application orchestration for sweep visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ultranerf.visualization.render_controller import RenderController, RenderTriggerMode
from ultranerf.visualization.sweep_volume import (
    FusionDevice,
    FusionReductionMode,
    FusedSweepVolume,
    compute_sweep_bounds_mm,
    fuse_sweeps_to_volume,
    volume_geometry_from_bounds_mm,
)
from ultranerf.visualization.trajectory import TrajectoryOverlay, build_trajectory_overlay
from ultranerf.visualization.transforms import ProbeGeometry
from ultranerf.visualization.volume_cache import cache_metadata_matches, load_fused_volume_cache, save_fused_volume_cache
from ultranerf.visualization.volume_viewer import launch_basic_volume_viewer


@dataclass
class VisualizationAppState:
    """Prepared visualization state independent of the GUI."""

    dataset_dir: Path
    fused_volume: FusedSweepVolume
    trajectory: TrajectoryOverlay
    cache_path: Path | None
    cache_used: bool
    images: np.ndarray
    poses_mm: np.ndarray
    probe_geometry: ProbeGeometry
    preset_name: str
    fusion_device: FusionDevice
    reduction_mode: FusionReductionMode


@dataclass
class VisualizationLaunchSession:
    """Live visualization session objects after a GUI launch."""

    viewer: Any
    ui_controller: Any
    render_controller: RenderController | None


@dataclass(frozen=True)
class NerfLaunchConfig:
    """Optional runtime configuration for checkpoint-backed NeRF rendering."""

    checkpoint_path: Path
    config_path: Path
    trigger_mode: RenderTriggerMode = "manual"
    min_render_interval_ms: float = 0.0
    render_image_shape: tuple[int, int] | None = None
    device: str | None = None


def load_visualization_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load visualization inputs directly from disk in millimeters."""
    dataset_path = Path(dataset_dir)
    images = np.load(dataset_path / "images.npy").astype(np.float32)
    poses = np.load(dataset_path / "poses.npy").astype(np.float32)
    return images, poses


def build_or_load_fused_volume(
    *,
    dataset_dir: str | Path,
    probe_geometry: ProbeGeometry,
    spacing_mm: tuple[float, float, float],
    pixel_stride: tuple[int, int],
    cache_path: str | Path | None = None,
    fusion_device: FusionDevice = "auto",
    reduction_mode: FusionReductionMode = "max",
) -> tuple[FusedSweepVolume, Path | None, bool, np.ndarray, np.ndarray]:
    """Load a cached volume when possible, otherwise build and optionally cache it."""
    images, poses_mm = load_visualization_dataset(dataset_dir)
    dataset_path = Path(dataset_dir)
    cache = Path(cache_path) if cache_path is not None else None
    fusion_params = {
        "pixel_stride": list(pixel_stride),
        "spacing_mm": list(spacing_mm),
        "sampling_mode": "nearest",
        "reduction_mode": str(reduction_mode),
    }
    metadata = {
        "dataset_id": str(dataset_path.resolve()),
        "probe_geometry": {"width_mm": probe_geometry.width_mm, "depth_mm": probe_geometry.depth_mm},
        "fusion_params": fusion_params,
    }

    if cache is not None and cache.exists():
        loaded = load_fused_volume_cache(cache)
        if cache_metadata_matches(
            loaded.metadata,
            dataset_id=metadata["dataset_id"],
            probe_geometry=metadata["probe_geometry"],
            fusion_params=metadata["fusion_params"],
        ):
            return loaded.fused_volume, cache, True, images, poses_mm

    bounds_min_mm, bounds_max_mm = compute_sweep_bounds_mm(poses_mm, probe_geometry)
    volume_geometry, volume_shape = volume_geometry_from_bounds_mm(bounds_min_mm, bounds_max_mm, spacing_mm)
    fused = fuse_sweeps_to_volume(
        images=images,
        poses_probe_to_world=poses_mm,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=volume_shape,
        pixel_stride=pixel_stride,
        device=fusion_device,
        reduction_mode=reduction_mode,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_fused_volume_cache(cache, fused, metadata=metadata)
    return fused, cache, False, images, poses_mm


def resolve_render_image_shape(
    images: np.ndarray,
    *,
    render_height: int | None = None,
    render_width: int | None = None,
) -> tuple[int, int]:
    """Resolve the NeRF render image size from overrides or dataset frames."""
    image_array = np.asarray(images)
    if image_array.ndim < 3:
        raise ValueError("images must have shape (N, H, W)")
    height = int(render_height) if render_height is not None else int(image_array.shape[1])
    width = int(render_width) if render_width is not None else int(image_array.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError("render_height and render_width must be positive")
    return (height, width)


def build_render_controller(
    state: VisualizationAppState,
    nerf_config: NerfLaunchConfig,
    *,
    nerf_session_factory: Any | None = None,
) -> RenderController:
    """Build a checkpoint-backed render controller for the visualization app."""
    checkpoint_path = Path(nerf_config.checkpoint_path)
    config_path = Path(nerf_config.config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if nerf_session_factory is None:
        from ultranerf.visualization.nerf_session import NerfSession

        nerf_session_factory = NerfSession.from_checkpoint

    image_shape = nerf_config.render_image_shape or resolve_render_image_shape(state.images)
    nerf_session = nerf_session_factory(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        image_shape=image_shape,
        probe_width_mm=state.probe_geometry.width_mm,
        probe_depth_mm=state.probe_geometry.depth_mm,
        device=nerf_config.device,
    )
    return RenderController(
        nerf_session=nerf_session,
        trigger_mode=nerf_config.trigger_mode,
        min_render_interval_s=float(nerf_config.min_render_interval_ms) / 1000.0,
    )


def prepare_visualization_app(
    *,
    dataset_dir: str | Path,
    probe_width_mm: float,
    probe_depth_mm: float,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pixel_stride: tuple[int, int] = (2, 2),
    cache_path: str | Path | None = None,
    preset_name: str = "soft_tissue",
    fusion_device: FusionDevice = "auto",
    reduction_mode: FusionReductionMode = "max",
) -> VisualizationAppState:
    """Prepare the fused volume and trajectory for the visualization app."""
    probe_geometry = ProbeGeometry(width_mm=float(probe_width_mm), depth_mm=float(probe_depth_mm))
    fused_volume, cache, cache_used, images, poses_mm = build_or_load_fused_volume(
        dataset_dir=dataset_dir,
        probe_geometry=probe_geometry,
        spacing_mm=spacing_mm,
        pixel_stride=pixel_stride,
        cache_path=cache_path,
        fusion_device=fusion_device,
        reduction_mode=reduction_mode,
    )
    trajectory = build_trajectory_overlay(poses_mm)
    return VisualizationAppState(
        dataset_dir=Path(dataset_dir),
        fused_volume=fused_volume,
        trajectory=trajectory,
        cache_path=cache,
        cache_used=cache_used,
        images=images,
        poses_mm=poses_mm,
        probe_geometry=probe_geometry,
        preset_name=preset_name,
        fusion_device=fusion_device,
        reduction_mode=reduction_mode,
    )


def launch_visualization_app(
    state: VisualizationAppState,
    *,
    initial_pose_index: int = 0,
    render_controller: Any | None = None,
    nerf_config: NerfLaunchConfig | None = None,
) -> VisualizationLaunchSession:
    """Launch the napari viewer and attach scene overlays."""
    from ultranerf.visualization.napari_ui import VisualizationUIController

    if render_controller is None and nerf_config is not None:
        render_controller = build_render_controller(state, nerf_config)

    viewer = launch_basic_volume_viewer(
        state.fused_volume,
        viewer_title=f"UltraNeRF Sweep Volume: {state.dataset_dir.name}",
        preset_name=state.preset_name,
    )
    ui_controller = VisualizationUIController(viewer, state, render_controller=render_controller)
    if render_controller is not None and hasattr(viewer, "window"):
        from ultranerf.visualization.comparison_panel import create_comparison_panel
        from ultranerf.visualization.render_panel import create_render_panel

        render_panel = create_render_panel(ui_controller)
        viewer.window.add_dock_widget(render_panel.widget, area="right", name="NeRF Render")
        ui_controller.attach_render_panel(render_panel)
        comparison_panel = create_comparison_panel()
        viewer.window.add_dock_widget(comparison_panel.widget, area="right", name="Nearest Recorded Frame")
        ui_controller.attach_comparison_panel(comparison_panel)
    if hasattr(viewer, "window"):
        from ultranerf.visualization.probe_controls import create_probe_controls

        probe_controls = create_probe_controls(ui_controller, num_frames=state.poses_mm.shape[0])
        viewer.window.add_dock_widget(probe_controls.widget, area="left", name="Probe Controls")
        ui_controller.attach_probe_controls(probe_controls)
    safe_index = min(max(int(initial_pose_index), 0), state.poses_mm.shape[0] - 1)
    ui_controller.initialize(state.poses_mm[safe_index])
    return VisualizationLaunchSession(
        viewer=viewer,
        ui_controller=ui_controller,
        render_controller=render_controller,
    )
