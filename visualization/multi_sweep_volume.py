"""Multi-sweep fusion and overlay helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.sweep_volume import (
    FusionDevice,
    FusionReductionMode,
    FusedSweepVolume,
    compute_sweep_bounds_mm,
    fuse_sweeps_to_volume,
    volume_geometry_from_bounds_mm,
)
from visualization.trajectory import TrajectoryOverlay, build_trajectory_overlay


@dataclass(frozen=True)
class SweepVolumeOverlay:
    """Per-sweep fused volume and trajectory overlay payload."""

    sweep_id: str
    display_name: str
    color_rgb: tuple[float, float, float] | None
    enabled: bool
    fused_volume: FusedSweepVolume | None
    trajectory: TrajectoryOverlay


@dataclass(frozen=True)
class MultiSweepFusionResult:
    """Aggregate fusion outputs for a multi-sweep scene."""

    aggregate_volume: FusedSweepVolume
    sweep_overlays: tuple[SweepVolumeOverlay, ...]
    enabled_sweep_ids: tuple[str, ...]
    bounds_min_mm: np.ndarray
    bounds_max_mm: np.ndarray


def compute_scene_bounds_mm(
    sweeps: Iterable[SweepRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute union world-space bounds over several sweeps."""
    bounds_min = []
    bounds_max = []
    for sweep in sweeps:
        sweep_min, sweep_max = compute_sweep_bounds_mm(sweep.poses_mm, sweep.probe_geometry)
        bounds_min.append(sweep_min)
        bounds_max.append(sweep_max)
    if not bounds_min:
        raise ValueError("At least one sweep is required to compute scene bounds")
    return (
        np.min(np.stack(bounds_min, axis=0), axis=0).astype(np.float32),
        np.max(np.stack(bounds_max, axis=0), axis=0).astype(np.float32),
    )


def build_sweep_overlay(
    sweep: SweepRecord,
    *,
    spacing_mm: tuple[float, float, float],
    pixel_stride: tuple[int, int] = (1, 1),
    axis_stride: int = 10,
    fusion_device: FusionDevice = "auto",
    reduction_mode: FusionReductionMode = "max",
    include_volume: bool = True,
) -> SweepVolumeOverlay:
    """Build one per-sweep fused volume plus trajectory overlay."""
    fused_volume = None
    if include_volume:
        bounds_min_mm, bounds_max_mm = compute_sweep_bounds_mm(sweep.poses_mm, sweep.probe_geometry)
        volume_geometry, volume_shape = volume_geometry_from_bounds_mm(bounds_min_mm, bounds_max_mm, spacing_mm)
        fused_volume = fuse_sweeps_to_volume(
            images=sweep.images,
            poses_probe_to_world=sweep.poses_mm,
            probe_geometry=sweep.probe_geometry,
            volume_geometry=volume_geometry,
            volume_shape=volume_shape,
            pixel_stride=pixel_stride,
            device=fusion_device,
            reduction_mode=reduction_mode,
        )
    trajectory = build_trajectory_overlay(sweep.poses_mm, axis_stride=axis_stride)
    return SweepVolumeOverlay(
        sweep_id=sweep.sweep_id,
        display_name=sweep.display_name or sweep.sweep_id,
        color_rgb=sweep.color_rgb,
        enabled=sweep.enabled,
        fused_volume=fused_volume,
        trajectory=trajectory,
    )


def fuse_multi_sweep_scene(
    scene: MultiSweepScene,
    *,
    spacing_mm: tuple[float, float, float],
    pixel_stride: tuple[int, int] = (1, 1),
    enabled_sweep_ids: Iterable[str] | None = None,
    axis_stride: int = 10,
    fusion_device: FusionDevice = "auto",
    reduction_mode: FusionReductionMode = "max",
    include_per_sweep_volumes: bool = True,
) -> MultiSweepFusionResult:
    """Fuse several sweeps into one aggregate volume plus per-sweep overlays."""
    if enabled_sweep_ids is None:
        selected_sweeps = scene.enabled_sweeps or scene.sweeps
    else:
        selected_id_set = {str(sweep_id) for sweep_id in enabled_sweep_ids}
        selected_sweeps = tuple(sweep for sweep in scene.sweeps if sweep.sweep_id in selected_id_set)

    if not selected_sweeps:
        raise ValueError("At least one enabled sweep is required for multi-sweep fusion")

    bounds_min_mm, bounds_max_mm = compute_scene_bounds_mm(selected_sweeps)
    volume_geometry, volume_shape = volume_geometry_from_bounds_mm(bounds_min_mm, bounds_max_mm, spacing_mm)
    overlays = []
    aggregate_images = []
    aggregate_poses = []

    for sweep in selected_sweeps:
        overlays.append(
            build_sweep_overlay(
                sweep,
                spacing_mm=spacing_mm,
                pixel_stride=pixel_stride,
                axis_stride=axis_stride,
                fusion_device=fusion_device,
                reduction_mode=reduction_mode,
                include_volume=include_per_sweep_volumes,
            )
        )
        aggregate_images.append(sweep.images)
        aggregate_poses.append(sweep.poses_mm)

    aggregate_volume = fuse_sweeps_to_volume(
        images=np.concatenate(aggregate_images, axis=0),
        poses_probe_to_world=np.concatenate(aggregate_poses, axis=0),
        probe_geometry=selected_sweeps[0].probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=volume_shape,
        pixel_stride=pixel_stride,
        device=fusion_device,
        reduction_mode=reduction_mode,
    )
    return MultiSweepFusionResult(
        aggregate_volume=aggregate_volume,
        sweep_overlays=tuple(overlays),
        enabled_sweep_ids=tuple(sweep.sweep_id for sweep in selected_sweeps),
        bounds_min_mm=bounds_min_mm,
        bounds_max_mm=bounds_max_mm,
    )
