"""Data structures for multi-sweep visualization scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visualization.transforms import ProbeGeometry, ensure_pose_matrix


AlignmentSource = str
ComparisonPolicy = str


def _normalize_color_rgb(color_rgb: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if color_rgb is None:
        return None
    color = tuple(float(value) for value in color_rgb)
    if len(color) != 3:
        raise ValueError("color_rgb must have exactly three components")
    if any(value < 0.0 or value > 1.0 for value in color):
        raise ValueError("color_rgb values must be in the range [0, 1]")
    return color


def validate_sweep_images_and_poses(images: np.ndarray, poses_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize one sweep worth of tracked images and poses."""
    image_array = np.asarray(images, dtype=np.float32)
    pose_array = np.asarray(poses_mm, dtype=np.float32)

    if image_array.ndim != 3:
        raise ValueError("images must have shape (N, H, W)")
    if image_array.shape[0] == 0:
        raise ValueError("images must contain at least one frame")
    if pose_array.ndim != 3:
        raise ValueError("poses_mm must have shape (N, 3, 4) or (N, 4, 4)")
    if pose_array.shape[0] != image_array.shape[0]:
        raise ValueError("images and poses_mm must contain the same number of frames")

    normalized_poses = np.stack([ensure_pose_matrix(pose) for pose in pose_array], axis=0).astype(np.float32)
    return image_array, normalized_poses


@dataclass(frozen=True)
class SweepRecord:
    """One tracked ultrasound sweep that participates in a shared scene."""

    sweep_id: str
    images: np.ndarray
    poses_mm: np.ndarray
    probe_geometry: ProbeGeometry
    dataset_dir: Path | None = None
    display_name: str | None = None
    color_rgb: tuple[float, float, float] | None = None
    enabled: bool = True
    alignment_source: AlignmentSource = "assumed_from_training"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sweep_id:
            raise ValueError("sweep_id must be non-empty")

        images, poses = validate_sweep_images_and_poses(self.images, self.poses_mm)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "poses_mm", poses)
        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir) if self.dataset_dir is not None else None)
        object.__setattr__(self, "display_name", self.display_name or self.sweep_id)
        object.__setattr__(self, "color_rgb", _normalize_color_rgb(self.color_rgb))
        object.__setattr__(self, "alignment_source", str(self.alignment_source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def frame_count(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_shape(self) -> tuple[int, int]:
        return (int(self.images.shape[1]), int(self.images.shape[2]))


@dataclass(frozen=True)
class MultiSweepScene:
    """A collection of sweeps that share one viewer world frame."""

    sweeps: tuple[SweepRecord, ...]
    active_sweep_id: str | None = None
    comparison_policy: ComparisonPolicy = "all_enabled"
    world_unit: str = "mm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sweeps:
            raise ValueError("MultiSweepScene requires at least one sweep")

        normalized_sweeps = tuple(self.sweeps)
        sweep_ids = [sweep.sweep_id for sweep in normalized_sweeps]
        if len(set(sweep_ids)) != len(sweep_ids):
            raise ValueError("Sweep ids must be unique within a scene")
        if self.world_unit != "mm":
            raise ValueError("world_unit must be 'mm' for visualization scenes")

        active_id = self.active_sweep_id or normalized_sweeps[0].sweep_id
        if active_id not in set(sweep_ids):
            raise ValueError("active_sweep_id must refer to one of the scene sweeps")

        object.__setattr__(self, "sweeps", normalized_sweeps)
        object.__setattr__(self, "active_sweep_id", active_id)
        object.__setattr__(self, "comparison_policy", str(self.comparison_policy))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def sweep_ids(self) -> tuple[str, ...]:
        return tuple(sweep.sweep_id for sweep in self.sweeps)

    @property
    def active_sweep(self) -> SweepRecord:
        return self.get_sweep(self.active_sweep_id)

    @property
    def enabled_sweeps(self) -> tuple[SweepRecord, ...]:
        return tuple(sweep for sweep in self.sweeps if sweep.enabled)

    def get_sweep(self, sweep_id: str) -> SweepRecord:
        for sweep in self.sweeps:
            if sweep.sweep_id == sweep_id:
                return sweep
        raise KeyError(f"Unknown sweep_id: {sweep_id}")

    def with_active_sweep(self, sweep_id: str) -> "MultiSweepScene":
        return MultiSweepScene(
            sweeps=self.sweeps,
            active_sweep_id=sweep_id,
            comparison_policy=self.comparison_policy,
            world_unit=self.world_unit,
            metadata=self.metadata,
        )
