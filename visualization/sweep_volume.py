"""Sweep-to-volume fusion utilities for tracked ultrasound data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from visualization.transforms import (
    ProbeGeometry,
    VolumeGeometry,
    pixel_to_probe_local,
    probe_plane_corners,
    probe_local_to_world,
    world_to_voxel,
)


FusionDevice = str


@dataclass(frozen=True)
class FusedSweepVolume:
    """Fused scalar sweep volume and associated metadata.

    The volume arrays use axis order ``[X, Y, Z]`` in millimeters, matching the
    voxel coordinate convention used by the transform helpers.
    """

    scalar_volume: np.ndarray
    weight_volume: np.ndarray
    origin_mm: np.ndarray
    spacing_mm: np.ndarray
    bounds_min_mm: np.ndarray
    bounds_max_mm: np.ndarray


def compute_sweep_bounds_mm(poses_probe_to_world: np.ndarray, geometry: ProbeGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned world-space bounds covering all scan planes."""
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[0] == 0 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("poses_probe_to_world must have shape (N, 3, 4) or (N, 4, 4)")

    all_corners = [probe_plane_corners(pose, geometry) for pose in poses]
    corners = np.concatenate(all_corners, axis=0)
    return corners.min(axis=0).astype(np.float32), corners.max(axis=0).astype(np.float32)


def volume_geometry_from_bounds_mm(
    bounds_min_mm: np.ndarray,
    bounds_max_mm: np.ndarray,
    spacing_mm: Iterable[float],
) -> Tuple[VolumeGeometry, Tuple[int, int, int]]:
    """Create a volume geometry and array shape from world-space bounds."""
    bounds_min = np.asarray(bounds_min_mm, dtype=np.float32)
    bounds_max = np.asarray(bounds_max_mm, dtype=np.float32)
    spacing = np.asarray(tuple(spacing_mm), dtype=np.float32)
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("bounds must have shape (3,)")
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must have shape (3,)")
    if np.any(spacing <= 0):
        raise ValueError("spacing_mm must be strictly positive")
    if np.any(bounds_max < bounds_min):
        raise ValueError("bounds_max_mm must be greater than or equal to bounds_min_mm")

    extents = bounds_max - bounds_min
    shape = np.floor(extents / spacing).astype(np.int32) + 1
    geometry = VolumeGeometry(origin_mm=bounds_min, spacing_mm=spacing)
    return geometry, tuple(int(v) for v in shape)


def _iter_sampled_pixels(image_shape: Tuple[int, int], pixel_stride: Tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    row_stride, col_stride = pixel_stride
    rows = np.arange(0, image_shape[0], row_stride, dtype=np.float32)
    cols = np.arange(0, image_shape[1], col_stride, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def resolve_fusion_device(device: FusionDevice = "auto") -> str:
    """Resolve the requested fusion device into ``cpu`` or ``cuda``."""
    requested = str(device).strip().lower()
    if requested == "auto":
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("Torch is required for CUDA-backed sweep fusion") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Fusion device was set to CUDA, but torch.cuda.is_available() is False")
        return requested
    if requested == "cpu":
        return "cpu"
    raise ValueError("fusion device must be one of: auto, cpu, cuda, cuda:<index>")


def _finalize_fused_volume(
    *,
    sum_volume: np.ndarray,
    weight_volume: np.ndarray,
    volume_geometry: VolumeGeometry,
    volume_shape: Tuple[int, int, int],
) -> FusedSweepVolume:
    scalar_volume = np.divide(
        sum_volume,
        np.maximum(weight_volume, 1.0),
        out=np.zeros_like(sum_volume),
        where=weight_volume > 0,
    )

    bounds_min = volume_geometry.origin_mm.astype(np.float32)
    bounds_max = (
        volume_geometry.origin_mm + (np.asarray(volume_shape, dtype=np.float32) - 1.0) * volume_geometry.spacing_mm
    ).astype(np.float32)

    return FusedSweepVolume(
        scalar_volume=scalar_volume,
        weight_volume=weight_volume,
        origin_mm=volume_geometry.origin_mm.astype(np.float32),
        spacing_mm=volume_geometry.spacing_mm.astype(np.float32),
        bounds_min_mm=bounds_min,
        bounds_max_mm=bounds_max,
    )


def _fuse_sweeps_to_volume_numpy(
    images: np.ndarray,
    poses_probe_to_world: np.ndarray,
    probe_geometry: ProbeGeometry,
    volume_geometry: VolumeGeometry,
    volume_shape: Tuple[int, int, int],
    pixel_stride: Tuple[int, int],
) -> FusedSweepVolume:
    """Fuse sweeps using the original NumPy implementation."""
    imgs = np.asarray(images, dtype=np.float32)
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)

    sum_volume = np.zeros(volume_shape, dtype=np.float32)
    weight_volume = np.zeros(volume_shape, dtype=np.float32)

    image_shape = (int(imgs.shape[1]), int(imgs.shape[2]))
    rows, cols = _iter_sampled_pixels(image_shape, pixel_stride)
    local_points = pixel_to_probe_local(rows, cols, image_shape=image_shape, geometry=probe_geometry)

    for image, pose in zip(imgs, poses):
        world_points = probe_local_to_world(local_points, pose)
        voxel_points = world_to_voxel(world_points, volume_geometry)
        voxel_indices = np.rint(voxel_points).astype(np.int32)

        flat_values = image[rows.astype(np.int32), cols.astype(np.int32)]
        finite_mask = np.isfinite(flat_values)

        valid_mask = np.logical_and.reduce(
            [
                voxel_indices[:, 0] >= 0,
                voxel_indices[:, 0] < volume_shape[0],
                voxel_indices[:, 1] >= 0,
                voxel_indices[:, 1] < volume_shape[1],
                voxel_indices[:, 2] >= 0,
                voxel_indices[:, 2] < volume_shape[2],
                finite_mask,
            ]
        )

        valid_indices = voxel_indices[valid_mask]
        valid_values = flat_values[valid_mask]

        for idx, value in zip(valid_indices, valid_values):
            sum_volume[idx[0], idx[1], idx[2]] += float(value)
            weight_volume[idx[0], idx[1], idx[2]] += 1.0

    return _finalize_fused_volume(
        sum_volume=sum_volume,
        weight_volume=weight_volume,
        volume_geometry=volume_geometry,
        volume_shape=volume_shape,
    )


def _fuse_sweeps_to_volume_torch(
    images: np.ndarray,
    poses_probe_to_world: np.ndarray,
    probe_geometry: ProbeGeometry,
    volume_geometry: VolumeGeometry,
    volume_shape: Tuple[int, int, int],
    pixel_stride: Tuple[int, int],
    device: str,
) -> FusedSweepVolume:
    """Fuse sweeps using torch scatter-add on CPU or CUDA."""
    import torch

    torch_device = torch.device(device)
    imgs = np.asarray(images, dtype=np.float32)
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)
    image_shape = (int(imgs.shape[1]), int(imgs.shape[2]))
    rows_np, cols_np = _iter_sampled_pixels(image_shape, pixel_stride)
    local_points_np = pixel_to_probe_local(rows_np, cols_np, image_shape=image_shape, geometry=probe_geometry)

    rows = torch.from_numpy(rows_np.astype(np.int64)).to(torch_device)
    cols = torch.from_numpy(cols_np.astype(np.int64)).to(torch_device)
    local_points = torch.from_numpy(local_points_np).to(torch_device)
    ones = torch.ones((local_points.shape[0], 1), dtype=torch.float32, device=torch_device)
    local_points_h = torch.cat([local_points, ones], dim=1)
    origin = torch.from_numpy(volume_geometry.origin_mm.astype(np.float32)).to(torch_device)
    spacing = torch.from_numpy(volume_geometry.spacing_mm.astype(np.float32)).to(torch_device)

    x_dim, y_dim, z_dim = (int(v) for v in volume_shape)
    flat_size = x_dim * y_dim * z_dim
    sum_volume_flat = torch.zeros(flat_size, dtype=torch.float32, device=torch_device)
    weight_volume_flat = torch.zeros(flat_size, dtype=torch.float32, device=torch_device)

    for image_np, pose_np in zip(imgs, poses):
        image = torch.from_numpy(image_np).to(torch_device)
        pose = torch.from_numpy(pose_np.astype(np.float32)).to(torch_device)
        world_points = (pose @ local_points_h.T).T[:, :3]
        voxel_points = (world_points - origin) / spacing
        voxel_indices = torch.round(voxel_points).to(torch.int64)
        flat_values = image[rows, cols]
        finite_mask = torch.isfinite(flat_values)
        valid_mask = (
            (voxel_indices[:, 0] >= 0)
            & (voxel_indices[:, 0] < x_dim)
            & (voxel_indices[:, 1] >= 0)
            & (voxel_indices[:, 1] < y_dim)
            & (voxel_indices[:, 2] >= 0)
            & (voxel_indices[:, 2] < z_dim)
            & finite_mask
        )
        if not torch.any(valid_mask):
            continue
        valid_indices = voxel_indices[valid_mask]
        valid_values = flat_values[valid_mask].to(torch.float32)
        linear_indices = valid_indices[:, 0] * (y_dim * z_dim) + valid_indices[:, 1] * z_dim + valid_indices[:, 2]
        sum_volume_flat.index_add_(0, linear_indices, valid_values)
        weight_volume_flat.index_add_(0, linear_indices, torch.ones_like(valid_values))

    sum_volume = sum_volume_flat.reshape(volume_shape).detach().cpu().numpy().astype(np.float32)
    weight_volume = weight_volume_flat.reshape(volume_shape).detach().cpu().numpy().astype(np.float32)
    return _finalize_fused_volume(
        sum_volume=sum_volume,
        weight_volume=weight_volume,
        volume_geometry=volume_geometry,
        volume_shape=volume_shape,
    )


def fuse_sweeps_to_volume(
    images: np.ndarray,
    poses_probe_to_world: np.ndarray,
    probe_geometry: ProbeGeometry,
    volume_geometry: VolumeGeometry,
    volume_shape: Tuple[int, int, int],
    pixel_stride: Tuple[int, int] = (1, 1),
    device: FusionDevice = "auto",
) -> FusedSweepVolume:
    """Fuse tracked 2D sweeps into a dense scalar volume.

    Nearest-neighbor splatting is used in the first implementation. Intensities
    are accumulated into `weight_volume`, and the returned `scalar_volume` is the
    normalized mean intensity per voxel.
    """
    imgs = np.asarray(images, dtype=np.float32)
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)

    if imgs.ndim != 3:
        raise ValueError("images must have shape (N, H, W)")
    if poses.ndim != 3 or poses.shape[0] != imgs.shape[0]:
        raise ValueError("poses_probe_to_world must match image batch size")
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must have length 3")
    resolved_device = resolve_fusion_device(device)
    if resolved_device == "cpu":
        return _fuse_sweeps_to_volume_numpy(
            images=imgs,
            poses_probe_to_world=poses,
            probe_geometry=probe_geometry,
            volume_geometry=volume_geometry,
            volume_shape=volume_shape,
            pixel_stride=pixel_stride,
        )
    return _fuse_sweeps_to_volume_torch(
        images=imgs,
        poses_probe_to_world=poses,
        probe_geometry=probe_geometry,
        volume_geometry=volume_geometry,
        volume_shape=volume_shape,
        pixel_stride=pixel_stride,
        device=resolved_device,
    )
