"""Loading helpers for multi-sweep visualization scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from visualization.multi_sweep import MultiSweepScene, SweepRecord
from visualization.transforms import ProbeGeometry, ensure_pose_matrix


def _load_npy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


def _parse_probe_geometry(manifest: dict[str, Any], probe_geometry: ProbeGeometry | None) -> ProbeGeometry:
    if probe_geometry is not None:
        return probe_geometry
    if "probe_geometry" not in manifest:
        raise ValueError("probe geometry must be provided either as an argument or in the manifest")
    geometry_data = manifest["probe_geometry"]
    return ProbeGeometry(
        width_mm=float(geometry_data["width_mm"]),
        depth_mm=float(geometry_data["depth_mm"]),
    )


def _resolve_transform_matrix(entry: dict[str, Any], base_dir: Path) -> np.ndarray | None:
    if "world_transform_mm" in entry:
        return ensure_pose_matrix(np.asarray(entry["world_transform_mm"], dtype=np.float32))
    transform_path = entry.get("world_transform_path")
    if transform_path is None:
        return None
    return ensure_pose_matrix(_load_npy_array((base_dir / transform_path).resolve()))


def load_sweep_record(
    *,
    dataset_dir: str | Path,
    sweep_id: str,
    probe_geometry: ProbeGeometry,
    display_name: str | None = None,
    image_path: str | Path | None = None,
    pose_path: str | Path | None = None,
    color_rgb: tuple[float, float, float] | None = None,
    enabled: bool = True,
    alignment_source: str = "assumed_from_training",
    world_transform_mm: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> SweepRecord:
    """Load one sweep from disk into a structured record."""
    dataset_path = Path(dataset_dir)
    images_path = Path(image_path) if image_path is not None else dataset_path / "images.npy"
    poses_path = Path(pose_path) if pose_path is not None else dataset_path / "poses.npy"
    images = _load_npy_array(images_path).astype(np.float32)
    poses_mm = _load_npy_array(poses_path).astype(np.float32)
    return SweepRecord(
        sweep_id=sweep_id,
        display_name=display_name,
        dataset_dir=dataset_path,
        images=images,
        poses_mm=poses_mm,
        probe_geometry=probe_geometry,
        world_transform_mm=world_transform_mm if world_transform_mm is not None else np.eye(4, dtype=np.float32),
        color_rgb=color_rgb,
        enabled=enabled,
        alignment_source=alignment_source,
        metadata=dict(metadata or {}),
    )


def load_multi_sweep_scene_from_manifest(
    manifest_path: str | Path,
    *,
    probe_geometry: ProbeGeometry | None = None,
) -> MultiSweepScene:
    """Load a multi-sweep scene from a JSON manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    geometry = _parse_probe_geometry(manifest, probe_geometry)
    sweeps: list[SweepRecord] = []

    for index, entry in enumerate(manifest.get("sweeps", [])):
        dataset_dir = (path.parent / entry["dataset_dir"]).resolve()
        entry_metadata = dict(entry.get("metadata", {}))
        resolved_transform = _resolve_transform_matrix(entry, path.parent)
        if resolved_transform is not None:
            entry_metadata["world_transform_mm"] = resolved_transform.astype(np.float32)
        sweeps.append(
            load_sweep_record(
                dataset_dir=dataset_dir,
                sweep_id=str(entry.get("sweep_id", f"sweep_{index:02d}")),
                display_name=entry.get("display_name"),
                image_path=(path.parent / entry["image_path"]).resolve() if entry.get("image_path") else None,
                pose_path=(path.parent / entry["pose_path"]).resolve() if entry.get("pose_path") else None,
                probe_geometry=geometry,
                color_rgb=tuple(entry["color_rgb"]) if entry.get("color_rgb") is not None else None,
                enabled=bool(entry.get("enabled", True)),
                alignment_source=str(entry.get("alignment_source", "assumed_from_training")),
                world_transform_mm=resolved_transform,
                metadata=entry_metadata,
            )
        )

    if not sweeps:
        raise ValueError("manifest must define at least one sweep")

    return MultiSweepScene(
        sweeps=tuple(sweeps),
        active_sweep_id=manifest.get("active_sweep_id"),
        comparison_policy=str(manifest.get("comparison_policy", "all_enabled")),
        metadata=dict(manifest.get("metadata", {})),
    )


def discover_sweep_directories(root_dir: str | Path) -> tuple[Path, ...]:
    """Discover sweep directories under a root directory."""
    root = Path(root_dir)
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images.npy").exists() and (child / "poses.npy").exists():
            candidates.append(child)
    return tuple(candidates)


def load_multi_sweep_scene_from_directory(
    root_dir: str | Path,
    *,
    probe_geometry: ProbeGeometry,
    comparison_policy: str = "all_enabled",
) -> MultiSweepScene:
    """Load a scene from a directory containing one subdirectory per sweep."""
    root = Path(root_dir)
    sweep_dirs = discover_sweep_directories(root)
    if not sweep_dirs:
        raise ValueError(f"No sweep directories found under {root}")

    sweeps = tuple(
        load_sweep_record(
            dataset_dir=sweep_dir,
            sweep_id=sweep_dir.name,
            display_name=sweep_dir.name,
            probe_geometry=probe_geometry,
        )
        for sweep_dir in sweep_dirs
    )
    return MultiSweepScene(
        sweeps=sweeps,
        active_sweep_id=sweeps[0].sweep_id,
        comparison_policy=comparison_policy,
        metadata={"source_root": str(root.resolve())},
    )
