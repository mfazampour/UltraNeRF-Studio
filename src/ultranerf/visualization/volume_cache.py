"""Cache helpers for fused sweep volumes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ultranerf.visualization.sweep_volume import FusedSweepVolume


CACHE_VERSION = 1


@dataclass(frozen=True)
class CachedVolume:
    """Loaded cached volume bundle."""

    fused_volume: FusedSweepVolume
    metadata: dict[str, Any]


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    return value


def save_fused_volume_cache(
    path: str | Path,
    fused_volume: FusedSweepVolume,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a fused volume cache to an ``.npz`` file."""
    output_path = Path(path)
    user_metadata = dict(metadata or {})
    full_metadata = {
        "cache_version": CACHE_VERSION,
        **{k: _normalize_jsonable(v) for k, v in user_metadata.items()},
    }

    np.savez_compressed(
        output_path,
        scalar_volume=fused_volume.scalar_volume,
        weight_volume=fused_volume.weight_volume,
        origin_mm=fused_volume.origin_mm,
        spacing_mm=fused_volume.spacing_mm,
        bounds_min_mm=fused_volume.bounds_min_mm,
        bounds_max_mm=fused_volume.bounds_max_mm,
        metadata_json=np.array(json.dumps(full_metadata), dtype=np.str_),
    )
    return output_path


def load_fused_volume_cache(path: str | Path) -> CachedVolume:
    """Load a fused volume cache from disk."""
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        fused = FusedSweepVolume(
            scalar_volume=data["scalar_volume"].astype(np.float32),
            weight_volume=data["weight_volume"].astype(np.float32),
            origin_mm=data["origin_mm"].astype(np.float32),
            spacing_mm=data["spacing_mm"].astype(np.float32),
            bounds_min_mm=data["bounds_min_mm"].astype(np.float32),
            bounds_max_mm=data["bounds_max_mm"].astype(np.float32),
        )
    return CachedVolume(fused_volume=fused, metadata=metadata)


def cache_metadata_matches(
    metadata: Mapping[str, Any],
    *,
    dataset_id: str | None = None,
    probe_geometry: Mapping[str, Any] | None = None,
    fusion_params: Mapping[str, Any] | None = None,
) -> bool:
    """Check whether cached metadata matches the requested visualization setup."""
    if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
        return False
    if dataset_id is not None and metadata.get("dataset_id") != dataset_id:
        return False
    if probe_geometry is not None and metadata.get("probe_geometry") != _normalize_jsonable(dict(probe_geometry)):
        return False
    if fusion_params is not None and metadata.get("fusion_params") != _normalize_jsonable(dict(fusion_params)):
        return False
    return True
