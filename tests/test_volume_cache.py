import numpy as np

from visualization.sweep_volume import FusedSweepVolume
from visualization.volume_cache import cache_metadata_matches, load_fused_volume_cache, save_fused_volume_cache


def make_fused_volume() -> FusedSweepVolume:
    scalar = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    weights = np.ones((2, 2, 2), dtype=np.float32)
    return FusedSweepVolume(
        scalar_volume=scalar,
        weight_volume=weights,
        origin_mm=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        spacing_mm=np.array([0.5, 0.5, 1.0], dtype=np.float32),
        bounds_min_mm=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bounds_max_mm=np.array([1.5, 2.5, 4.0], dtype=np.float32),
    )


def test_volume_cache_round_trip_preserves_arrays_and_metadata(tmp_path):
    fused = make_fused_volume()
    cache_path = tmp_path / "fused_volume.npz"
    metadata = {
        "dataset_id": "synthetic-r2",
        "probe_geometry": {"width_mm": 80.0, "depth_mm": 140.0},
        "fusion_params": {"pixel_stride": [2, 2]},
    }

    save_fused_volume_cache(cache_path, fused, metadata=metadata)
    loaded = load_fused_volume_cache(cache_path)

    assert np.allclose(loaded.fused_volume.scalar_volume, fused.scalar_volume)
    assert np.allclose(loaded.fused_volume.weight_volume, fused.weight_volume)
    assert np.allclose(loaded.fused_volume.origin_mm, fused.origin_mm)
    assert np.allclose(loaded.fused_volume.spacing_mm, fused.spacing_mm)
    assert loaded.metadata["dataset_id"] == "synthetic-r2"
    assert loaded.metadata["probe_geometry"] == {"width_mm": 80.0, "depth_mm": 140.0}
    assert loaded.metadata["fusion_params"] == {"pixel_stride": [2, 2]}


def test_cache_metadata_matches_detects_dataset_probe_and_params():
    metadata = {
        "cache_version": 1,
        "dataset_id": "synthetic-r2",
        "probe_geometry": {"width_mm": 80.0, "depth_mm": 140.0},
        "fusion_params": {"pixel_stride": [1, 1], "sampling_mode": "nearest", "reduction_mode": "max"},
    }

    assert cache_metadata_matches(
        metadata,
        dataset_id="synthetic-r2",
        probe_geometry={"width_mm": 80.0, "depth_mm": 140.0},
        fusion_params={"pixel_stride": [1, 1], "sampling_mode": "nearest", "reduction_mode": "max"},
    )
    assert not cache_metadata_matches(metadata, dataset_id="synthetic-l2")
    assert not cache_metadata_matches(metadata, probe_geometry={"width_mm": 37.0, "depth_mm": 100.0})
    assert not cache_metadata_matches(
        metadata,
        fusion_params={"pixel_stride": [2, 2], "sampling_mode": "nearest", "reduction_mode": "max"},
    )
