import json
from pathlib import Path

import numpy as np

from visualization.app import build_or_load_fused_volume, prepare_visualization_app


def make_dataset(tmp_path: Path):
    images = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    poses[1, 0, 3] = 10.0
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "poses.npy", poses)
    return images, poses


def test_build_or_load_fused_volume_writes_and_reuses_cache(tmp_path):
    make_dataset(tmp_path)
    cache_path = tmp_path / "cache" / "volume.npz"

    fused_a, cache_a, cache_used_a, _, _ = build_or_load_fused_volume(
        dataset_dir=tmp_path,
        probe_geometry=type("Probe", (), {"width_mm": 4.0, "depth_mm": 4.0})(),
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=cache_path,
    )
    fused_b, cache_b, cache_used_b, _, _ = build_or_load_fused_volume(
        dataset_dir=tmp_path,
        probe_geometry=type("Probe", (), {"width_mm": 4.0, "depth_mm": 4.0})(),
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=cache_path,
    )

    assert cache_a == cache_path
    assert cache_b == cache_path
    assert not cache_used_a
    assert cache_used_b
    assert np.allclose(fused_a.scalar_volume, fused_b.scalar_volume)


def test_prepare_visualization_app_returns_volume_and_trajectory(tmp_path):
    images, poses = make_dataset(tmp_path)

    state = prepare_visualization_app(
        dataset_dir=tmp_path,
        probe_width_mm=4.0,
        probe_depth_mm=4.0,
        spacing_mm=(2.0, 2.0, 1.0),
        pixel_stride=(1, 1),
        cache_path=None,
        preset_name="soft_tissue",
    )

    assert state.images.shape == images.shape
    assert state.poses_mm.shape == poses.shape
    assert state.fused_volume.scalar_volume.ndim == 3
    assert state.trajectory.centers_mm.shape[0] == poses.shape[0]
    assert state.preset_name == "soft_tissue"
