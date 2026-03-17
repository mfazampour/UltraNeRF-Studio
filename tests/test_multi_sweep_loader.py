import json
from pathlib import Path

import numpy as np
import pytest

from visualization.multi_sweep_loader import (
    discover_sweep_directories,
    load_multi_sweep_scene_from_directory,
    load_multi_sweep_scene_from_manifest,
    load_sweep_record,
)
from visualization.transforms import ProbeGeometry


def make_images(num_frames: int = 2, height: int = 3, width: int = 4) -> np.ndarray:
    return np.arange(num_frames * height * width, dtype=np.float32).reshape(num_frames, height, width)


def make_poses(num_frames: int = 2) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_frames, axis=0)
    for idx in range(num_frames):
        poses[idx, 0, 3] = float(idx * 10)
    return poses


def write_sweep_dir(path: Path, *, images: np.ndarray | None = None, poses: np.ndarray | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "images.npy", make_images() if images is None else images)
    np.save(path / "poses.npy", make_poses() if poses is None else poses)


def test_load_sweep_record_reads_default_image_and_pose_files(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep_a"
    write_sweep_dir(sweep_dir)

    record = load_sweep_record(
        dataset_dir=sweep_dir,
        sweep_id="sweep_a",
        probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
    )

    assert record.sweep_id == "sweep_a"
    assert record.dataset_dir == sweep_dir
    assert record.images.shape == (2, 3, 4)
    assert record.poses_mm.shape == (2, 4, 4)


def test_load_multi_sweep_scene_from_manifest_supports_inline_metadata_and_transform(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep_a"
    write_sweep_dir(sweep_dir)
    manifest = {
        "probe_geometry": {"width_mm": 80.0, "depth_mm": 139.0},
        "active_sweep_id": "sweep_a",
        "sweeps": [
            {
                "sweep_id": "sweep_a",
                "dataset_dir": "sweep_a",
                "display_name": "Anterior Sweep",
                "color_rgb": [0.2, 0.4, 0.6],
                "world_transform_mm": np.eye(4, dtype=np.float32).tolist(),
                "metadata": {"angle_deg": 30},
            }
        ],
    }
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(json.dumps(manifest))

    scene = load_multi_sweep_scene_from_manifest(manifest_path)

    assert scene.active_sweep.sweep_id == "sweep_a"
    assert scene.active_sweep.display_name == "Anterior Sweep"
    assert scene.active_sweep.color_rgb == (0.2, 0.4, 0.6)
    assert scene.active_sweep.metadata["angle_deg"] == 30
    assert np.allclose(scene.active_sweep.metadata["world_transform_mm"], np.eye(4, dtype=np.float32))
    assert np.allclose(scene.active_sweep.world_transform_mm, np.eye(4, dtype=np.float32))


def test_load_multi_sweep_scene_from_directory_discovers_subdirectories(tmp_path: Path) -> None:
    write_sweep_dir(tmp_path / "sweep_a")
    write_sweep_dir(tmp_path / "sweep_b")
    (tmp_path / "ignore.txt").write_text("ignore")

    scene = load_multi_sweep_scene_from_directory(
        tmp_path,
        probe_geometry=ProbeGeometry(width_mm=80.0, depth_mm=139.0),
    )

    assert scene.sweep_ids == ("sweep_a", "sweep_b")
    assert scene.metadata["source_root"] == str(tmp_path.resolve())


def test_discover_sweep_directories_ignores_incomplete_entries(tmp_path: Path) -> None:
    write_sweep_dir(tmp_path / "complete")
    (tmp_path / "incomplete").mkdir()
    np.save(tmp_path / "incomplete" / "images.npy", make_images())

    sweep_dirs = discover_sweep_directories(tmp_path)

    assert sweep_dirs == (tmp_path / "complete",)


def test_manifest_requires_probe_geometry_when_not_passed_explicitly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(json.dumps({"sweeps": []}))

    with pytest.raises(ValueError, match="probe geometry"):
        load_multi_sweep_scene_from_manifest(manifest_path)
