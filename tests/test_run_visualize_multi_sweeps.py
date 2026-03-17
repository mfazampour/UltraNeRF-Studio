import json
import subprocess
from pathlib import Path

import numpy as np


def write_sweep_dir(path: Path, offset_mm: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = np.ones((2, 3, 4), dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = offset_mm
    np.save(path / "images.npy", images)
    np.save(path / "poses.npy", poses)


def write_manifest(tmp_path: Path) -> Path:
    write_sweep_dir(tmp_path / "sweep_a", 0.0)
    write_sweep_dir(tmp_path / "sweep_b", 30.0)
    manifest = {
        "probe_geometry": {"width_mm": 20.0, "depth_mm": 20.0},
        "active_sweep_id": "sweep_a",
        "comparison_policy": "all_enabled",
        "sweeps": [
            {"sweep_id": "sweep_a", "dataset_dir": "sweep_a"},
            {"sweep_id": "sweep_b", "dataset_dir": "sweep_b"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_run_visualize_multi_sweeps_no_gui_prints_summary(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)

    result = subprocess.run(
        [
            "/workspace/miniconda3/bin/conda",
            "run",
            "-n",
            "ultranerf",
            "python",
            "/workspace/run_visualize_multi_sweeps.py",
            "--manifest-path",
            str(manifest_path),
            "--spacing-mm",
            "5",
            "5",
            "5",
            "--pixel-stride",
            "2",
            "2",
            "--no-gui",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["manifest_path"] == str(manifest_path.resolve())
    assert summary["num_sweeps"] == 2
    assert summary["sweep_ids"] == ["sweep_a", "sweep_b"]
    assert summary["active_sweep_id"] == "sweep_a"
    assert summary["nerf_enabled"] is False
    assert "alignment_warning_count" in summary


def test_run_visualize_multi_sweeps_no_gui_reports_nerf_config(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    checkpoint_path = tmp_path / "model.tar"
    config_path = tmp_path / "config.txt"
    checkpoint_path.write_text("checkpoint")
    config_path.write_text("config")

    result = subprocess.run(
        [
            "/workspace/miniconda3/bin/conda",
            "run",
            "-n",
            "ultranerf",
            "python",
            "/workspace/run_visualize_multi_sweeps.py",
            "--manifest-path",
            str(manifest_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--config-path",
            str(config_path),
            "--render-height",
            "8",
            "--render-width",
            "9",
            "--no-gui",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["nerf_enabled"] is True
    assert summary["checkpoint_path"] == str(checkpoint_path.resolve())
    assert summary["config_path"] == str(config_path.resolve())
    assert summary["render_image_shape"] == [8, 9]
