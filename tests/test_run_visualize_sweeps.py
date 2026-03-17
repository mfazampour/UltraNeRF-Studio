import json
import subprocess
from pathlib import Path

import numpy as np


def make_dataset(tmp_path: Path):
    images = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "poses.npy", poses)


def test_run_visualize_sweeps_no_gui_prepares_volume_and_prints_summary(tmp_path):
    make_dataset(tmp_path)
    cache_path = tmp_path / "cached_volume.npz"

    result = subprocess.run(
        [
            "/workspace/miniconda3/bin/conda",
            "run",
            "-n",
            "ultranerf",
            "python",
            "/workspace/run_visualize_sweeps.py",
            "--dataset-dir",
            str(tmp_path),
            "--probe-width-mm",
            "4",
            "--probe-depth-mm",
            "4",
            "--spacing-mm",
            "2",
            "2",
            "1",
            "--pixel-stride",
            "1",
            "1",
            "--cache-path",
            str(cache_path),
            "--preset",
            "soft_tissue",
            "--no-gui",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["dataset_dir"] == str(tmp_path.resolve())
    assert summary["cache_path"] == str(cache_path)
    assert summary["cache_used"] is False
    assert summary["preset"] == "soft_tissue"
    assert summary["num_frames"] == 2
    assert summary["nerf_enabled"] is False
    assert summary["render_image_shape"] is None
    assert cache_path.exists()


def test_run_visualize_sweeps_no_gui_reports_nerf_launch_configuration(tmp_path):
    make_dataset(tmp_path)
    cache_path = tmp_path / "cached_volume.npz"
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
            "/workspace/run_visualize_sweeps.py",
            "--dataset-dir",
            str(tmp_path),
            "--probe-width-mm",
            "4",
            "--probe-depth-mm",
            "4",
            "--spacing-mm",
            "2",
            "2",
            "1",
            "--pixel-stride",
            "1",
            "1",
            "--cache-path",
            str(cache_path),
            "--preset",
            "soft_tissue",
            "--checkpoint-path",
            str(checkpoint_path),
            "--config-path",
            str(config_path),
            "--render-trigger-mode",
            "manual",
            "--render-height",
            "6",
            "--render-width",
            "7",
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
    assert summary["render_trigger_mode"] == "manual"
    assert summary["render_image_shape"] == [6, 7]
