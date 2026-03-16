"""Launch the sweep-volume visualization workflow.

This entry point prepares a fused 3D sweep volume from tracked 2D ultrasound
data, reuses a cached volume when available, and can either:

- launch a basic napari viewer
- or run in ``--no-gui`` mode for headless validation / preprocessing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualization.app import launch_visualization_app, prepare_visualization_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and launch 3D sweep visualization")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path containing images.npy and poses.npy")
    parser.add_argument("--cache-path", type=str, default=None, help="Optional path to a fused volume cache (.npz)")
    parser.add_argument("--probe-width-mm", type=float, required=True, help="Probe width in millimeters")
    parser.add_argument("--probe-depth-mm", type=float, required=True, help="Probe depth in millimeters")
    parser.add_argument("--spacing-mm", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Voxel spacing in mm")
    parser.add_argument("--pixel-stride", type=int, nargs=2, default=(2, 2), help="Image sampling stride (row, col)")
    parser.add_argument(
        "--preset",
        type=str,
        default="soft_tissue",
        choices=("soft_tissue", "high_contrast", "sparse_signal"),
        help="Initial volume visualization preset",
    )
    parser.add_argument("--no-gui", action="store_true", help="Prepare inputs and print a summary without launching napari")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    state = prepare_visualization_app(
        dataset_dir=args.dataset_dir,
        probe_width_mm=args.probe_width_mm,
        probe_depth_mm=args.probe_depth_mm,
        spacing_mm=tuple(float(v) for v in args.spacing_mm),
        pixel_stride=tuple(int(v) for v in args.pixel_stride),
        cache_path=args.cache_path,
        preset_name=args.preset,
    )

    summary = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "volume_shape": list(state.fused_volume.scalar_volume.shape),
        "spacing_mm": state.fused_volume.spacing_mm.tolist(),
        "origin_mm": state.fused_volume.origin_mm.tolist(),
        "cache_path": str(state.cache_path) if state.cache_path is not None else None,
        "cache_used": state.cache_used,
        "preset": state.preset_name,
        "num_frames": int(state.images.shape[0]),
    }
    print(json.dumps(summary, indent=2))

    if args.no_gui:
        return 0

    viewer = launch_visualization_app(state)
    import napari

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
