"""Launch the sweep-volume visualization workflow.

This entry point prepares a fused 3D sweep volume from tracked 2D ultrasound
data, reuses a cached volume when available, and can either:

- launch a basic napari viewer
- or run in ``--no-gui`` mode for headless validation / preprocessing
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if ROOT.name == "scripts":
    SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
from pathlib import Path

from ultranerf.visualization.app import NerfLaunchConfig, launch_visualization_app, prepare_visualization_app, resolve_render_image_shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and launch 3D sweep visualization")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path containing images.npy and poses.npy")
    parser.add_argument("--cache-path", type=str, default=None, help="Optional path to a fused volume cache (.npz)")
    parser.add_argument("--probe-width-mm", type=float, required=True, help="Probe width in millimeters")
    parser.add_argument("--probe-depth-mm", type=float, required=True, help="Probe depth in millimeters")
    parser.add_argument(
        "--fusion-device",
        type=str,
        default="auto",
        help="Device for sweep-to-volume fusion: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--fusion-reduction",
        type=str,
        default="max",
        choices=("mean", "max"),
        help="How overlapping voxel contributions are combined during sweep fusion",
    )
    parser.add_argument("--spacing-mm", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Voxel spacing in mm")
    parser.add_argument("--pixel-stride", type=int, nargs=2, default=(2, 2), help="Image sampling stride (row, col)")
    parser.add_argument(
        "--preset",
        type=str,
        default="soft_tissue",
        choices=("soft_tissue", "high_contrast", "sparse_signal"),
        help="Initial volume visualization preset",
    )
    parser.add_argument(
        "--initial-pose-index",
        type=int,
        default=0,
        help="Recorded pose index used to initialize the probe overlay",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional UltraNeRF checkpoint to attach")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config_base_nerf.txt",
        help="Config file used to initialize the NeRF runtime when a checkpoint is provided",
    )
    parser.add_argument(
        "--render-trigger-mode",
        type=str,
        default="manual",
        choices=("manual", "on_pose_change", "on_pose_change_throttled"),
        help="When NeRF is enabled, control whether renders happen manually or on probe pose changes",
    )
    parser.add_argument(
        "--min-render-interval-ms",
        type=float,
        default=0.0,
        help="Minimum spacing between automatic renders when throttled mode is active",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=None,
        help="Optional override for the NeRF render height in pixels",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=None,
        help="Optional override for the NeRF render width in pixels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override for checkpoint-backed rendering, for example cpu or cuda:0",
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
        fusion_device=args.fusion_device,
        reduction_mode=args.fusion_reduction,
    )
    nerf_config = None
    nerf_enabled = args.checkpoint_path is not None
    render_shape = None
    if nerf_enabled:
        render_shape = resolve_render_image_shape(
            state.images,
            render_height=args.render_height,
            render_width=args.render_width,
        )
        checkpoint_path = Path(args.checkpoint_path)
        config_path = Path(args.config_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        nerf_config = NerfLaunchConfig(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            trigger_mode=args.render_trigger_mode,
            min_render_interval_ms=args.min_render_interval_ms,
            render_image_shape=render_shape,
            device=args.device,
        )

    summary = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "volume_shape": list(state.fused_volume.scalar_volume.shape),
        "spacing_mm": state.fused_volume.spacing_mm.tolist(),
        "origin_mm": state.fused_volume.origin_mm.tolist(),
        "cache_path": str(state.cache_path) if state.cache_path is not None else None,
        "cache_used": state.cache_used,
        "preset": state.preset_name,
        "fusion_device": state.fusion_device,
        "fusion_reduction": state.reduction_mode,
        "num_frames": int(state.images.shape[0]),
        "nerf_enabled": nerf_enabled,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()) if args.checkpoint_path is not None else None,
        "config_path": str(Path(args.config_path).resolve()) if nerf_enabled else None,
        "render_trigger_mode": args.render_trigger_mode if nerf_enabled else None,
        "min_render_interval_ms": args.min_render_interval_ms if nerf_enabled else None,
        "render_image_shape": list(render_shape) if render_shape is not None else None,
    }
    print(json.dumps(summary, indent=2))

    if args.no_gui:
        return 0

    _session = launch_visualization_app(
        state,
        initial_pose_index=args.initial_pose_index,
        nerf_config=nerf_config,
    )
    import napari

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
