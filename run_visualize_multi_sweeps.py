"""Launch the multi-sweep visualization workflow from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualization.app import NerfLaunchConfig
from visualization.multi_sweep_app import (
    launch_multi_sweep_visualization_app,
    prepare_multi_sweep_visualization_app,
    resolve_multi_sweep_render_image_shape,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and launch multi-sweep 3D visualization")
    parser.add_argument("--manifest-path", type=str, required=True, help="Path to a multi-sweep JSON manifest")
    parser.add_argument("--cache-root", type=str, default=None, help="Optional cache root for future multi-sweep caches")
    parser.add_argument("--spacing-mm", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Voxel spacing in mm")
    parser.add_argument("--pixel-stride", type=int, nargs=2, default=(2, 2), help="Image sampling stride (row, col)")
    parser.add_argument(
        "--preset",
        type=str,
        default="soft_tissue",
        choices=("soft_tissue", "high_contrast", "sparse_signal"),
        help="Initial volume visualization preset",
    )
    parser.add_argument("--initial-pose-index", type=int, default=0, help="Initial recorded pose index within the active sweep")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional UltraNeRF checkpoint to attach")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config_base_nerf.txt",
        help="Config file used when a checkpoint is provided",
    )
    parser.add_argument(
        "--render-trigger-mode",
        type=str,
        default="manual",
        choices=("manual", "on_pose_change", "on_pose_change_throttled"),
        help="When NeRF is enabled, control whether renders happen manually or on probe pose changes",
    )
    parser.add_argument("--min-render-interval-ms", type=float, default=0.0, help="Minimum spacing between automatic renders")
    parser.add_argument("--render-height", type=int, default=None, help="Optional override for render height")
    parser.add_argument("--render-width", type=int, default=None, help="Optional override for render width")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override, for example cpu or cuda:0")
    parser.add_argument("--no-gui", action="store_true", help="Prepare inputs and print a summary without launching napari")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    state = prepare_multi_sweep_visualization_app(
        manifest_path=args.manifest_path,
        spacing_mm=tuple(float(v) for v in args.spacing_mm),
        pixel_stride=tuple(int(v) for v in args.pixel_stride),
        preset_name=args.preset,
        cache_root=args.cache_root,
    )
    nerf_enabled = args.checkpoint_path is not None
    render_shape = None
    nerf_config = None
    if nerf_enabled:
        render_shape = resolve_multi_sweep_render_image_shape(
            state.scene,
            render_height=args.render_height,
            render_width=args.render_width,
            active_sweep_id=state.scene_controller.state.active_sweep_id,
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
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "num_sweeps": len(state.scene.sweeps),
        "sweep_ids": list(state.scene.sweep_ids),
        "active_sweep_id": state.scene_controller.state.active_sweep_id,
        "enabled_sweep_ids": list(state.scene_controller.state.enabled_sweep_ids),
        "aggregate_volume_shape": list(state.fusion_result.aggregate_volume.scalar_volume.shape),
        "spacing_mm": state.fusion_result.aggregate_volume.spacing_mm.tolist(),
        "origin_mm": state.fusion_result.aggregate_volume.origin_mm.tolist(),
        "alignment_ok": state.alignment_validation.is_plausibly_aligned,
        "alignment_warning_count": len(state.alignment_validation.warnings),
        "alignment_warnings": list(state.alignment_validation.warnings),
        "cache_root": str(state.cache_root) if state.cache_root is not None else None,
        "preset": state.preset_name,
        "nerf_enabled": nerf_enabled,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()) if nerf_enabled else None,
        "config_path": str(Path(args.config_path).resolve()) if nerf_enabled else None,
        "render_trigger_mode": args.render_trigger_mode if nerf_enabled else None,
        "min_render_interval_ms": args.min_render_interval_ms if nerf_enabled else None,
        "render_image_shape": list(render_shape) if render_shape is not None else None,
    }
    print(json.dumps(summary, indent=2))
    if args.no_gui:
        return 0

    _session = launch_multi_sweep_visualization_app(
        state,
        initial_pose_index=args.initial_pose_index,
        nerf_config=nerf_config,
    )
    import napari

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
