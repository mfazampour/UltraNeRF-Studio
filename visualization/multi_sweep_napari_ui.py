"""Napari scene composition for multi-sweep visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visualization.comparison_panel import extract_matched_image, format_comparison_metadata
from visualization.multi_sweep_app import MultiSweepVisualizationAppState
from visualization.multi_sweep_comparison import build_multi_sweep_comparison_payload
from visualization.multi_sweep_ui import MultiSweepViewerState
from visualization.probe_orientation import pose_from_yaw_pitch_roll, pose_to_yaw_pitch_roll
from visualization.probe_representation import build_probe_representation
from visualization.render_controller import RenderController
from visualization.render_panel import extract_render_image, format_render_metadata
from visualization.transforms import ensure_pose_matrix
from visualization.volume_viewer import build_volume_layer_config_from_preset


def _color_to_hex(color: tuple[float, float, float] | None, *, default: str) -> str:
    """Convert a normalized RGB tuple into a napari-safe hex color string."""
    if color is None:
        return default
    rgb = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    if rgb.shape != (3,):
        return default
    r, g, b = np.rint(rgb * 255.0).astype(np.uint8).tolist()
    return f"#{r:02x}{g:02x}{b:02x}"


def _polyline_shape(points_mm: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(points_mm, dtype=np.float32)]


def _line_shape(points_mm: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(points_mm, dtype=np.float32)]


def _polygon_shape(points_mm: np.ndarray) -> list[np.ndarray]:
    polygon = np.asarray(points_mm, dtype=np.float32)
    if polygon.shape[0] >= 3 and not np.allclose(polygon[0], polygon[-1]):
        polygon = np.concatenate([polygon, polygon[:1]], axis=0)
    return [polygon]


def _vectors_from_axes(origin_mm: np.ndarray, endpoints_mm: dict[str, np.ndarray]) -> np.ndarray:
    vectors = []
    for axis_name in ("x", "y", "z"):
        endpoint = np.asarray(endpoints_mm[axis_name], dtype=np.float32)
        origin = np.asarray(origin_mm, dtype=np.float32)
        vectors.append(np.stack([origin, endpoint - origin], axis=0))
    return np.asarray(vectors, dtype=np.float32)


def _vectors_from_trajectory(trajectory) -> np.ndarray:
    if trajectory.axis_origins_mm.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(
        [
            trajectory.axis_origins_mm,
            trajectory.axis_endpoints_mm - trajectory.axis_origins_mm,
        ],
        axis=1,
    ).astype(np.float32)


def _set_layer_visibility(layer: Any, visible: bool) -> None:
    try:
        layer.visible = bool(visible)
    except Exception:
        setattr(layer, "visible", bool(visible))


def _compute_aggregate_contrast_limits(volume_data: np.ndarray) -> tuple[float, float]:
    """Return aggregate-specific contrast limits with strong low-signal suppression."""
    data = np.asarray(volume_data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0)
    positive = finite[finite > 0.0]
    values = positive if positive.size > 0 else finite
    lower = float(np.percentile(values, 88.0))
    upper = float(np.percentile(values, 99.7))
    if upper <= lower:
        lower = float(np.min(values))
        upper = float(np.max(values))
    if upper <= lower:
        upper = lower + 1.0
    return (lower, upper)


@dataclass
class MultiSweepSceneState:
    probe_pose_mm: np.ndarray
    comparison_payload: dict[str, Any]
    rendered_output: dict[str, Any] | None = None


class MultiSweepVisualizationUIController:
    """Manage napari overlays for a multi-sweep visualization session."""

    def __init__(
        self,
        viewer: Any,
        app_state: MultiSweepVisualizationAppState,
        *,
        render_controller: RenderController | None = None,
    ) -> None:
        self.viewer = viewer
        self.app_state = app_state
        self.render_controller = render_controller
        self.state: MultiSweepSceneState | None = None
        self._layers: dict[str, Any] = {}
        self.render_panel: Any | None = None
        self.probe_controls: Any | None = None
        self.comparison_panel: Any | None = None
        self.multi_sweep_controls: Any | None = None
        self.sweep_selection_controls: Any | None = None

    def attach_render_panel(self, render_panel: Any) -> None:
        self.render_panel = render_panel
        self._refresh_render_panel()

    def attach_probe_controls(self, probe_controls: Any) -> None:
        self.probe_controls = probe_controls
        self._refresh_probe_controls()

    def attach_comparison_panel(self, comparison_panel: Any) -> None:
        self.comparison_panel = comparison_panel
        self._refresh_comparison_panel()

    def attach_multi_sweep_controls(self, multi_sweep_controls: Any) -> None:
        self.multi_sweep_controls = multi_sweep_controls

    def attach_sweep_selection_controls(self, sweep_selection_controls: Any) -> None:
        self.sweep_selection_controls = sweep_selection_controls

    def initialize(self, probe_pose_mm: np.ndarray | None = None) -> MultiSweepSceneState:
        if probe_pose_mm is None:
            probe_pose_mm = self.app_state.scene.active_sweep.poses_mm[0]
        pose = ensure_pose_matrix(probe_pose_mm).astype(np.float32)
        self._refresh_multi_sweep_scene_layers()
        self._set_probe_layers(pose)
        comparison_payload = self._build_comparison_payload(pose, rendered_output={})
        rendered_output = None
        if self.render_controller is not None:
            self.render_controller.initialize(pose)
            rendered_output = self.render_controller.state.last_render_output
            if rendered_output is not None:
                comparison_payload = self._build_comparison_payload(pose, rendered_output=rendered_output)
        self.state = MultiSweepSceneState(
            probe_pose_mm=pose,
            comparison_payload=comparison_payload,
            rendered_output=rendered_output,
        )
        self._refresh_probe_controls()
        self._refresh_sweep_selection_controls()
        self._refresh_comparison_panel()
        self._refresh_render_panel()
        return self.state

    def handle_multi_sweep_state_change(self, _viewer_state: MultiSweepViewerState) -> None:
        self._refresh_multi_sweep_scene_layers()
        if self.state is not None:
            self.state.comparison_payload = self._build_comparison_payload(
                self.state.probe_pose_mm,
                rendered_output=self.state.rendered_output or {},
            )
        self._refresh_probe_controls()
        self._refresh_sweep_selection_controls()
        self._refresh_comparison_panel()

    def set_active_sweep(self, sweep_id: str) -> MultiSweepSceneState:
        self.app_state.scene_controller.set_active_sweep(sweep_id)
        self._refresh_multi_sweep_scene_layers()
        if self.multi_sweep_controls is not None:
            self.multi_sweep_controls.refresh()
        if self.state is None:
            return self.initialize(self.app_state.scene.get_sweep(sweep_id).poses_mm[0])
        self.state.comparison_payload = self._build_comparison_payload(
            self.state.probe_pose_mm,
            rendered_output=self.state.rendered_output or {},
        )
        self._refresh_probe_controls()
        self._refresh_sweep_selection_controls()
        self._refresh_comparison_panel()
        return self.state

    def set_probe_pose(self, probe_pose_mm: np.ndarray) -> MultiSweepSceneState:
        if self.state is None:
            return self.initialize(probe_pose_mm)
        pose = ensure_pose_matrix(probe_pose_mm).astype(np.float32)
        self._set_probe_layers(pose)
        rendered_output = self.state.rendered_output
        if self.render_controller is not None:
            maybe_output = self.render_controller.set_probe_pose(pose)
            rendered_output = maybe_output if maybe_output is not None else self.render_controller.state.last_render_output
        self.state = MultiSweepSceneState(
            probe_pose_mm=pose,
            comparison_payload=self._build_comparison_payload(pose, rendered_output=rendered_output or {}),
            rendered_output=rendered_output,
        )
        self._refresh_probe_controls()
        self._refresh_comparison_panel()
        self._refresh_render_panel()
        return self.state

    def set_probe_pose_from_components(
        self,
        *,
        origin_mm: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
    ) -> MultiSweepSceneState:
        pose = pose_from_yaw_pitch_roll(
            np.asarray(origin_mm, dtype=np.float32),
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
        )
        return self.set_probe_pose(pose)

    def set_probe_to_recorded_pose(self, index: int) -> MultiSweepSceneState:
        active_sweep = self.app_state.scene.get_sweep(self.app_state.scene_controller.state.active_sweep_id)
        safe_index = min(max(int(index), 0), active_sweep.frame_count - 1)
        return self.set_probe_pose(active_sweep.poses_mm[safe_index])

    def snap_probe_to_nearest_recorded_pose(self) -> MultiSweepSceneState:
        if self.state is None:
            raise RuntimeError("MultiSweepVisualizationUIController must be initialized before use")
        active_sweep = self.app_state.scene.get_sweep(self.app_state.scene_controller.state.active_sweep_id)
        centers = active_sweep.poses_mm[:, :3, 3]
        distances = np.linalg.norm(centers - self.state.probe_pose_mm[:3, 3][None, :], axis=1)
        return self.set_probe_to_recorded_pose(int(np.argmin(distances)))

    def render_now(self) -> dict[str, Any]:
        if self.render_controller is None:
            raise RuntimeError("MultiSweepVisualizationUIController has no render controller configured")
        if self.state is None:
            raise RuntimeError("MultiSweepVisualizationUIController must be initialized before rendering")
        output = self.render_controller.render_current_pose(force=True)
        self.state.rendered_output = output
        self.state.comparison_payload = self._build_comparison_payload(self.state.probe_pose_mm, rendered_output=output)
        self._refresh_comparison_panel()
        self._refresh_render_panel()
        return output

    def _build_comparison_payload(self, pose_mm: np.ndarray, *, rendered_output: dict[str, Any]) -> dict[str, Any]:
        state = self.app_state.scene_controller.state
        return build_multi_sweep_comparison_payload(
            rendered_output=rendered_output,
            query_pose_mm=pose_mm,
            scene=self.app_state.scene,
            active_sweep_id=state.active_sweep_id,
            comparison_policy=state.comparison_policy,
            allowed_sweep_ids=state.enabled_sweep_ids,
        )

    def _refresh_multi_sweep_scene_layers(self) -> None:
        fusion_result = self.app_state.scene_controller.build_fusion_result()
        self.app_state.fusion_result = fusion_result
        state = self.app_state.scene_controller.state
        active_id = state.active_sweep_id
        trajectory_visible_ids = {active_id} if state.show_aggregate_volume else set(fusion_result.enabled_sweep_ids)
        aggregate_config = build_volume_layer_config_from_preset(
            fusion_result.aggregate_volume,
            preset_name=self.app_state.preset_name,
            name="sweep_volume__aggregate",
        )
        aggregate_layer = self._layers.get("sweep_volume__aggregate")
        if aggregate_layer is None:
            self._layers["sweep_volume__aggregate"] = self.viewer.add_image(
                aggregate_config.data,
                scale=aggregate_config.scale,
                translate=aggregate_config.translate,
                name=aggregate_config.name,
                rendering=aggregate_config.rendering,
                colormap=aggregate_config.colormap,
                opacity=aggregate_config.opacity,
                blending=aggregate_config.blending,
                contrast_limits=aggregate_config.contrast_limits,
            )
            aggregate_layer = self._layers["sweep_volume__aggregate"]
        else:
            aggregate_layer.data = aggregate_config.data
            if hasattr(aggregate_layer, "scale"):
                aggregate_layer.scale = aggregate_config.scale
            if hasattr(aggregate_layer, "translate"):
                aggregate_layer.translate = aggregate_config.translate
        if hasattr(aggregate_layer, "opacity"):
            aggregate_layer.opacity = min(float(aggregate_config.opacity), 0.10)
        if hasattr(aggregate_layer, "contrast_limits"):
            aggregate_layer.contrast_limits = _compute_aggregate_contrast_limits(aggregate_config.data)
        _set_layer_visibility(aggregate_layer, state.show_aggregate_volume)

        visible_ids = set(fusion_result.enabled_sweep_ids)
        for overlay in fusion_result.sweep_overlays:
            color = _color_to_hex(overlay.color_rgb, default="#cccc33")
            volume_name = f"sweep_volume__{overlay.sweep_id}"
            layer = self._layers.get(volume_name)
            if overlay.fused_volume is not None:
                volume_config = build_volume_layer_config_from_preset(
                    overlay.fused_volume,
                    preset_name=self.app_state.preset_name,
                    name=volume_name,
                )
                if layer is None:
                    self._layers[volume_name] = self.viewer.add_image(
                        volume_config.data,
                        scale=volume_config.scale,
                        translate=volume_config.translate,
                        name=volume_config.name,
                        rendering=volume_config.rendering,
                        colormap=volume_config.colormap,
                        opacity=max(0.10, min(0.22, volume_config.opacity * 0.45)),
                        blending=volume_config.blending,
                        contrast_limits=volume_config.contrast_limits,
                    )
                    layer = self._layers[volume_name]
                else:
                    layer.data = volume_config.data
                    if hasattr(layer, "scale"):
                        layer.scale = volume_config.scale
                    if hasattr(layer, "translate"):
                        layer.translate = volume_config.translate
                if hasattr(layer, "opacity"):
                    layer.opacity = max(0.10, min(0.22, volume_config.opacity * 0.45))
                _set_layer_visibility(layer, not state.show_aggregate_volume)
            else:
                if layer is not None:
                    _set_layer_visibility(layer, False)

            path_name = f"trajectory_path__{overlay.sweep_id}"
            centers_name = f"trajectory_centers__{overlay.sweep_id}"
            axes_name = f"trajectory_axes__{overlay.sweep_id}"
            path_layer = self._layers.get(path_name)
            if path_layer is None:
                self._layers[path_name] = self.viewer.add_shapes(
                    _polyline_shape(overlay.trajectory.polyline_mm),
                    shape_type="path",
                    name=path_name,
                    edge_color=color,
                    edge_width=4 if overlay.sweep_id == active_id else 2,
                    face_color="transparent",
                )
                path_layer = self._layers[path_name]
            else:
                path_layer.data = _polyline_shape(overlay.trajectory.polyline_mm)
            if hasattr(path_layer, "edge_width"):
                path_layer.edge_width = 4 if overlay.sweep_id == active_id else 2
            _set_layer_visibility(path_layer, overlay.sweep_id in trajectory_visible_ids)

            centers_layer = self._layers.get(centers_name)
            if centers_layer is None:
                self._layers[centers_name] = self.viewer.add_points(
                    overlay.trajectory.centers_mm,
                    name=centers_name,
                    size=6 if overlay.sweep_id == active_id else 3,
                    face_color=color,
                    border_color="black",
                )
                centers_layer = self._layers[centers_name]
            else:
                centers_layer.data = overlay.trajectory.centers_mm
            if hasattr(centers_layer, "size"):
                centers_layer.size = 6 if overlay.sweep_id == active_id else 3
            _set_layer_visibility(centers_layer, overlay.sweep_id in trajectory_visible_ids)

            axes_layer = self._layers.get(axes_name)
            if axes_layer is None:
                self._layers[axes_name] = self.viewer.add_vectors(
                    _vectors_from_trajectory(overlay.trajectory),
                    name=axes_name,
                    edge_color=color,
                    vector_style="line",
                    edge_width=1,
                )
                axes_layer = self._layers[axes_name]
            else:
                axes_layer.data = _vectors_from_trajectory(overlay.trajectory)
            _set_layer_visibility(axes_layer, overlay.sweep_id == active_id if state.show_aggregate_volume else overlay.sweep_id in trajectory_visible_ids)

        for sweep in self.app_state.scene.sweeps:
            if sweep.sweep_id in visible_ids:
                continue
            for prefix in ("sweep_volume__", "trajectory_path__", "trajectory_centers__", "trajectory_axes__"):
                layer = self._layers.get(f"{prefix}{sweep.sweep_id}")
                if layer is not None:
                    _set_layer_visibility(layer, False)

    def _set_probe_layers(self, probe_pose_mm: np.ndarray) -> None:
        representation = build_probe_representation(probe_pose_mm, self.app_state.probe_geometry)
        probe_vectors = _vectors_from_axes(representation.origin_mm, representation.axes_endpoints_mm)
        layer_specs = {
            "probe_origin": (
                "points",
                representation.origin_mm[None, :],
                {"name": "probe_origin", "size": 8, "face_color": "cyan", "border_color": "black"},
            ),
            "probe_axes": (
                "vectors",
                probe_vectors,
                {"name": "probe_axes", "edge_color": "cyan", "vector_style": "line", "edge_width": 2},
            ),
            "probe_scan_plane": (
                "shapes",
                _polygon_shape(representation.scan_plane_corners_mm),
                {
                    "name": "probe_scan_plane",
                    "shape_type": "polygon",
                    "edge_color": "cyan",
                    "edge_width": 1,
                    "face_color": "transparent",
                },
            ),
            "probe_beam_line": (
                "shapes",
                _line_shape(representation.beam_line_mm),
                {
                    "name": "probe_beam_line",
                    "shape_type": "line",
                    "edge_color": "magenta",
                    "edge_width": 2,
                    "face_color": "transparent",
                },
            ),
            "probe_face_line": (
                "shapes",
                _line_shape(representation.probe_face_line_mm),
                {
                    "name": "probe_face_line",
                    "shape_type": "line",
                    "edge_color": "cyan",
                    "edge_width": 3,
                    "face_color": "transparent",
                },
            ),
        }
        for layer_name, (layer_type, data, kwargs) in layer_specs.items():
            layer = self._layers.get(layer_name)
            if layer is None:
                add_method = getattr(self.viewer, f"add_{layer_type}")
                self._layers[layer_name] = add_method(data, **kwargs)
            else:
                layer.data = data

    def _refresh_render_panel(self) -> None:
        if self.render_panel is None:
            return
        if self.render_controller is None:
            self.render_panel.set_status("Sweep-only mode")
            self.render_panel.set_metadata("No checkpoint configured")
            return
        render_state = self.render_controller.state
        if render_state is not None and render_state.last_error is not None:
            self.render_panel.set_status("Render failed")
            self.render_panel.set_metadata(render_state.last_error)
            return
        if render_state is not None and render_state.is_rendering:
            self.render_panel.set_status("Rendering...")
            self.render_panel.set_metadata("Render in progress")
            return
        if self.state is None or self.state.rendered_output is None:
            self.render_panel.set_status("Ready")
            self.render_panel.set_metadata("No render available")
            return
        self.render_panel.set_status("Rendered")
        self.render_panel.set_metadata(format_render_metadata(self.state.rendered_output))
        self.render_panel.set_image(extract_render_image(self.state.rendered_output))

    def _refresh_probe_controls(self) -> None:
        if self.probe_controls is None or self.state is None:
            return
        active_sweep = self.app_state.scene.get_sweep(self.app_state.scene_controller.state.active_sweep_id)
        centers = active_sweep.poses_mm[:, :3, 3]
        distances = np.linalg.norm(centers - self.state.probe_pose_mm[:3, 3][None, :], axis=1)
        recorded_index = int(np.argmin(distances))
        origin = self.state.probe_pose_mm[:3, 3]
        yaw_deg, pitch_deg, roll_deg = pose_to_yaw_pitch_roll(self.state.probe_pose_mm)
        if hasattr(self.probe_controls, "set_num_frames"):
            self.probe_controls.set_num_frames(active_sweep.frame_count)
        self.probe_controls.set_pose_values(
            origin_mm=origin,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            recorded_index=recorded_index,
        )

    def _refresh_sweep_selection_controls(self) -> None:
        if self.sweep_selection_controls is None:
            return
        self.sweep_selection_controls.refresh()

    def _refresh_comparison_panel(self) -> None:
        if self.comparison_panel is None:
            return
        if self.state is None:
            self.comparison_panel.set_status("No comparison available")
            self.comparison_panel.set_metadata("")
            return
        self.comparison_panel.set_status("Comparison ready")
        self.comparison_panel.set_metadata(format_comparison_metadata(self.state.comparison_payload))
        self.comparison_panel.set_image(extract_matched_image(self.state.comparison_payload))
