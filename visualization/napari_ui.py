"""Napari scene composition for sweep visualization.

This module keeps the scene-management logic separate from the Qt event loop so
it can be tested against simple fake viewers in CLI mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visualization.app import VisualizationAppState
from visualization.comparison import build_comparison_payload
from visualization.probe_orientation import pose_from_yaw_pitch_roll, pose_to_yaw_pitch_roll
from visualization.probe_representation import ProbeRepresentation, build_probe_representation
from visualization.render_controller import RenderController
from visualization.render_panel import extract_render_image, format_render_metadata
from visualization.trajectory import TrajectoryOverlay, nearest_trajectory_index
from visualization.transforms import ensure_pose_matrix


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


def _vectors_from_trajectory(trajectory: TrajectoryOverlay) -> np.ndarray:
    if trajectory.axis_origins_mm.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(
        [
            trajectory.axis_origins_mm,
            trajectory.axis_endpoints_mm - trajectory.axis_origins_mm,
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class SceneState:
    """UI-facing scene state kept in sync with the viewer."""

    probe_pose_mm: np.ndarray
    probe_representation: ProbeRepresentation
    comparison_payload: dict[str, Any]
    rendered_output: dict[str, Any] | None = None


class VisualizationUIController:
    """Manage napari overlays for the visualization workflow."""

    def __init__(
        self,
        viewer: Any,
        app_state: VisualizationAppState,
        *,
        render_controller: RenderController | None = None,
    ) -> None:
        self.viewer = viewer
        self.app_state = app_state
        self.render_controller = render_controller
        self.state: SceneState | None = None
        self._layers: dict[str, Any] = {}
        self.render_panel: Any | None = None
        self.probe_controls: Any | None = None
        self.comparison_panel: Any | None = None

    def attach_render_panel(self, render_panel: Any) -> None:
        """Attach an optional render panel to receive render-status updates."""
        self.render_panel = render_panel
        self._refresh_render_panel()

    def attach_probe_controls(self, probe_controls: Any) -> None:
        """Attach an optional probe-controls panel to receive pose updates."""
        self.probe_controls = probe_controls
        self._refresh_probe_controls()

    def attach_comparison_panel(self, comparison_panel: Any) -> None:
        """Attach an optional nearest-frame comparison panel."""
        self.comparison_panel = comparison_panel
        self._refresh_comparison_panel()

    def initialize(self, probe_pose_mm: np.ndarray | None = None) -> SceneState:
        """Add trajectory and probe overlays to the viewer."""
        if probe_pose_mm is None:
            probe_pose_mm = self.app_state.poses_mm[0]
        pose = ensure_pose_matrix(probe_pose_mm).astype(np.float32)
        self._add_trajectory_layers()
        self._set_probe_layers(pose)
        comparison_payload = build_comparison_payload(
            rendered_output={},
            query_pose_mm=pose,
            recorded_images=self.app_state.images,
            recorded_poses_mm=self.app_state.poses_mm,
        )
        rendered_output = None
        if self.render_controller is not None:
            self.render_controller.initialize(pose)
            rendered_output = self.render_controller.state.last_render_output
            if rendered_output is not None:
                comparison_payload = build_comparison_payload(
                    rendered_output=rendered_output,
                    query_pose_mm=pose,
                    recorded_images=self.app_state.images,
                    recorded_poses_mm=self.app_state.poses_mm,
                )
        self.state = SceneState(
            probe_pose_mm=pose,
            probe_representation=build_probe_representation(pose, self.app_state.probe_geometry),
            comparison_payload=comparison_payload,
            rendered_output=rendered_output,
        )
        self._refresh_render_panel()
        self._refresh_probe_controls()
        self._refresh_comparison_panel()
        return self.state

    def set_probe_pose(self, probe_pose_mm: np.ndarray) -> SceneState:
        """Update the probe overlay and refresh derived state."""
        if self.state is None:
            return self.initialize(probe_pose_mm)

        pose = ensure_pose_matrix(probe_pose_mm).astype(np.float32)
        self._set_probe_layers(pose)
        rendered_output = self.state.rendered_output
        if self.render_controller is not None:
            maybe_output = self.render_controller.set_probe_pose(pose)
            rendered_output = maybe_output if maybe_output is not None else self.render_controller.state.last_render_output
        comparison_payload = build_comparison_payload(
            rendered_output=rendered_output or {},
            query_pose_mm=pose,
            recorded_images=self.app_state.images,
            recorded_poses_mm=self.app_state.poses_mm,
        )
        self.state = SceneState(
            probe_pose_mm=pose,
            probe_representation=build_probe_representation(pose, self.app_state.probe_geometry),
            comparison_payload=comparison_payload,
            rendered_output=rendered_output,
        )
        self._refresh_render_panel()
        self._refresh_probe_controls()
        self._refresh_comparison_panel()
        return self.state

    def set_probe_to_recorded_pose(self, index: int) -> SceneState:
        """Move the virtual probe to one of the recorded sweep poses."""
        return self.set_probe_pose(self.app_state.poses_mm[index])

    def set_probe_pose_from_components(
        self,
        *,
        origin_mm: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
    ) -> SceneState:
        """Set the probe pose from translation and yaw/pitch/roll controls."""
        pose = pose_from_yaw_pitch_roll(
            np.asarray(origin_mm, dtype=np.float32),
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
        )
        return self.set_probe_pose(pose)

    def snap_probe_to_nearest_recorded_pose(self) -> SceneState:
        """Snap the probe to the nearest recorded trajectory center."""
        if self.state is None:
            raise RuntimeError("VisualizationUIController must be initialized before use")
        current_origin = self.state.probe_pose_mm[:3, 3]
        nearest_index = nearest_trajectory_index(current_origin, self.app_state.trajectory.centers_mm)
        return self.set_probe_to_recorded_pose(nearest_index)

    def render_now(self) -> dict[str, Any]:
        """Render the current probe pose and refresh comparison state."""
        if self.render_controller is None:
            raise RuntimeError("VisualizationUIController has no render controller configured")
        if self.state is None:
            raise RuntimeError("VisualizationUIController must be initialized before rendering")
        output = self.render_controller.render_current_pose(force=True)
        self.state.rendered_output = output
        self.state.comparison_payload = build_comparison_payload(
            rendered_output=output,
            query_pose_mm=self.state.probe_pose_mm,
            recorded_images=self.app_state.images,
            recorded_poses_mm=self.app_state.poses_mm,
        )
        self._refresh_render_panel()
        self._refresh_comparison_panel()
        return output

    def _add_trajectory_layers(self) -> None:
        trajectory = self.app_state.trajectory
        if "trajectory_path" not in self._layers:
            self._layers["trajectory_path"] = self.viewer.add_shapes(
                _polyline_shape(trajectory.polyline_mm),
                shape_type="path",
                name="trajectory_path",
                edge_color="yellow",
                edge_width=2,
                face_color="transparent",
            )
        if "trajectory_centers" not in self._layers:
            self._layers["trajectory_centers"] = self.viewer.add_points(
                trajectory.centers_mm,
                name="trajectory_centers",
                size=4,
                face_color="yellow",
                border_color="black",
            )
        if "trajectory_axes" not in self._layers:
            self._layers["trajectory_axes"] = self.viewer.add_vectors(
                _vectors_from_trajectory(trajectory),
                name="trajectory_axes",
                edge_color="orange",
                vector_style="line",
                edge_width=1,
            )

    def _set_probe_layers(self, probe_pose_mm: np.ndarray) -> None:
        representation = build_probe_representation(probe_pose_mm, self.app_state.probe_geometry)
        probe_vectors = _vectors_from_axes(representation.origin_mm, representation.axes_endpoints_mm)
        layer_specs = {
            "probe_origin": (
                "points",
                representation.origin_mm[None, :],
                {
                    "name": "probe_origin",
                    "size": 8,
                    "face_color": "cyan",
                    "border_color": "black",
                },
            ),
            "probe_axes": (
                "vectors",
                probe_vectors,
                {
                    "name": "probe_axes",
                    "edge_color": "cyan",
                    "vector_style": "line",
                    "edge_width": 2,
                },
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
        origin = self.state.probe_pose_mm[:3, 3]
        yaw_deg, pitch_deg, roll_deg = pose_to_yaw_pitch_roll(self.state.probe_pose_mm)
        recorded_index = int(self.state.comparison_payload["matched_index"])
        self.probe_controls.set_pose_values(
            origin_mm=origin,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            recorded_index=recorded_index,
        )

    def _refresh_comparison_panel(self) -> None:
        if self.comparison_panel is None:
            return
        if self.state is None:
            self.comparison_panel.set_status("No comparison available")
            self.comparison_panel.set_metadata("")
            return
        from visualization.comparison_panel import extract_matched_image, format_comparison_metadata

        self.comparison_panel.set_status("Comparison ready")
        self.comparison_panel.set_metadata(format_comparison_metadata(self.state.comparison_payload))
        self.comparison_panel.set_image(extract_matched_image(self.state.comparison_payload))
