"""Controller logic that links probe pose updates to NeRF rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


RenderTriggerMode = Literal["manual", "on_pose_change"]


@dataclass
class RenderState:
    """Current render-controller state."""

    probe_pose_mm: np.ndarray
    last_render_output: dict[str, Any] | None = None
    last_render_pose_mm: np.ndarray | None = None
    dirty: bool = True


@dataclass
class RenderController:
    """Backend controller for connecting viewer probe pose to NeRF rendering."""

    nerf_session: Any
    trigger_mode: RenderTriggerMode = "manual"
    render_overrides: dict[str, Any] = field(default_factory=dict)
    state: RenderState | None = None

    def initialize(self, probe_pose_mm: np.ndarray) -> None:
        """Initialize controller state with an initial probe pose."""
        pose = np.asarray(probe_pose_mm, dtype=np.float32)
        self.state = RenderState(probe_pose_mm=pose.copy(), dirty=True)
        if self.trigger_mode == "on_pose_change":
            self.render_current_pose()

    def set_probe_pose(self, probe_pose_mm: np.ndarray) -> dict[str, Any] | None:
        """Update the current probe pose and render if configured to do so."""
        if self.state is None:
            self.initialize(probe_pose_mm)
            return self.state.last_render_output

        pose = np.asarray(probe_pose_mm, dtype=np.float32)
        pose_changed = not np.allclose(self.state.probe_pose_mm, pose)
        self.state.probe_pose_mm = pose.copy()
        self.state.dirty = self.state.dirty or pose_changed
        if self.trigger_mode == "on_pose_change" and pose_changed:
            return self.render_current_pose()
        return None

    def render_current_pose(self) -> dict[str, Any]:
        """Render the NeRF output for the currently selected probe pose."""
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before rendering")

        output = self.nerf_session.render_pose(self.state.probe_pose_mm, **self.render_overrides)
        self.state.last_render_output = output
        self.state.last_render_pose_mm = self.state.probe_pose_mm.copy()
        self.state.dirty = False
        return output

    def mark_dirty(self) -> None:
        """Mark the render state as needing refresh."""
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before use")
        self.state.dirty = True
