"""Controller logic that links probe pose updates to NeRF rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal

import numpy as np


RenderTriggerMode = Literal["manual", "on_pose_change", "on_pose_change_throttled"]


@dataclass
class RenderState:
    """Current render-controller state."""

    probe_pose_mm: np.ndarray
    last_render_output: dict[str, Any] | None = None
    last_render_pose_mm: np.ndarray | None = None
    last_render_time_s: float | None = None
    last_error: str | None = None
    is_rendering: bool = False
    dirty: bool = True


@dataclass
class RenderController:
    """Backend controller for connecting viewer probe pose to NeRF rendering."""

    nerf_session: Any
    trigger_mode: RenderTriggerMode = "manual"
    min_render_interval_s: float = 0.0
    render_overrides: dict[str, Any] = field(default_factory=dict)
    time_fn: Any = monotonic
    state: RenderState | None = None

    def initialize(self, probe_pose_mm: np.ndarray) -> None:
        """Initialize controller state with an initial probe pose."""
        pose = np.asarray(probe_pose_mm, dtype=np.float32)
        self.state = RenderState(probe_pose_mm=pose.copy(), dirty=True)
        if self.trigger_mode in ("on_pose_change", "on_pose_change_throttled"):
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
        if self.trigger_mode == "on_pose_change_throttled" and pose_changed:
            return self.flush_pending_render()
        return None

    def render_current_pose(self, *, force: bool = False) -> dict[str, Any]:
        """Render the NeRF output for the currently selected probe pose."""
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before rendering")
        if not force and self.trigger_mode == "on_pose_change_throttled" and not self._can_render_now():
            raise RuntimeError("Render is throttled; wait for the throttle interval or call flush_pending_render later")

        self.state.is_rendering = True
        self.state.last_error = None
        try:
            output = self.nerf_session.render_pose(self.state.probe_pose_mm, **self.render_overrides)
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.is_rendering = False
            raise
        self.state.last_render_output = output
        self.state.last_render_pose_mm = self.state.probe_pose_mm.copy()
        self.state.last_render_time_s = float(self.time_fn())
        self.state.is_rendering = False
        self.state.dirty = False
        return output

    def flush_pending_render(self, *, force: bool = False) -> dict[str, Any] | None:
        """Render a pending dirty pose if the trigger policy allows it."""
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before rendering")
        if not self.state.dirty and not force:
            return self.state.last_render_output
        if not force and self.trigger_mode == "on_pose_change_throttled" and not self._can_render_now():
            return None
        return self.render_current_pose(force=force)

    def mark_dirty(self) -> None:
        """Mark the render state as needing refresh."""
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before use")
        self.state.dirty = True

    def _can_render_now(self) -> bool:
        if self.state is None:
            raise RuntimeError("RenderController must be initialized before rendering")
        if self.min_render_interval_s <= 0.0:
            return True
        if self.state.last_render_time_s is None:
            return True
        return float(self.time_fn()) - self.state.last_render_time_s >= self.min_render_interval_s
