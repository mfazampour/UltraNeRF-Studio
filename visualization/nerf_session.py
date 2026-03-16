"""Lazy runtime wrapper for arbitrary-pose NeRF rendering."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

from visualization.transforms import ensure_pose_matrix


class SupportsParseArgs(Protocol):
    def parse_args(self, args: list[str] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class NerfRuntime:
    """Runtime components imported lazily from the existing training code."""

    torch: Any
    config_parser: Any
    create_nerf: Any
    render_us: Any


def import_nerf_runtime() -> NerfRuntime:
    """Import the existing PyTorch runtime lazily.

    This keeps the visualization backend importable in non-GUI and non-rendering
    contexts. The actual training/runtime modules are only imported when a
    session is created.
    """
    import torch

    from nerf_utils import create_nerf, render_us
    from unerf_config import config_parser

    return NerfRuntime(
        torch=torch,
        config_parser=config_parser,
        create_nerf=create_nerf,
        render_us=render_us,
    )


def pose_mm_to_model_pose_m(pose_probe_to_world_mm: np.ndarray) -> np.ndarray:
    """Convert a visualization-space pose in millimeters into model-space meters."""
    pose_mm = ensure_pose_matrix(pose_probe_to_world_mm).astype(np.float32).copy()
    pose_mm[:3, 3] *= 0.001
    return pose_mm


@dataclass
class NerfSession:
    """Wrapper for arbitrary-pose rendering against the existing PyTorch runtime."""

    runtime: NerfRuntime
    args: Any
    device: Any
    image_shape: tuple[int, int]
    probe_width_mm: float
    probe_depth_mm: float
    render_kwargs: dict[str, Any]
    sw_m: float
    sh_m: float
    near_m: float
    far_m: float

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config_path: str,
        checkpoint_path: str,
        image_shape: tuple[int, int],
        probe_width_mm: float,
        probe_depth_mm: float,
        device: str | None = None,
        runtime: NerfRuntime | None = None,
    ) -> "NerfSession":
        runtime = runtime or import_nerf_runtime()
        parser: SupportsParseArgs = runtime.config_parser()
        args = parser.parse_args(["--config", config_path, "--ft_path", checkpoint_path])
        runtime_device = runtime.torch.device(device or ("cuda" if runtime.torch.cuda.is_available() else "cpu"))

        _, render_kwargs_test, _, _, _ = runtime.create_nerf(args, device=runtime_device, mode="test")

        height, width = image_shape
        probe_width_m = float(probe_width_mm) * 0.001
        probe_depth_m = float(probe_depth_mm) * 0.001
        sw_m = probe_width_m / float(width)
        sh_m = probe_depth_m / float(height)
        near_m = 0.0
        far_m = probe_depth_m
        render_kwargs = dict(render_kwargs_test)
        render_kwargs.update({"near": near_m, "far": far_m})

        return cls(
            runtime=runtime,
            args=args,
            device=runtime_device,
            image_shape=(height, width),
            probe_width_mm=float(probe_width_mm),
            probe_depth_mm=float(probe_depth_mm),
            render_kwargs=render_kwargs,
            sw_m=sw_m,
            sh_m=sh_m,
            near_m=near_m,
            far_m=far_m,
        )

    def render_pose(self, pose_probe_to_world_mm: np.ndarray, **render_overrides: Any) -> dict[str, Any]:
        """Render the NeRF output for an arbitrary probe pose in millimeters."""
        pose_m = pose_mm_to_model_pose_m(pose_probe_to_world_mm)
        pose_tensor = self.runtime.torch.from_numpy(pose_m[:3, :4]).to(self.device).unsqueeze(0)
        kwargs = dict(self.render_kwargs)
        kwargs.update(render_overrides)
        return self.runtime.render_us(
            self.image_shape[0],
            self.image_shape[1],
            self.sw_m,
            self.sh_m,
            c2w=pose_tensor,
            chunk=self.args.chunk,
            **kwargs,
        )
