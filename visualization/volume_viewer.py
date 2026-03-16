"""Basic napari-backed 3D viewer for fused sweep volumes.

This module keeps GUI imports lazy so the backend can be developed and tested in
CLI mode. The fused volume remains the source of truth; the viewer only prepares
layer arguments and creates a napari session when explicitly launched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visualization.sweep_volume import FusedSweepVolume
from visualization.volume_presets import VolumePreset, get_volume_preset


@dataclass(frozen=True)
class VolumeLayerConfig:
    """Napari layer configuration for a fused scalar volume."""

    data: np.ndarray
    scale: tuple[float, float, float]
    translate: tuple[float, float, float]
    name: str
    rendering: str
    colormap: str
    opacity: float
    blending: str
    axis_labels: tuple[str, str, str]
    contrast_limits: tuple[float, float]


def build_volume_layer_config(
    fused_volume: FusedSweepVolume,
    *,
    name: str = "sweep_volume",
    rendering: str = "mip",
    colormap: str = "gray",
    opacity: float = 0.5,
    blending: str = "translucent",
) -> VolumeLayerConfig:
    """Convert a fused volume into a napari layer configuration."""
    data = np.asarray(fused_volume.scalar_volume, dtype=np.float32)
    contrast_limits = (float(np.min(data)), float(np.max(data)))
    return VolumeLayerConfig(
        data=data,
        scale=tuple(float(v) for v in fused_volume.spacing_mm),
        translate=tuple(float(v) for v in fused_volume.origin_mm),
        name=name,
        rendering=rendering,
        colormap=colormap,
        opacity=float(opacity),
        blending=blending,
        axis_labels=("x_mm", "y_mm", "z_mm"),
        contrast_limits=contrast_limits,
    )


def build_volume_layer_config_from_preset(
    fused_volume: FusedSweepVolume,
    *,
    preset_name: str,
    name: str = "sweep_volume",
) -> VolumeLayerConfig:
    """Build a napari layer config from a named volume preset."""
    preset: VolumePreset = get_volume_preset(preset_name)
    return build_volume_layer_config(
        fused_volume,
        name=name,
        rendering=preset.rendering,
        colormap=preset.colormap,
        opacity=preset.opacity,
        blending=preset.blending,
    )


def launch_basic_volume_viewer(
    fused_volume: FusedSweepVolume,
    *,
    viewer_title: str = "UltraNeRF Sweep Volume",
    show_axes: bool = True,
    preset_name: str | None = None,
    layer_kwargs: dict[str, Any] | None = None,
):
    """Launch a basic 3D napari viewer for a fused sweep volume.

    Returns the created napari viewer. This function must only be called in an
    environment where a Qt backend is available.
    """
    try:
        import napari
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "napari is not installed. Install the visualization dependencies to launch the viewer."
        ) from exc

    config = (
        build_volume_layer_config_from_preset(fused_volume, preset_name=preset_name)
        if preset_name is not None
        else build_volume_layer_config(fused_volume)
    )
    kwargs = {
        "scale": config.scale,
        "translate": config.translate,
        "name": config.name,
        "rendering": config.rendering,
        "colormap": config.colormap,
        "opacity": config.opacity,
        "blending": config.blending,
        "contrast_limits": config.contrast_limits,
    }
    if layer_kwargs:
        kwargs.update(layer_kwargs)

    viewer = napari.Viewer(title=viewer_title, ndisplay=3)
    viewer.add_image(config.data, **kwargs)
    if show_axes:
        viewer.axes.visible = True
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "mm"
    return viewer
