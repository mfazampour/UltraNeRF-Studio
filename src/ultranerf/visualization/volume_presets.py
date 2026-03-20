"""Transfer-function style presets for sweep volume visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VolumePreset:
    """Viewer preset for scalar sweep volumes."""

    name: str
    rendering: str
    colormap: str
    opacity: float
    blending: str


PRESETS: dict[str, VolumePreset] = {
    "soft_tissue": VolumePreset(
        name="soft_tissue",
        rendering="attenuated_mip",
        colormap="gray",
        opacity=0.35,
        blending="translucent",
    ),
    "high_contrast": VolumePreset(
        name="high_contrast",
        rendering="mip",
        colormap="inferno",
        opacity=0.55,
        blending="additive",
    ),
    "sparse_signal": VolumePreset(
        name="sparse_signal",
        rendering="iso",
        colormap="magenta",
        opacity=0.7,
        blending="translucent",
    ),
}


def get_volume_preset(name: str) -> VolumePreset:
    """Return a named visualization preset."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown volume preset: {name}") from exc
