import numpy as np

from ultranerf.visualization.sweep_volume import FusedSweepVolume
from ultranerf.visualization.volume_viewer import build_volume_layer_config, build_volume_layer_config_from_preset


def make_fused_volume() -> FusedSweepVolume:
    scalar = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    weights = np.ones((2, 2, 2), dtype=np.float32)
    return FusedSweepVolume(
        scalar_volume=scalar,
        weight_volume=weights,
        origin_mm=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        spacing_mm=np.array([0.5, 1.0, 2.0], dtype=np.float32),
        bounds_min_mm=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bounds_max_mm=np.array([1.5, 3.0, 5.0], dtype=np.float32),
    )


def test_build_volume_layer_config_maps_volume_metadata_to_napari_config():
    fused = make_fused_volume()

    config = build_volume_layer_config(fused, name="test_volume", rendering="attenuated_mip", colormap="inferno", opacity=0.7)

    assert config.data.shape == (2, 2, 2)
    assert config.scale == (0.5, 1.0, 2.0)
    assert config.translate == (1.0, 2.0, 3.0)
    assert config.name == "test_volume"
    assert config.rendering == "attenuated_mip"
    assert config.colormap == "inferno"
    assert config.opacity == 0.7
    assert config.axis_labels == ("x_mm", "y_mm", "z_mm")
    assert config.contrast_limits == (0.0, 7.0)


def test_build_volume_layer_config_from_preset_applies_named_preset():
    fused = make_fused_volume()

    config = build_volume_layer_config_from_preset(fused, preset_name="high_contrast", name="preset_volume")

    assert config.name == "preset_volume"
    assert config.rendering == "mip"
    assert config.colormap == "inferno"
    assert config.blending == "additive"
