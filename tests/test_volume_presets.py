from visualization.volume_presets import PRESETS, get_volume_preset


def test_get_volume_preset_returns_expected_named_preset():
    preset = get_volume_preset("soft_tissue")

    assert preset.name == "soft_tissue"
    assert preset.rendering == "attenuated_mip"
    assert preset.colormap == "gray"


def test_volume_presets_define_multiple_viewing_modes():
    assert set(PRESETS) >= {"soft_tissue", "high_contrast", "sparse_signal"}
    assert PRESETS["high_contrast"].blending == "additive"
