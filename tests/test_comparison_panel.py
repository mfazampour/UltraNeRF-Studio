import numpy as np

from visualization.comparison_panel import extract_matched_image, format_comparison_metadata


def test_extract_matched_image_returns_2d_array():
    image = extract_matched_image({"matched_image": np.ones((4, 5), dtype=np.float32)})

    assert image.shape == (4, 5)


def test_format_comparison_metadata_summarizes_distances():
    metadata = format_comparison_metadata(
        {
            "matched_index": 12,
            "translation_distance_mm": 1.25,
            "rotation_distance_deg": 3.5,
        }
    )

    assert metadata == "Frame 12 | dT=1.25 mm | dR=3.50 deg"


def test_format_comparison_metadata_includes_matched_sweep_name_when_present():
    metadata = format_comparison_metadata(
        {
            "matched_sweep_name": "Sweep B",
            "matched_index": 7,
            "translation_distance_mm": 2.0,
            "rotation_distance_deg": 4.0,
        }
    )

    assert metadata == "Sweep B | Frame 7 | dT=2.00 mm | dR=4.00 deg"
