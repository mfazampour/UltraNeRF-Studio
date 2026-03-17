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
