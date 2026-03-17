import numpy as np

from visualization.render_panel import extract_render_image, format_render_metadata, normalize_image_for_display


def test_extract_render_image_prefers_intensity_map_and_squeezes_batch_dims():
    image = extract_render_image({"intensity_map": np.ones((1, 1, 4, 5), dtype=np.float32)})

    assert image.shape == (4, 5)


def test_extract_render_image_moves_channel_first_rgb_to_channel_last():
    image = extract_render_image({"rgb_map": np.ones((3, 4, 5), dtype=np.float32)})

    assert image.shape == (4, 5, 3)


def test_normalize_image_for_display_returns_uint8():
    display = normalize_image_for_display(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))

    assert display.dtype == np.uint8
    assert display.shape == (2, 2)
    assert display.min() == 0
    assert display.max() == 255


def test_format_render_metadata_reports_image_shape():
    metadata = format_render_metadata({"intensity_map": np.ones((1, 1, 4, 5), dtype=np.float32)})

    assert metadata == "Image shape: (4, 5) | min=1 max=1"


def test_normalize_image_for_display_handles_sparse_nonnegative_signal():
    image = np.zeros((8, 8), dtype=np.float32)
    image[3, 4] = 1e-6
    image[6, 2] = 1.0

    display = normalize_image_for_display(image)

    assert display.dtype == np.uint8
    assert display.max() == 255
    assert display[6, 2] == 255
