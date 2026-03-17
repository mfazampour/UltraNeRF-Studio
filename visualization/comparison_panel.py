"""Recorded-frame comparison panel for the napari visualization app."""

from __future__ import annotations

from typing import Any

import numpy as np

def extract_matched_image(comparison_payload: dict[str, Any]) -> np.ndarray:
    """Extract the nearest recorded frame image from a comparison payload."""
    if "matched_image" not in comparison_payload:
        raise KeyError("comparison_payload is missing matched_image")
    return np.asarray(comparison_payload["matched_image"], dtype=np.float32)


def normalize_recorded_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a recorded ultrasound frame into an 8-bit display buffer.

    Recorded frames should preserve their original grayscale appearance. If the
    image is already normalized to ``[0, 1]``, map that directly to ``[0, 255]``
    without the aggressive contrast enhancement used for sparse NeRF renders.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Comparison panel expects a 2D grayscale image, got {array.shape}")

    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.uint8)

    valid = array[finite]
    min_value = float(np.min(valid))
    max_value = float(np.max(valid))

    if min_value >= 0.0 and max_value <= 1.0:
        scaled = np.clip(array, 0.0, 1.0) * 255.0
        return np.round(scaled).astype(np.uint8)

    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)

    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def format_comparison_metadata(comparison_payload: dict[str, Any]) -> str:
    """Create a short textual summary of the current comparison match."""
    prefix = ""
    if "matched_sweep_name" in comparison_payload:
        prefix = f"{comparison_payload['matched_sweep_name']} | "
    elif "matched_sweep_id" in comparison_payload:
        prefix = f"{comparison_payload['matched_sweep_id']} | "
    return (
        f"{prefix}Frame {int(comparison_payload['matched_index'])} | "
        f"dT={float(comparison_payload['translation_distance_mm']):.2f} mm | "
        f"dR={float(comparison_payload['rotation_distance_deg']):.2f} deg"
    )


class ComparisonDockWidget:
    """Qt dock widget that displays the nearest recorded frame."""

    def __init__(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

        self.widget = QWidget()
        self.widget.setMinimumWidth(420)
        self.widget.setMinimumHeight(360)
        layout = QVBoxLayout(self.widget)
        self.title_label = QLabel("Nearest Recorded Frame")
        self.status_label = QLabel("No comparison available")
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 320)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.metadata_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text))

    def set_metadata(self, text: str) -> None:
        self.metadata_label.setText(str(text))

    def set_image(self, image: np.ndarray) -> None:
        display_buffer = normalize_recorded_image_for_display(image)
        from PyQt5.QtGui import QImage, QPixmap

        if display_buffer.ndim != 2:
            raise ValueError(f"Comparison panel expects a 2D grayscale image, got {display_buffer.shape}")
        height, width = display_buffer.shape
        qimage = QImage(
            display_buffer.data,
            width,
            height,
            width,
            QImage.Format_Grayscale8,
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(qimage))


def create_comparison_panel() -> ComparisonDockWidget:
    """Create the Qt dock widget for nearest-frame comparison."""
    return ComparisonDockWidget()
