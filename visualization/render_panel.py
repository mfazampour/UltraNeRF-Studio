"""Rendered-output panel for checkpoint-backed visualization sessions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(array_like: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    return np.asarray(array_like)


def extract_render_image(rendered_output: dict[str, Any]) -> np.ndarray:
    """Extract a displayable image array from a renderer output payload."""
    if not rendered_output:
        raise ValueError("rendered_output is empty")

    for key in ("intensity_map", "image", "rgb_map"):
        if key in rendered_output:
            image = _to_numpy(rendered_output[key])
            break
    else:
        raise KeyError("No supported render image key found in rendered_output")

    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported rendered image shape: {image.shape}")
    return image.astype(np.float32)


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a render image into an 8-bit display buffer."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        finite = np.isfinite(array)
        if not np.any(finite):
            return np.zeros_like(array, dtype=np.uint8)
        valid = array[finite]
        if float(np.min(valid)) >= 0.0:
            array = np.log1p(array)
            valid = array[finite]
        min_value = float(np.percentile(valid, 1.0))
        max_value = float(np.percentile(valid, 99.5))
        if max_value <= min_value:
            min_value = float(np.min(valid))
            max_value = float(np.max(valid))
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.uint8)
        scaled = (array - min_value) / (max_value - min_value)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)

    if array.ndim == 3 and array.shape[-1] in (3, 4):
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        array = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
        return array

    raise ValueError(f"Unsupported image shape for display normalization: {array.shape}")


def format_render_metadata(rendered_output: dict[str, Any] | None) -> str:
    """Create a short metadata summary for the current rendered output."""
    if not rendered_output:
        return "No render available"
    try:
        image = extract_render_image(rendered_output)
    except Exception:
        return "Render available"
    return (
        f"Image shape: {tuple(int(v) for v in image.shape)} | "
        f"min={float(np.min(image)):.3g} max={float(np.max(image)):.3g}"
    )


class RenderOutputDockWidget:
    """Qt dock widget that displays the most recent NeRF-rendered image."""

    def __init__(self, ui_controller: Any):
        self.ui_controller = ui_controller
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        self._Qt = Qt
        self._QLabel = QLabel
        self._QPushButton = QPushButton
        self._QVBoxLayout = QVBoxLayout
        self._QWidget = QWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        self.title_label = QLabel("NeRF Render")
        self.status_label = QLabel("No render yet")
        self.metadata_label = QLabel("No render available")
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(256, 256)
        self.image_label.setScaledContents(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.metadata_label)
        layout.addWidget(self.image_label, stretch=1)

        self.render_button = QPushButton("Render Now")
        self.render_button.clicked.connect(self._handle_render_now)
        layout.addWidget(self.render_button)

    def _handle_render_now(self) -> None:
        self.set_status("Rendering...")
        try:
            self.ui_controller.render_now()
        except Exception as exc:
            self.set_status(f"Render failed: {exc}")

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text))

    def set_metadata(self, text: str) -> None:
        self.metadata_label.setText(str(text))

    def set_image(self, image: np.ndarray) -> None:
        display_buffer = normalize_image_for_display(image)
        from PyQt5.QtGui import QImage, QPixmap

        if display_buffer.ndim == 2:
            height, width = display_buffer.shape
            bytes_per_line = width
            qimage = QImage(
                display_buffer.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8,
            ).copy()
        elif display_buffer.ndim == 3 and display_buffer.shape[-1] == 3:
            height, width, _ = display_buffer.shape
            bytes_per_line = width * 3
            qimage = QImage(
                display_buffer.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            ).copy()
        elif display_buffer.ndim == 3 and display_buffer.shape[-1] == 4:
            height, width, _ = display_buffer.shape
            bytes_per_line = width * 4
            qimage = QImage(
                display_buffer.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGBA8888,
            ).copy()
        else:
            raise ValueError(f"Unsupported display buffer shape: {display_buffer.shape}")
        self.image_label.setPixmap(QPixmap.fromImage(qimage))


def create_render_panel(ui_controller: Any) -> RenderOutputDockWidget:
    """Create the Qt render panel for a visualization UI controller."""
    return RenderOutputDockWidget(ui_controller)
