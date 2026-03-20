"""Rendered-output panel for checkpoint-backed visualization sessions."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_RENDER_MAP_KEY = "intensity_map"
PREFERRED_RENDER_MAP_KEYS = (
    "intensity_map",
    "attenuation_coeff",
    "reflection_coeff",
    "attenuation_total",
    "reflection_total",
    "scatterers_density",
    "scatterers_density_coeff",
    "scatter_amplitude",
    "confidence_maps",
    "b",
    "r",
    "r_amplified",
    "image",
    "rgb_map",
)


def _to_numpy(array_like: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    return np.asarray(array_like)


def _normalize_render_image_shape(image: Any) -> np.ndarray:
    """Convert renderer output arrays into a displayable 2D or RGB image."""
    image = _to_numpy(image)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported rendered image shape: {image.shape}")
    return image.astype(np.float32)


def get_available_render_map_keys(rendered_output: dict[str, Any] | None) -> list[str]:
    """Return supported intermediate image/map keys present in a render payload."""
    if not rendered_output:
        return []
    available: list[str] = []
    for key in PREFERRED_RENDER_MAP_KEYS:
        if key not in rendered_output:
            continue
        try:
            _normalize_render_image_shape(rendered_output[key])
        except Exception:
            continue
        available.append(key)
    return available


def resolve_render_map_key(rendered_output: dict[str, Any], map_key: str | None = None) -> str:
    """Resolve a requested render-map key against the current payload."""
    available = get_available_render_map_keys(rendered_output)
    if not available:
        raise KeyError("No supported render image key found in rendered_output")
    if map_key and map_key in available:
        return map_key
    if DEFAULT_RENDER_MAP_KEY in available:
        return DEFAULT_RENDER_MAP_KEY
    return available[0]


def extract_render_image(rendered_output: dict[str, Any], map_key: str | None = None) -> np.ndarray:
    """Extract a displayable image array from a renderer output payload."""
    if not rendered_output:
        raise ValueError("rendered_output is empty")
    resolved_key = resolve_render_map_key(rendered_output, map_key)
    return _normalize_render_image_shape(rendered_output[resolved_key])


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


def format_render_metadata(rendered_output: dict[str, Any] | None, map_key: str | None = None) -> str:
    """Create a short metadata summary for the current rendered output."""
    if not rendered_output:
        return "No render available"
    try:
        resolved_key = resolve_render_map_key(rendered_output, map_key)
        image = extract_render_image(rendered_output, resolved_key)
    except Exception:
        return "Render available"
    return (
        f"Map: {resolved_key} | Image shape: {tuple(int(v) for v in image.shape)} | "
        f"min={float(np.min(image)):.3g} max={float(np.max(image)):.3g}"
    )


class RenderOutputDockWidget:
    """Qt dock widget that displays the most recent NeRF-rendered image."""

    def __init__(self, ui_controller: Any):
        self.ui_controller = ui_controller
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QComboBox, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

        self._Qt = Qt
        self._QLabel = QLabel
        self._QComboBox = QComboBox
        self._QPushButton = QPushButton
        self._QVBoxLayout = QVBoxLayout
        self._QWidget = QWidget
        self._last_rendered_output: dict[str, Any] | None = None
        self._selected_map_key = DEFAULT_RENDER_MAP_KEY

        self.widget = QWidget()
        self.widget.setMinimumWidth(420)
        self.widget.setMinimumHeight(360)
        layout = QVBoxLayout(self.widget)

        self.title_label = QLabel("NeRF Render")
        self.status_label = QLabel("No render yet")
        self.map_selector = QComboBox()
        self.map_selector.addItem(DEFAULT_RENDER_MAP_KEY, DEFAULT_RENDER_MAP_KEY)
        self.map_selector.currentIndexChanged.connect(self._handle_map_selection_changed)
        self.metadata_label = QLabel("No render available")
        self.metadata_label.setWordWrap(True)
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 320)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.map_selector)
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

    def selected_map_key(self) -> str:
        return self._selected_map_key

    def set_render_output(self, rendered_output: dict[str, Any]) -> None:
        self._last_rendered_output = rendered_output
        available_keys = get_available_render_map_keys(rendered_output)
        self._populate_map_selector(available_keys)
        self._refresh_selected_render_map()

    def _handle_map_selection_changed(self) -> None:
        selected = self.map_selector.currentData()
        self._selected_map_key = str(selected or DEFAULT_RENDER_MAP_KEY)
        self._refresh_selected_render_map()

    def _populate_map_selector(self, available_keys: list[str]) -> None:
        selected_key = self._selected_map_key
        if selected_key not in available_keys:
            selected_key = resolve_render_map_key(self._last_rendered_output or {}, selected_key) if available_keys else DEFAULT_RENDER_MAP_KEY
        self.map_selector.blockSignals(True)
        self.map_selector.clear()
        for key in available_keys or [DEFAULT_RENDER_MAP_KEY]:
            self.map_selector.addItem(key, key)
        index = self.map_selector.findData(selected_key)
        if index >= 0:
            self.map_selector.setCurrentIndex(index)
        self.map_selector.blockSignals(False)
        self._selected_map_key = str(selected_key)

    def _refresh_selected_render_map(self) -> None:
        if not self._last_rendered_output:
            return
        image = extract_render_image(self._last_rendered_output, self._selected_map_key)
        self.set_metadata(format_render_metadata(self._last_rendered_output, self._selected_map_key))
        self.set_image(image)

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
