"""Embedded napari image-view panels for multi-view visualization layouts."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from visualization.comparison_panel import normalize_recorded_image_for_display
from visualization.render_panel import (
    DEFAULT_RENDER_MAP_KEY,
    extract_render_image,
    format_render_metadata,
    get_available_render_map_keys,
    normalize_image_for_display,
    resolve_render_map_key,
)


def _hide_napari_side_docks(viewer: Any) -> None:
    """Hide napari's built-in layer/control docks for embedded viewers."""
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        qt_viewer = getattr(viewer.window, "qt_viewer", None)
    if qt_viewer is None:
        return
    for name in ("dockLayerList", "_dockLayerList", "dockLayerControls", "_dockLayerControls"):
        dock = getattr(qt_viewer, name, None)
        if dock is not None:
            try:
                dock.hide()
            except Exception:
                pass


class EmbeddedNapariImagePanel:
    """Small QWidget wrapper around an embedded napari 2D viewer."""

    def __init__(
        self,
        *,
        title: str,
        empty_status: str,
        image_normalizer: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        import napari
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

        self._image_normalizer = image_normalizer
        self.viewer = napari.Viewer(title=title, ndisplay=2, show=False)
        _hide_napari_side_docks(self.viewer)

        self.widget = QWidget()
        self.widget.setMinimumWidth(440)
        self.widget.setMinimumHeight(360)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.status_label = QLabel(empty_status)
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)

        qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        if qt_viewer is None:
            qt_viewer = getattr(self.viewer.window, "qt_viewer", None)
        if qt_viewer is None:
            raise RuntimeError("Embedded napari panel requires a Qt viewer widget")
        qt_viewer.setMinimumSize(420, 300)
        qt_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.metadata_label)
        layout.addWidget(qt_viewer, stretch=1)

        self._Qt = Qt
        self._image_layer = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text))

    def set_metadata(self, text: str) -> None:
        self.metadata_label.setText(str(text))

    def set_image(self, image: np.ndarray, *, scale_mm: tuple[float, float] | None = None) -> None:
        display_buffer = self._image_normalizer(np.asarray(image))
        rgb = bool(display_buffer.ndim == 3 and display_buffer.shape[-1] in (3, 4))
        if self._image_layer is None:
            kwargs = {"name": "image"}
            if not rgb:
                kwargs["colormap"] = "gray"
            if scale_mm is not None:
                kwargs["scale"] = tuple(float(v) for v in scale_mm)
            self._image_layer = self.viewer.add_image(display_buffer, rgb=rgb, **kwargs)
        else:
            self._image_layer.data = display_buffer
            if hasattr(self._image_layer, "rgb"):
                self._image_layer.rgb = rgb
            if scale_mm is not None and hasattr(self._image_layer, "scale"):
                self._image_layer.scale = tuple(float(v) for v in scale_mm)
        try:
            self.viewer.reset_view()
        except Exception:
            pass


class EmbeddedNapariRenderPanel(EmbeddedNapariImagePanel):
    """Embedded render panel with a manual Render button."""

    def __init__(self, ui_controller: Any) -> None:
        from PyQt5.QtWidgets import QComboBox, QPushButton

        super().__init__(
            title="NeRF Render",
            empty_status="No render yet",
            image_normalizer=normalize_image_for_display,
        )
        self.ui_controller = ui_controller
        self._last_rendered_output: dict[str, Any] | None = None
        self._last_scale_mm: tuple[float, float] | None = None
        self._selected_map_key = DEFAULT_RENDER_MAP_KEY
        self.map_selector = QComboBox()
        self.map_selector.addItem(DEFAULT_RENDER_MAP_KEY, DEFAULT_RENDER_MAP_KEY)
        self.map_selector.currentIndexChanged.connect(self._handle_map_selection_changed)
        self.widget.layout().addWidget(self.map_selector)
        self.render_button = QPushButton("Render Now")
        self.render_button.clicked.connect(self._handle_render_now)
        self.widget.layout().addWidget(self.render_button)

    def _handle_render_now(self) -> None:
        self.set_status("Rendering...")
        try:
            self.ui_controller.render_now()
        except Exception as exc:
            self.set_status(f"Render failed: {exc}")

    def set_render_output(
        self,
        rendered_output: dict[str, Any],
        *,
        scale_mm: tuple[float, float] | None = None,
    ) -> None:
        self._last_rendered_output = rendered_output
        self._last_scale_mm = tuple(scale_mm) if scale_mm is not None else None
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
            selected_key = (
                resolve_render_map_key(self._last_rendered_output or {}, selected_key)
                if available_keys
                else DEFAULT_RENDER_MAP_KEY
            )
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
        self.set_image(image, scale_mm=self._last_scale_mm)


def create_embedded_comparison_panel() -> EmbeddedNapariImagePanel:
    """Create an embedded 2D napari viewer for nearest-frame comparison."""
    return EmbeddedNapariImagePanel(
        title="Nearest Recorded Frame",
        empty_status="No comparison available",
        image_normalizer=normalize_recorded_image_for_display,
    )


def create_embedded_render_panel(ui_controller: Any) -> EmbeddedNapariRenderPanel:
    """Create an embedded 2D napari viewer for NeRF rendering."""
    return EmbeddedNapariRenderPanel(ui_controller)
