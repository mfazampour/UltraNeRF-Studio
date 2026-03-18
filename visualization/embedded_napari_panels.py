"""Embedded napari image-view panels for multi-view visualization layouts."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from visualization.comparison_panel import normalize_recorded_image_for_display
from visualization.render_panel import normalize_image_for_display


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
        from PyQt5.QtWidgets import QPushButton

        super().__init__(
            title="NeRF Render",
            empty_status="No render yet",
            image_normalizer=normalize_image_for_display,
        )
        self.ui_controller = ui_controller
        self.render_button = QPushButton("Render Now")
        self.render_button.clicked.connect(self._handle_render_now)
        self.widget.layout().addWidget(self.render_button)

    def _handle_render_now(self) -> None:
        self.set_status("Rendering...")
        try:
            self.ui_controller.render_now()
        except Exception as exc:
            self.set_status(f"Render failed: {exc}")


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
