"""Probe manipulation controls for the napari visualization app."""

from __future__ import annotations

import numpy as np


class ProbeControlsDockWidget:
    """Qt dock widget for editing the virtual probe pose."""

    def __init__(self, ui_controller, *, num_frames: int):
        from PyQt5.QtWidgets import (
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.ui_controller = ui_controller
        self._updating_fields = False
        self.widget = QWidget()

        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Probe Controls"))

        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        self.x_spin = self._make_double_spinbox(QDoubleSpinBox)
        self.y_spin = self._make_double_spinbox(QDoubleSpinBox)
        self.z_spin = self._make_double_spinbox(QDoubleSpinBox)
        self.yaw_spin = self._make_double_spinbox(QDoubleSpinBox, minimum=-360.0, maximum=360.0)
        self.pitch_spin = self._make_double_spinbox(QDoubleSpinBox, minimum=-360.0, maximum=360.0)
        self.roll_spin = self._make_double_spinbox(QDoubleSpinBox, minimum=-360.0, maximum=360.0)
        self.recorded_index_spin = QSpinBox()
        self.recorded_index_spin.setRange(0, max(num_frames - 1, 0))

        form_layout.addRow("X (mm)", self.x_spin)
        form_layout.addRow("Y (mm)", self.y_spin)
        form_layout.addRow("Z (mm)", self.z_spin)
        form_layout.addRow("Yaw (deg)", self.yaw_spin)
        form_layout.addRow("Pitch (deg)", self.pitch_spin)
        form_layout.addRow("Roll (deg)", self.roll_spin)
        form_layout.addRow("Frame Index", self.recorded_index_spin)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.reset_button = QPushButton("Reset To Frame")
        self.snap_button = QPushButton("Snap To Nearest")
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.snap_button)

        for spin in (self.x_spin, self.y_spin, self.z_spin, self.yaw_spin, self.pitch_spin, self.roll_spin):
            spin.valueChanged.connect(self._handle_pose_change)
        self.reset_button.clicked.connect(self._handle_reset_to_frame)
        self.snap_button.clicked.connect(self._handle_snap_to_nearest)

    @staticmethod
    def _make_double_spinbox(spinbox_class, *, minimum: float = -1000.0, maximum: float = 1000.0):
        spin = spinbox_class()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        return spin

    def set_num_frames(self, num_frames: int) -> None:
        self.recorded_index_spin.setRange(0, max(int(num_frames) - 1, 0))
        if self.recorded_index_spin.value() > max(int(num_frames) - 1, 0):
            self.recorded_index_spin.setValue(max(int(num_frames) - 1, 0))

    def set_pose_values(
        self,
        *,
        origin_mm: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        recorded_index: int,
    ) -> None:
        self._updating_fields = True
        try:
            self.x_spin.setValue(float(origin_mm[0]))
            self.y_spin.setValue(float(origin_mm[1]))
            self.z_spin.setValue(float(origin_mm[2]))
            self.yaw_spin.setValue(float(yaw_deg))
            self.pitch_spin.setValue(float(pitch_deg))
            self.roll_spin.setValue(float(roll_deg))
            self.recorded_index_spin.setValue(int(recorded_index))
        finally:
            self._updating_fields = False

    def _handle_pose_change(self, _value: float) -> None:
        if self._updating_fields:
            return
        origin_mm = np.array(
            [
                self.x_spin.value(),
                self.y_spin.value(),
                self.z_spin.value(),
            ],
            dtype=np.float32,
        )
        self.ui_controller.set_probe_pose_from_components(
            origin_mm=origin_mm,
            yaw_deg=self.yaw_spin.value(),
            pitch_deg=self.pitch_spin.value(),
            roll_deg=self.roll_spin.value(),
        )

    def _handle_reset_to_frame(self) -> None:
        self.ui_controller.set_probe_to_recorded_pose(self.recorded_index_spin.value())

    def _handle_snap_to_nearest(self) -> None:
        self.ui_controller.snap_probe_to_nearest_recorded_pose()


def create_probe_controls(ui_controller, *, num_frames: int) -> ProbeControlsDockWidget:
    """Create the Qt probe-controls dock widget."""
    return ProbeControlsDockWidget(ui_controller, num_frames=num_frames)
