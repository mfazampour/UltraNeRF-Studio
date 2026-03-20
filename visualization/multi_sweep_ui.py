"""State management and lightweight UI helpers for multi-sweep scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visualization.multi_sweep import MultiSweepScene
from visualization.multi_sweep_volume import MultiSweepFusionResult, SweepVolumeOverlay, build_sweep_overlay, fuse_multi_sweep_scene
from visualization.sweep_volume import FusionDevice, FusionReductionMode


@dataclass(frozen=True)
class MultiSweepViewerState:
    """UI-facing state for a multi-sweep visualization session."""

    active_sweep_id: str
    enabled_sweep_ids: tuple[str, ...]
    visible_sweep_ids: tuple[str, ...]
    comparison_policy: str
    show_aggregate_volume: bool


class MultiSweepSceneController:
    """Manage multi-sweep viewer state independently of the GUI toolkit."""

    def __init__(
        self,
        scene: MultiSweepScene,
        *,
        spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        pixel_stride: tuple[int, int] = (2, 2),
        fusion_device: FusionDevice = "auto",
        reduction_mode: FusionReductionMode = "max",
    ) -> None:
        self.scene = scene
        self.spacing_mm = tuple(float(v) for v in spacing_mm)
        self.pixel_stride = tuple(int(v) for v in pixel_stride)
        self.fusion_device = str(fusion_device)
        self.reduction_mode = str(reduction_mode)
        self._aggregate_fusion_cache: MultiSweepFusionResult | None = None
        self._per_sweep_volume_cache: dict[str, SweepVolumeOverlay] = {}
        self._trajectory_overlay_cache: dict[str, SweepVolumeOverlay] = {}
        self.state = MultiSweepViewerState(
            active_sweep_id=scene.active_sweep_id,
            enabled_sweep_ids=tuple(sweep.sweep_id for sweep in (scene.enabled_sweeps or scene.sweeps)),
            visible_sweep_ids=(scene.active_sweep_id,),
            comparison_policy=scene.comparison_policy,
            show_aggregate_volume=True,
        )

    def set_active_sweep(self, sweep_id: str) -> MultiSweepViewerState:
        self.scene.get_sweep(sweep_id)
        self.state = MultiSweepViewerState(
            active_sweep_id=sweep_id,
            enabled_sweep_ids=self.state.enabled_sweep_ids,
            visible_sweep_ids=self.state.visible_sweep_ids,
            comparison_policy=self.state.comparison_policy,
            show_aggregate_volume=self.state.show_aggregate_volume,
        )
        return self.state

    def set_enabled_sweeps(self, sweep_ids: tuple[str, ...]) -> MultiSweepViewerState:
        resolved_ids = tuple(sweep.sweep_id for sweep in self.scene.sweeps if sweep.sweep_id in set(sweep_ids))
        if not resolved_ids:
            raise ValueError("At least one sweep must remain enabled")
        active_id = self.state.active_sweep_id if self.state.active_sweep_id in resolved_ids else resolved_ids[0]
        self.state = MultiSweepViewerState(
            active_sweep_id=active_id,
            enabled_sweep_ids=resolved_ids,
            visible_sweep_ids=self.state.visible_sweep_ids,
            comparison_policy=self.state.comparison_policy,
            show_aggregate_volume=self.state.show_aggregate_volume,
        )
        return self.state

    def set_visible_sweeps(self, sweep_ids: tuple[str, ...]) -> MultiSweepViewerState:
        resolved_ids = tuple(sweep.sweep_id for sweep in self.scene.sweeps if sweep.sweep_id in set(sweep_ids))
        if not resolved_ids:
            raise ValueError("At least one sweep must remain visible")
        self.state = MultiSweepViewerState(
            active_sweep_id=self.state.active_sweep_id,
            enabled_sweep_ids=self.state.enabled_sweep_ids,
            visible_sweep_ids=resolved_ids,
            comparison_policy=self.state.comparison_policy,
            show_aggregate_volume=self.state.show_aggregate_volume,
        )
        return self.state

    def set_comparison_policy(self, policy: str) -> MultiSweepViewerState:
        self.state = MultiSweepViewerState(
            active_sweep_id=self.state.active_sweep_id,
            enabled_sweep_ids=self.state.enabled_sweep_ids,
            visible_sweep_ids=self.state.visible_sweep_ids,
            comparison_policy=str(policy),
            show_aggregate_volume=self.state.show_aggregate_volume,
        )
        return self.state

    def set_show_aggregate_volume(self, show_aggregate_volume: bool) -> MultiSweepViewerState:
        self.state = MultiSweepViewerState(
            active_sweep_id=self.state.active_sweep_id,
            enabled_sweep_ids=self.state.enabled_sweep_ids,
            visible_sweep_ids=self.state.visible_sweep_ids,
            comparison_policy=self.state.comparison_policy,
            show_aggregate_volume=bool(show_aggregate_volume),
        )
        return self.state

    def build_fusion_result(self) -> MultiSweepFusionResult:
        aggregate = self._get_aggregate_fusion()
        if self.state.show_aggregate_volume:
            overlays = tuple(self._get_trajectory_overlay(sweep.sweep_id) for sweep in self.scene.sweeps)
            return MultiSweepFusionResult(
                aggregate_volume=aggregate.aggregate_volume,
                sweep_overlays=overlays,
                enabled_sweep_ids=self.state.enabled_sweep_ids,
                bounds_min_mm=aggregate.bounds_min_mm,
                bounds_max_mm=aggregate.bounds_max_mm,
            )

        overlays = tuple(self._get_per_sweep_overlay(sweep_id) for sweep_id in self.state.visible_sweep_ids)
        return MultiSweepFusionResult(
            aggregate_volume=aggregate.aggregate_volume,
            sweep_overlays=overlays,
            enabled_sweep_ids=self.state.visible_sweep_ids,
            bounds_min_mm=aggregate.bounds_min_mm,
            bounds_max_mm=aggregate.bounds_max_mm,
        )

    def _get_aggregate_fusion(self) -> MultiSweepFusionResult:
        if self._aggregate_fusion_cache is None:
            self._aggregate_fusion_cache = fuse_multi_sweep_scene(
                self.scene,
                spacing_mm=self.spacing_mm,
                pixel_stride=self.pixel_stride,
                enabled_sweep_ids=tuple(sweep.sweep_id for sweep in self.scene.sweeps),
                fusion_device=self.fusion_device,
                reduction_mode=self.reduction_mode,
                include_per_sweep_volumes=False,
            )
        return self._aggregate_fusion_cache

    def _get_trajectory_overlay(self, sweep_id: str) -> SweepVolumeOverlay:
        cached = self._trajectory_overlay_cache.get(sweep_id)
        if cached is not None:
            return cached
        sweep = self.scene.get_sweep(sweep_id)
        overlay = build_sweep_overlay(
            sweep,
            spacing_mm=self.spacing_mm,
            pixel_stride=self.pixel_stride,
            fusion_device=self.fusion_device,
            reduction_mode=self.reduction_mode,
            include_volume=False,
        )
        self._trajectory_overlay_cache[sweep_id] = overlay
        return overlay

    def _get_per_sweep_overlay(self, sweep_id: str) -> SweepVolumeOverlay:
        cached = self._per_sweep_volume_cache.get(sweep_id)
        if cached is not None:
            return cached
        sweep = self.scene.get_sweep(sweep_id)
        overlay = build_sweep_overlay(
            sweep,
            spacing_mm=self.spacing_mm,
            pixel_stride=self.pixel_stride,
            fusion_device=self.fusion_device,
            reduction_mode=self.reduction_mode,
            include_volume=True,
        )
        self._per_sweep_volume_cache[sweep_id] = overlay
        return overlay


class MultiSweepControlsDockWidget:
    """Qt dock widget for multi-sweep scene settings."""

    def __init__(self, controller: MultiSweepSceneController, *, on_state_changed: Any | None = None):
        from PyQt5.QtWidgets import (
            QCheckBox,
            QComboBox,
            QFormLayout,
            QLabel,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.on_state_changed = on_state_changed
        self._updating = False
        outer_widget = QWidget()
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        self.widget = outer_widget
        content_widget = QWidget()
        content_widget.setMinimumWidth(280)
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget)
        layout.addWidget(QLabel("Multi-Sweep Controls"))
        layout.addWidget(QLabel("Enabled sweeps affect per-sweep mode and comparison."))

        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        self.active_sweep_combo = QComboBox()
        for sweep in controller.scene.sweeps:
            self.active_sweep_combo.addItem(sweep.display_name or sweep.sweep_id, sweep.sweep_id)
        form_layout.addRow("Active Sweep", self.active_sweep_combo)

        self.comparison_policy_combo = QComboBox()
        for label, value in (
            ("All Enabled Sweeps", "all_enabled"),
            ("Active Sweep Only", "active_only"),
        ):
            self.comparison_policy_combo.addItem(label, value)
        form_layout.addRow("Comparison", self.comparison_policy_combo)

        self.aggregate_checkbox = QCheckBox("Show Aggregate Volume")
        layout.addWidget(self.aggregate_checkbox)

        self.active_sweep_combo.currentIndexChanged.connect(self._handle_active_sweep_change)
        self.comparison_policy_combo.currentIndexChanged.connect(self._handle_comparison_policy_change)
        self.aggregate_checkbox.stateChanged.connect(self._handle_aggregate_change)

        self.refresh()

    def refresh(self) -> None:
        state = self.controller.state
        self._updating = True
        try:
            active_index = self.active_sweep_combo.findData(state.active_sweep_id)
            if active_index >= 0:
                self.active_sweep_combo.setCurrentIndex(active_index)
            policy_index = self.comparison_policy_combo.findData(state.comparison_policy)
            if policy_index >= 0:
                self.comparison_policy_combo.setCurrentIndex(policy_index)
            self.aggregate_checkbox.setChecked(state.show_aggregate_volume)
        finally:
            self._updating = False

    def _handle_active_sweep_change(self, _index: int) -> None:
        if self._updating:
            return
        self.controller.set_active_sweep(str(self.active_sweep_combo.currentData()))
        if self.on_state_changed is not None:
            self.on_state_changed(self.controller.state)

    def _handle_comparison_policy_change(self, _index: int) -> None:
        if self._updating:
            return
        self.controller.set_comparison_policy(str(self.comparison_policy_combo.currentData()))
        if self.on_state_changed is not None:
            self.on_state_changed(self.controller.state)

    def _handle_aggregate_change(self, state: int) -> None:
        if self._updating:
            return
        self.controller.set_show_aggregate_volume(bool(state))
        if self.on_state_changed is not None:
            self.on_state_changed(self.controller.state)

class SweepSelectionDockWidget:
    """Qt dock widget for selecting which sweeps participate in per-sweep mode."""

    def __init__(self, controller: MultiSweepSceneController, *, on_apply: Any | None = None):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (
            QAbstractItemView,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        self.controller = controller
        self.on_apply = on_apply
        self._updating = False
        outer_widget = QWidget()
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)
        self.widget = outer_widget

        content_widget = QWidget()
        content_widget.setMinimumWidth(280)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.addWidget(QLabel("Sweep Selection"))
        layout.addWidget(QLabel("Enable = comparison/search. Visible = per-sweep display."))

        self.selection_summary_label = QLabel("")
        layout.addWidget(self.selection_summary_label)

        self.selection_table = QTableWidget(len(controller.scene.sweeps), 3)
        self.selection_table.setHorizontalHeaderLabels(["Sweep", "Enabled", "Visible"])
        self.selection_table.verticalHeader().setVisible(False)
        self.selection_table.setMinimumHeight(240)
        self.selection_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selection_table.horizontalHeader().setStretchLastSection(False)
        self.selection_table.horizontalHeader().setSectionResizeMode(0, self.selection_table.horizontalHeader().Stretch)
        for row, sweep in enumerate(controller.scene.sweeps):
            label_item = QTableWidgetItem(sweep.display_name or sweep.sweep_id)
            label_item.setData(1, sweep.sweep_id)
            self.selection_table.setItem(row, 0, label_item)
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            self.selection_table.setItem(row, 1, enabled_item)
            visible_item = QTableWidgetItem()
            visible_item.setFlags(visible_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            self.selection_table.setItem(row, 2, visible_item)
        layout.addWidget(self.selection_table)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)
        self.apply_button = QPushButton("Apply Selection")
        self.reset_button = QPushButton("Reset")
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.reset_button)

        self.selection_table.itemChanged.connect(self._handle_selection_changed)
        self.apply_button.clicked.connect(self._handle_apply)
        self.reset_button.clicked.connect(self.refresh)

        self.refresh()

    def refresh(self) -> None:
        enabled_ids = set(self.controller.state.enabled_sweep_ids)
        visible_ids = set(self.controller.state.visible_sweep_ids)
        self._updating = True
        try:
            for row in range(self.selection_table.rowCount()):
                sweep_id = str(self.selection_table.item(row, 0).data(1))
                self.selection_table.item(row, 1).setCheckState(2 if sweep_id in enabled_ids else 0)
                self.selection_table.item(row, 2).setCheckState(2 if sweep_id in visible_ids else 0)
            self._update_summary_label(pending=False)
        finally:
            self._updating = False

    def _collect_checked_ids(self, column: int) -> tuple[str, ...]:
        checked = []
        for row in range(self.selection_table.rowCount()):
            item = self.selection_table.item(row, column)
            if item.checkState() == 2:
                checked.append(str(self.selection_table.item(row, 0).data(1)))
        return tuple(checked)

    def _update_summary_label(self, *, pending: bool) -> None:
        enabled = self._collect_checked_ids(1)
        visible = self._collect_checked_ids(2)
        prefix = "Pending" if pending else "Active"
        self.selection_summary_label.setText(
            f"{prefix} enabled: {len(enabled)} | visible: {len(visible)}"
        )

    def _handle_selection_changed(self, _item: Any) -> None:
        if self._updating:
            return
        self._update_summary_label(pending=True)

    def _handle_apply(self) -> None:
        enabled = self._collect_checked_ids(1)
        visible = self._collect_checked_ids(2)
        self.controller.set_enabled_sweeps(enabled)
        self.controller.set_visible_sweeps(visible)
        self._update_summary_label(pending=False)
        if self.on_apply is not None:
            self.on_apply(self.controller.state)


def create_multi_sweep_controls(
    controller: MultiSweepSceneController,
    *,
    on_state_changed: Any | None = None,
) -> MultiSweepControlsDockWidget:
    """Create the Qt dock widget for multi-sweep controls."""
    return MultiSweepControlsDockWidget(controller, on_state_changed=on_state_changed)


def create_sweep_selection_controls(
    controller: MultiSweepSceneController,
    *,
    on_apply: Any | None = None,
) -> SweepSelectionDockWidget:
    """Create the Qt dock widget for sweep selection."""
    return SweepSelectionDockWidget(controller, on_apply=on_apply)
