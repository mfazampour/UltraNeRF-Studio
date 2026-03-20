"""Shared busy-indicator utilities for Qt-backed visualization workflows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _resolve_qt_parent(viewer: Any) -> Any | None:
    window = getattr(viewer, "window", None)
    if window is None:
        return None
    for attr in ("_qt_window", "_qt_viewer", "qt_viewer"):
        widget = getattr(window, attr, None)
        if widget is not None:
            return widget
    return None


@contextmanager
def ui_busy_feedback(viewer: Any, message: str) -> Iterator[None]:
    """Show a best-effort Qt busy indicator while synchronous work is running.

    The helper is intentionally centralized so long-running UI actions can share
    one behavior instead of open-coding dialogs and wait cursors in each slot.
    In headless or non-Qt contexts it degrades to a no-op.
    """

    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QProgressDialog
    except ModuleNotFoundError:
        yield
        return

    app = QApplication.instance()
    if app is None:
        yield
        return

    parent = _resolve_qt_parent(viewer)
    progress = QProgressDialog(str(message), "", 0, 0, parent)
    progress.setWindowTitle("Working")
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)
    progress.show()

    old_status = getattr(viewer, "status", None)
    try:
        if hasattr(viewer, "status"):
            viewer.status = str(message)
        app.setOverrideCursor(Qt.WaitCursor)
        app.processEvents()
        yield
    finally:
        progress.close()
        if hasattr(viewer, "status"):
            viewer.status = old_status if old_status is not None else ""
        try:
            app.restoreOverrideCursor()
        except Exception:
            pass
        app.processEvents()
