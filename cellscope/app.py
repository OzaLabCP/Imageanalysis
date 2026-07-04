"""CellScope application shell: window, bottom tab bar, overlays, navigation."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from cellscope import __app_name__, config, theme
from cellscope.data import CephlaLoader, FolderLoader, MockLoader, NoImagesFoundError
from cellscope.state import AppState
from cellscope.views.cells_view import CellsView
from cellscope.views.results_view import ResultsView
from cellscope.views.viewer_view import ViewerView
from cellscope.views.wells_view import WellsView
from cellscope.widgets.controls import make_button
from cellscope.widgets.sheet import BottomSheet, Toast
from cellscope.widgets.tab_bar import TabBar
from cellscope.widgets.worker import run_async

WELLS, VIEWER, CELLS, RESULTS = range(4)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(__app_name__)
        self.resize(480, 860)
        self.setMinimumSize(380, 600)

        self.state = AppState()

        # Layered content + bottom tab bar.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._wells = WellsView(self.state, self)
        self._viewer = ViewerView(self.state, self)
        self._cells = CellsView(self.state, self)
        self._results = ResultsView(self.state, self)
        for view in (self._wells, self._viewer, self._cells, self._results):
            self._stack.addWidget(view)

        self._tabs = TabBar([
            ("Wells", "wells"),
            ("Viewer", "viewer"),
            ("Cells", "cells"),
            ("Results", "results"),
        ])
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs)

        # Overlays (children of the window, drawn above everything).
        self._sheet = BottomSheet(self)
        self._toast = Toast(self)

        # Surface well-load failures (real readers can fail on I/O).
        self.state.wellLoadFailed.connect(
            lambda wid, msg: self.toast(f"Could not open well {wid}. Try another well.")
        )

        # Load the demo dataset so every tab has data immediately.
        self.state.set_loader(MockLoader())

        if not config.get("first_run_done", False):
            # Show the welcome guide after the window is up.
            from PySide6.QtCore import QTimer
            QTimer.singleShot(250, self._show_welcome)

    # --- shell API (used by views) ----------------------------------------
    def show_wells(self) -> None:
        self._tabs.set_current_index(WELLS)

    def show_viewer(self) -> None:
        self._tabs.set_current_index(VIEWER)

    def show_cells(self) -> None:
        self._tabs.set_current_index(CELLS)

    def show_results(self) -> None:
        self._tabs.set_current_index(RESULTS)

    def present_sheet(self, title: str, content: QWidget) -> None:
        self._sheet.present(title, content)

    def dismiss_sheet(self) -> None:
        self._sheet.dismiss()

    def toast(self, message: str) -> None:
        self._toast.show_message(message)

    def open_experiment(self) -> None:
        # Native folder picker; Cancel is a no-op (don't wipe existing results).
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of TIFF, PNG, or JPG images")
        if not path:
            return
        self.load_folder(path)

    def load_folder(self, path: str) -> None:
        """Load a real image folder off the UI thread (scanning can be slow).

        A Cephla/Squid acquisition folder is detected and read with the dedicated
        reader; any other folder falls back to the generic TIFF/PNG/JPG reader.
        """
        self.toast("Opening folder...")

        def build():
            if CephlaLoader.looks_like(Path(path)):
                return CephlaLoader(path)
            return FolderLoader(path)

        def done(loader) -> None:
            self.state.set_loader(loader)
            self.show_wells()
            n = len(self.state.wells)
            self.toast(f"Loaded {n} position{'s' if n != 1 else ''} from {Path(path).name}")

        def failed(msg: str) -> None:
            if "No TIFF" in msg or "no readable" in msg or "No Cephla" in msg \
                    or "No numbered" in msg:
                self.toast("No images found in that folder")
            else:
                self.toast(f"Could not open that folder: {msg}")

        self._folder_tasks = getattr(self, "_folder_tasks", [])
        run_async(build, on_done=done, on_failed=failed, registry=self._folder_tasks)

    def load_demo(self) -> None:
        self.state.set_loader(MockLoader())
        self.show_wells()
        self.toast("Loaded the demo plate (synthetic data)")

    # --- navigation -------------------------------------------------------
    def _on_tab_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        view = self._stack.widget(index)
        if hasattr(view, "on_shown"):
            view.on_shown()

    # --- overlays geometry ------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._sheet.isVisible():
            self._sheet.setGeometry(self.rect())

    def closeEvent(self, event) -> None:  # noqa: N802
        # Let in-flight worker tasks finish before teardown so their signals
        # don't fire on deleted objects.
        from PySide6.QtCore import QThreadPool
        pool = QThreadPool.globalInstance()
        pool.clear()
        pool.waitForDone(3000)
        super().closeEvent(event)

    # --- welcome guide ----------------------------------------------------
    def _show_welcome(self) -> None:
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(22, 4, 22, 24)
        v.setSpacing(16)

        intro = QLabel(
            "CellScope follows and measures individual cells over time, across "
            "many wells, without code or parameter tuning."
        )
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        steps = [
            ("1. Pick wells", "Browse the plate on the Wells tab and tap a well to open it."),
            ("2. Detect cells", "On the Viewer, tap Detect Cells. One Sensitivity slider is all you need."),
            ("3. See results", "Watch cells get IDs and tracks, then open Results for charts and CSV export."),
        ]
        for title, body in steps:
            step = QLabel(f"<b>{title}</b><br><span>{body}</span>")
            step.setWordWrap(True)
            step.setTextFormat(Qt.TextFormat.RichText)
            v.addWidget(step)

        start = make_button("Start exploring", "primary")
        start.clicked.connect(self._finish_welcome)
        v.addWidget(start)

        self.present_sheet(f"Welcome to {__app_name__}", content)

    def _finish_welcome(self) -> None:
        config.set_value("first_run_done", True)
        self.dismiss_sheet()


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)

    theme.apply_theme(app, config.get("theme", "system"))

    window = MainWindow()
    window.show()

    # Optional: open a folder passed on the command line, e.g.
    #   cellscope "C:\path\to\images"
    args = [a for a in app.arguments()[1:] if not a.startswith("-")]
    if args and Path(args[0]).is_dir():
        window.load_folder(args[0])

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
