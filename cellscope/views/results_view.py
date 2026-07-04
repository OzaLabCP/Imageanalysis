"""Results tab: count-over-time chart, per-cell table, and CSV/PNG export."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellscope import config
from cellscope.colors import track_color
from cellscope.export import write_measurements_csv
from cellscope.widgets.chart import LineChart, Series
from cellscope.widgets.controls import Card, Header, make_button


class ResultsView(QWidget):
    def __init__(self, state, shell, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.shell = shell

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = Header("Results", "")
        root.addWidget(self._header)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(16, 4, 16, 16)
        bl.setSpacing(14)
        root.addWidget(body, 1)

        # Chart card.
        chart_card = Card()
        cc = QVBoxLayout(chart_card)
        cc.setContentsMargins(16, 14, 16, 16)
        cc.setSpacing(8)
        chart_title = QLabel("Cells over time")
        chart_title.setObjectName("CardTitle")
        cc.addWidget(chart_title)
        self._chart = LineChart()
        self._chart.set_axis_titles("Time (frame)", "Cell count")
        self._chart.set_empty_text("Run Detect Cells to see a count-over-time curve")
        cc.addWidget(self._chart)
        bl.addWidget(chart_card)

        # Summary line.
        self._summary = QLabel("")
        self._summary.setObjectName("Hint")
        bl.addWidget(self._summary)

        # Table.
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bl.addWidget(self._table, 1)

        # Export row.
        export_row = QHBoxLayout()
        export_row.setSpacing(12)
        self._note = QLabel("")
        self._note.setObjectName("Hint")
        export_row.addWidget(self._note)
        export_row.addStretch(1)
        self._csv_btn = make_button("Export CSV", "default")
        self._csv_btn.clicked.connect(self._export_csv)
        self._png_btn = make_button("Export PNG", "primary")
        self._png_btn.clicked.connect(self._export_png)
        export_row.addWidget(self._csv_btn)
        export_row.addWidget(self._png_btn)
        bl.addLayout(export_row)

        self._empty = QLabel(
            "No results yet.\nDetect cells on the Viewer tab, then come back here."
        )
        self._empty.setObjectName("Hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self._empty)

        state.analysisStarted.connect(lambda _w: self._rebuild())
        state.analysisFinished.connect(lambda _w: self._rebuild())
        state.analysisFailed.connect(lambda _w, _m: self._rebuild())
        state.currentWellChanged.connect(lambda _w: self._rebuild())
        state.selectionChanged.connect(self._rebuild)
        state.pixelSizeChanged.connect(lambda _v: self._rebuild())

    # --- helpers ----------------------------------------------------------
    def _included_wells(self) -> list[str]:
        wells = []
        if self.state.current_well_id and self.state.analysis_for(self.state.current_well_id):
            wells.append(self.state.current_well_id)
        for wid in sorted(self.state.selected_wells):
            if wid not in wells and self.state.analysis_for(wid):
                wells.append(wid)
        return wells

    def _rebuild(self) -> None:
        wells = self._included_wells()
        has = bool(wells)
        running = self.state.is_running(self.state.current_well_id)
        self._empty.setVisible(not has)
        self._csv_btn.setEnabled(has)
        self._png_btn.setEnabled(has)

        if not has:
            error = self.state.error_for(self.state.current_well_id)
            if error:
                text = (f"Detection failed for well {self.state.current_well_id}.\n"
                        "Tap Detect Cells on the Viewer to retry.")
            elif running:
                text = "Computing results...\nDetection is running in the background."
            else:
                text = "No results yet.\nDetect cells on the Viewer tab, then come back here."
            self._empty.setText(text)
            self._header.set_subtitle("")
            self._chart.clear()
            self._summary.setText("")
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._note.setText("")
            return

        if len(wells) == 1:
            self._header.set_subtitle(f"Well {wells[0]}")
            self._note.setText("Tip: select wells on the Wells tab to compare.")
        else:
            self._header.set_subtitle(f"{len(wells)} wells")
            self._note.setText("Comparing selected wells.")

        # Chart: one count-over-time line per included well.
        series = []
        for i, wid in enumerate(wells):
            wa = self.state.analysis_for(wid)
            x = np.arange(wa.n_time)
            y = wa.counts_per_frame.astype(float)
            color = (10, 132, 255) if len(wells) == 1 else track_color(i + 1)
            series.append(Series(f"Well {wid}", color, x, y))
        self._chart.set_series(series)

        # Summary for the primary well.
        primary = self.state.analysis_for(wells[0])
        mean_count = float(primary.counts_per_frame.mean())
        self._summary.setText(
            f"Well {wells[0]}: {primary.n_tracks} cells tracked, "
            f"{mean_count:.1f} visible per frame on average, "
            f"over {primary.n_time} timepoints."
        )

        self._populate_table(wells)

    def _populate_table(self, wells: list[str]) -> None:
        primary = self.state.analysis_for(wells[0])
        channel_names = primary.channel_names
        headers = (["Well", "Cell", "Time", "Area (um^2)", "Diameter (um)"]
                   + [f"{n} mean" for n in channel_names])

        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)

        rows = []
        for wid in wells:
            wa = self.state.analysis_for(wid)
            for m in wa.measurements:
                rows.append((wid, m))

        self._table.setRowCount(len(rows))
        for r, (wid, m) in enumerate(rows):
            self._set_cell(r, 0, wid)
            self._set_cell(r, 1, m.track_id, numeric=True)
            self._set_cell(r, 2, m.frame + 1, numeric=True)
            self._set_cell(r, 3, round(m.area_um2, 2), numeric=True)
            self._set_cell(r, 4, round(m.feret_diameter_um, 2), numeric=True)
            for c, _name in enumerate(channel_names):
                val = m.mean_intensity[c] if c < len(m.mean_intensity) else 0.0
                self._set_cell(r, 5 + c, round(val, 1), numeric=True)
        self._table.setSortingEnabled(True)

    def _set_cell(self, row: int, col: int, value, numeric: bool = False) -> None:
        item = QTableWidgetItem()
        if numeric:
            item.setData(Qt.ItemDataRole.DisplayRole, value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            item.setText(str(value))
        self._table.setItem(row, col, item)

    # --- export -----------------------------------------------------------
    def _default_dir(self) -> str:
        return config.get("last_export_dir", "") or str(__import__("pathlib").Path.home())

    def _export_csv(self) -> None:
        wells = self._included_wells()
        if not wells:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements as CSV",
            f"{self._default_dir()}/cellscope_measurements.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self.write_csv(path, wells)
            config.set_value("last_export_dir", str(__import__("pathlib").Path(path).parent))
            self.shell.toast(f"Saved CSV to {path}")
        except OSError as exc:
            self.shell.toast(f"Could not save CSV: {exc}")

    def write_csv(self, path: str, wells: list[str]) -> None:
        """Write per-cell, per-frame measurements for ``wells`` to a CSV file."""
        items = [(wid, self.state.condition_of(wid), self.state.analysis_for(wid))
                 for wid in wells]
        write_measurements_csv(path, items)

    def _export_png(self) -> None:
        if not self._included_wells():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export chart as PNG",
            f"{self._default_dir()}/cellscope_chart.png",
            "PNG image (*.png)",
        )
        if not path:
            return
        if self.write_png(path):
            config.set_value("last_export_dir", str(__import__("pathlib").Path(path).parent))
            self.shell.toast(f"Saved chart to {path}")
        else:
            self.shell.toast("Could not save PNG")

    def write_png(self, path: str) -> bool:
        """Save the count-over-time chart to ``path`` as a PNG. Returns success."""
        return bool(self._chart.grab().save(path, "PNG"))

    def on_shown(self) -> None:
        self._rebuild()
