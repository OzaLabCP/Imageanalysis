"""Cells tab: browse every tracked cell, inspect one, follow it in the Viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellscope.colors import track_color
from cellscope.widgets.chart import LineChart, Series
from cellscope.widgets.controls import Card, Header, card_shadow, make_button


class CellRow(QFrame):
    clicked = Signal(int)

    def __init__(self, track_id: int, title: str, summary: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.track_id = track_id
        self._selected = False
        self.setObjectName("Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        card_shadow(self, blur=14, dy=2, alpha=24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        r, g, b = track_color(track_id)
        swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border-radius: 7px;")
        layout.addWidget(swatch)

        text = QVBoxLayout()
        text.setSpacing(1)
        self._title = QLabel(title)
        self._title.setObjectName("CardTitle")
        text.addWidget(self._title)
        self._summary = QLabel(summary)
        self._summary.setObjectName("Hint")
        text.addWidget(self._summary)
        layout.addLayout(text)
        layout.addStretch(1)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setStyleSheet(
            "QFrame#Card { border: 2px solid palette(highlight); }" if selected else ""
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.track_id)


class CellsView(QWidget):
    def __init__(self, state, shell, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.shell = shell
        self._rows: dict[int, CellRow] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = Header("Cells", "")
        root.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self._scroll, 1)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(16, 4, 16, 24)
        self._body_layout.setSpacing(14)
        self._scroll.setWidget(self._body)

        # Detail card (for the selected cell).
        self._detail = Card()
        dl = QVBoxLayout(self._detail)
        dl.setContentsMargins(16, 14, 16, 16)
        dl.setSpacing(10)
        self._detail_title = QLabel("Select a cell")
        self._detail_title.setObjectName("CardTitle")
        dl.addWidget(self._detail_title)
        self._stats = QGridLayout()
        self._stats.setHorizontalSpacing(18)
        self._stats.setVerticalSpacing(4)
        dl.addLayout(self._stats)
        self._chart = LineChart()
        self._chart.set_axis_titles("Time", "Mean intensity")
        self._chart.set_empty_text("Tap a cell to see its intensity over time")
        self._chart.setMinimumHeight(180)
        dl.addWidget(self._chart)
        self._follow_btn = make_button("Follow in Viewer", "ghost")
        self._follow_btn.clicked.connect(self._follow)
        self._follow_btn.setEnabled(False)
        dl.addWidget(self._follow_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._body_layout.addWidget(self._detail)

        self._list_label = QLabel("All cells")
        self._list_label.setObjectName("SectionLabel")
        self._body_layout.addWidget(self._list_label)

        self._list_holder = QVBoxLayout()
        self._list_holder.setSpacing(10)
        self._body_layout.addLayout(self._list_holder)

        self._empty = QLabel(
            "No cells detected yet.\nGo to the Viewer tab and tap Detect Cells."
        )
        self._empty.setObjectName("Hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._body_layout.addWidget(self._empty)
        self._body_layout.addStretch(1)

        state.analysisStarted.connect(self._on_analysis_started)
        state.analysisFinished.connect(self._on_analysis_finished)
        state.analysisFailed.connect(lambda wid, _m: self._on_analysis_finished(wid))
        state.currentWellChanged.connect(lambda _w: self._rebuild())
        state.selectedTrackChanged.connect(self._on_selected)
        state.pixelSizeChanged.connect(lambda _v: self._rebuild())

    def _on_analysis_started(self, well_id: str) -> None:
        if well_id == self.state.current_well_id:
            self._rebuild()

    def _on_analysis_finished(self, well_id: str) -> None:
        if well_id == self.state.current_well_id:
            self._rebuild()

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        while self._list_holder.count():
            item = self._list_holder.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def _rebuild(self) -> None:
        self._clear_rows()
        wa = self.state.analysis_for(self.state.current_well_id)
        has = wa is not None and wa.n_tracks > 0
        running = self.state.is_running(self.state.current_well_id)
        self._empty.setVisible(not has)
        self._detail.setVisible(has)
        self._list_label.setVisible(has)

        if not has:
            wid = self.state.current_well_id
            error = self.state.error_for(wid)
            analyzed_empty = wa is not None and wa.n_tracks == 0
            if error:
                text = (f"Detection failed for well {wid}.\n"
                        "Tap Detect Cells on the Viewer to retry.")
            elif running:
                text = "Detecting cells...\nThis is running in the background."
            elif analyzed_empty:
                text = ("Detection finished but found no cells.\n"
                        "Try raising Sensitivity on the Viewer.")
            else:
                text = "No cells detected yet.\nGo to the Viewer tab and tap Detect Cells."
            self._empty.setText(text)
            self._header.set_subtitle(f"Well {wid}" if wid else "")
            self._detail_title.setText("Select a cell")
            self._chart.clear()
            self._clear_stats()
            return

        self._header.set_subtitle(f"Well {wa.well_id} · {wa.n_tracks} cells")
        self._list_label.setText(f"All cells ({wa.n_tracks})")

        # Order cells by how long they persist (most trackable first).
        order = sorted(wa.tracks.items(), key=lambda kv: -len(kv[1]))
        for tid, pts in order:
            ms = wa.measurements_for_track(tid)
            frames = sorted(m.frame for m in ms)
            mean_area = float(np.mean([m.area_um2 for m in ms])) if ms else 0.0
            summary = (
                f"Seen in {len(frames)} frames "
                f"(t{frames[0] + 1}-{frames[-1] + 1}) · {mean_area:.1f} um^2 avg"
            )
            row = CellRow(tid, f"Cell {tid}", summary)
            row.clicked.connect(self.state.set_selected_track)
            self._rows[tid] = row
            self._list_holder.addWidget(row)

        self._follow_btn.setEnabled(self.state.selected_track in self._rows)
        if self.state.selected_track in self._rows:
            self._show_detail(self.state.selected_track)
        else:
            self._detail_title.setText("Select a cell")
            self._clear_stats()
            self._chart.clear()

    def _clear_stats(self) -> None:
        while self._stats.count():
            item = self._stats.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def _show_detail(self, tid: int) -> None:
        wa = self.state.analysis_for(self.state.current_well_id)
        if wa is None:
            return
        ms = sorted(wa.measurements_for_track(tid), key=lambda m: m.frame)
        if not ms:
            return
        r, g, b = track_color(tid)
        self._detail_title.setText(f"Cell {tid}")
        self._detail_title.setStyleSheet(f"color: rgb({r},{g},{b});")

        self._clear_stats()
        frames = [m.frame for m in ms]
        mean_area = float(np.mean([m.area_um2 for m in ms]))
        mean_diam = float(np.mean([m.feret_diameter_um for m in ms]))
        stats = [
            ("Lifespan", f"{len(frames)} frames"),
            ("First seen", f"t{frames[0] + 1}"),
            ("Mean area", f"{mean_area:.1f} um^2"),
            ("Mean diameter", f"{mean_diam:.1f} um"),
        ]
        for c, name in enumerate(wa.channel_names):
            avg = float(np.mean([m.mean_intensity[c] for m in ms]))
            stats.append((f"Mean {name}", f"{avg:.0f}"))
        for i, (k, val) in enumerate(stats):
            key = QLabel(k)
            key.setObjectName("Hint")
            value = QLabel(val)
            value.setObjectName("CardTitle")
            self._stats.addWidget(key, (i // 2) * 2, i % 2)
            self._stats.addWidget(value, (i // 2) * 2 + 1, i % 2)

        # Per-channel intensity over time.
        x = np.array(frames)
        series = []
        from cellscope.colors import channel_colors_for
        cc = self.state.loader.channel_colors if self.state.loader else channel_colors_for(len(wa.channel_names))
        for c, name in enumerate(wa.channel_names):
            y = np.array([m.mean_intensity[c] for m in ms])
            series.append(Series(name, cc[c] if c < len(cc) else (180, 180, 180), x, y))
        self._chart.set_series(series)

    def _on_selected(self, tid: int) -> None:
        for rid, row in self._rows.items():
            row.set_selected(rid == tid)
        self._follow_btn.setEnabled(tid in self._rows)
        if tid in self._rows:
            self._show_detail(tid)

    def _follow(self) -> None:
        tid = self.state.selected_track
        if tid not in self._rows:
            self.shell.toast("Tap a cell first")
            return
        # Jump to a frame where the cell exists so it is actually visible.
        wa = self.state.analysis_for(self.state.current_well_id)
        if wa:
            ms = wa.measurements_for_track(tid)
            if ms:
                self.state.set_t(min(m.frame for m in ms))
        self.shell.show_viewer()

    def on_shown(self) -> None:
        self._rebuild()
