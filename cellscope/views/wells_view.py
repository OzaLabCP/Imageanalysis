"""Wells tab: a photo-gallery grid of well thumbnails."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellscope import theme
from cellscope.render import compose_rgb, rgb_to_qimage
from cellscope.widgets.controls import Header, IconButton, card_shadow, make_button
from cellscope.widgets.icons import paint_icon
from cellscope.widgets.worker import run_async

CARD_MIN_W = 178
THUMB_H = 150


class WellCard(QFrame):
    opened = Signal(str)
    toggled = Signal(str)

    def __init__(self, well_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.well_id = well_id
        self._analyzed = False
        self._selected = False
        self._is_current = False
        self.setObjectName("Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        card_shadow(self, blur=20, dy=4, alpha=30)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._thumb = QLabel()
        self._thumb.setFixedHeight(THUMB_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            f"QLabel {{ background-color: {theme.palette()['canvas_bg']};"
            f" border-radius: 10px; color: {theme.palette()['text_secondary']}; }}"
        )
        self._thumb.setText("...")
        layout.addWidget(self._thumb)

        self._caption = QLabel(f"Well {well_id}")
        self._caption.setObjectName("CardTitle")
        layout.addWidget(self._caption)

        self._cond = QLabel("")
        self._cond.setVisible(False)
        self._cond.setWordWrap(True)
        layout.addWidget(self._cond)

        self._status = QLabel("Not analyzed")
        self._status.setObjectName("Hint")
        layout.addWidget(self._status)

        # Corner toggle for multi-select "compare".
        self._select_btn = IconButton("check", size=30, parent=self)
        self._select_btn.clicked.connect(lambda: self.toggled.emit(self.well_id))
        self._select_btn.setToolTip("Add to comparison")

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._select_btn.move(self.width() - 38, 8)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            self._thumb.width() if self._thumb.width() > 0 else 160,
            THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)

    def set_analyzed(self, analyzed: bool, n_cells: int = 0) -> None:
        self._analyzed = analyzed
        if analyzed:
            self._status.setText(f"✓ {n_cells} cells tracked")
            self._status.setStyleSheet(f"color: {theme.palette()['success']};")
        else:
            self._status.setText("Not analyzed")
            self._status.setStyleSheet("")
        self.update()

    def set_condition(self, name: str, color) -> None:
        if name:
            r, g, b = color
            self._cond.setText(name)
            self._cond.setStyleSheet(
                f"QLabel {{ background-color: rgba({r},{g},{b},45);"
                f" color: {theme.palette()['text']};"
                f" border-left: 4px solid rgb({r},{g},{b});"
                f" border-radius: 6px; padding: 2px 7px; font-size: 12px; }}"
            )
            self._cond.setVisible(True)
        else:
            self._cond.setVisible(False)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def set_current(self, is_current: bool) -> None:
        self._is_current = is_current
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._select_btn.geometry().contains(event.position().toPoint()):
                self.opened.emit(self.well_id)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self._selected or self._is_current:
            pen = QPen(theme.color("accent"), 2.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(r, 16, 16)
        # Selection badge.
        badge = QRectF(self.width() - 38, 8, 30, 30)
        if self._selected:
            painter.setBrush(theme.color("accent"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(badge)
            paint_icon(painter, "check", badge, QColor("#FFFFFF"), weight=2.4)
        else:
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(badge)
            paint_icon(painter, "check", badge, QColor(255, 255, 255, 200), weight=2.0)


class WellsView(QWidget):
    def __init__(self, state, shell, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.shell = shell
        self._cards: dict[str, WellCard] = {}
        self._tasks: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = Header("Wells", "")
        platemap_btn = make_button("Plate map", "ghost")
        platemap_btn.setMinimumHeight(40)
        platemap_btn.clicked.connect(self._open_platemap)
        self._header.add_action(platemap_btn)
        demo_btn = make_button("Demo", "ghost")
        demo_btn.setMinimumHeight(40)
        demo_btn.clicked.connect(self.shell.load_demo)
        self._header.add_action(demo_btn)
        open_btn = make_button("Open", "primary")
        open_btn.setMinimumHeight(40)
        open_btn.clicked.connect(self.shell.open_experiment)
        self._header.add_action(open_btn)
        root.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self._scroll, 1)

        self._container = QWidget()
        self._grid_host = QVBoxLayout(self._container)
        self._grid_host.setContentsMargins(16, 4, 16, 24)
        self._grid_host.setSpacing(14)
        self._scroll.setWidget(self._container)

        self._empty = QLabel("No wells to show.\nTap Demo for the sample plate, "
                             "or Open for a folder of your own images.")
        self._empty.setObjectName("Hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._grid_host.addWidget(self._empty)

        self._rows_widget: QWidget | None = None

        state.experimentLoaded.connect(self._rebuild)
        state.analysisFinished.connect(self._refresh_badges)
        state.currentWellChanged.connect(self._refresh_current)
        state.selectionChanged.connect(self._refresh_selection)
        state.conditionsChanged.connect(self._refresh_conditions)

    def _open_platemap(self) -> None:
        if not self.state.wells:
            self.shell.toast("Open an experiment first")
            return
        from cellscope.views.platemap_view import PlateMapEditor
        self.shell.present_sheet("Plate map", PlateMapEditor(self.state, self.shell))

    def _refresh_conditions(self) -> None:
        for wid, card in self._cards.items():
            cond = self.state.condition_of(wid)
            card.set_condition(cond, self.state.condition_color(cond) if cond else None)

    def _rebuild(self) -> None:
        self._tasks.clear()  # drop references to any in-flight thumbnail loads
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        self._header.set_subtitle(
            f"{self.state.loader.name} · {len(self.state.wells)} wells"
            if self.state.loader else ""
        )
        self._empty.setVisible(not self.state.wells)

        for w in self.state.wells:
            card = WellCard(w.well_id)
            card.opened.connect(self._open_well)
            card.toggled.connect(self.state.toggle_well_selected)
            self._cards[w.well_id] = card

        self._reflow()
        self._load_thumbnails()
        self._refresh_current(self.state.current_well_id)
        self._refresh_conditions()

    def _open_well(self, well_id: str) -> None:
        self.state.set_current_well(well_id)
        self.shell.show_viewer()

    def _load_thumbnails(self) -> None:
        loader = self.state.loader
        if loader is None:
            return
        for well_id in list(self._cards.keys()):
            def make_done(wid: str):
                def done(frame):
                    card = self._cards.get(wid)
                    if card is None:
                        return
                    rgb = compose_rgb(frame, loader.channel_colors,
                                      [True] * frame.shape[0], 0.5, 0.62)
                    card.set_thumbnail(QPixmap.fromImage(rgb_to_qimage(rgb)))
                return done
            # get_thumbnail is cheap (nav cache / one downsampled frame), unlike
            # loading a whole multi-GB position just to draw a 150px card.
            run_async(loader.get_thumbnail, well_id,
                      on_done=make_done(well_id), registry=self._tasks)

    def _reflow(self) -> None:
        # Rebuild a fresh grid of rows sized to the available width.
        if self._rows_widget is not None:
            self._grid_host.removeWidget(self._rows_widget)
            self._rows_widget.setParent(None)
            self._rows_widget.deleteLater()
            self._rows_widget = None
        if not self._cards:
            return

        width = max(self._scroll.viewport().width(), 320)
        cols = max(2, (width - 32) // CARD_MIN_W)

        self._rows_widget = QWidget()
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(self._rows_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(14)
        for idx, well_id in enumerate(self._cards.keys()):
            row, col = divmod(idx, cols)
            grid.addWidget(self._cards[well_id], row, col)
        for c in range(cols):
            grid.setColumnStretch(c, 1)
        self._grid_host.insertWidget(0, self._rows_widget)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._cards:
            self._reflow()

    def _refresh_badges(self, well_id: str) -> None:
        card = self._cards.get(well_id)
        wa = self.state.analysis_for(well_id)
        if card and wa:
            card.set_analyzed(True, wa.n_tracks)

    def _refresh_current(self, well_id: str) -> None:
        for wid, card in self._cards.items():
            card.set_current(wid == well_id)

    def _refresh_selection(self) -> None:
        for wid, card in self._cards.items():
            card.set_selected(wid in self.state.selected_wells)

    def sizeHint(self) -> QSize:
        return QSize(520, 600)
