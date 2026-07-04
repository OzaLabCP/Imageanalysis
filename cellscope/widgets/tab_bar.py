"""iOS-style bottom tab bar."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from cellscope import theme
from cellscope.widgets import icons


class TabBar(QWidget):
    currentChanged = Signal(int)

    def __init__(self, items: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items = list(items)  # (label, icon_name)
        self._current = 0
        self.setFixedHeight(64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def current_index(self) -> int:
        return self._current

    def set_current_index(self, index: int, emit: bool = True) -> None:
        index = max(0, min(index, len(self._items) - 1))
        if index == self._current:
            return
        self._current = index
        self.update()
        if emit:
            self.currentChanged.emit(index)

    def _item_width(self) -> float:
        return self.width() / max(1, len(self._items))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        idx = int(event.position().x() // self._item_width())
        self.set_current_index(idx)

    def sizeHint(self) -> QSize:
        return QSize(400, 64)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), theme.color("tabbar_bg"))

        painter.setPen(QPen(theme.color("separator"), 1))
        painter.drawLine(0, 0, self.width(), 0)

        item_w = self._item_width()
        font = QFont(self.font())
        font.setPointSizeF(9.5)
        painter.setFont(font)

        for i, (label, icon_name) in enumerate(self._items):
            active = i == self._current
            col = theme.color("accent") if active else theme.color("tabbar_inactive")
            cx = i * item_w + item_w / 2
            icon_rect = QRectF(cx - 14, 9, 28, 26)
            icons.paint_icon(painter, icon_name, icon_rect, col, weight=2.0)
            painter.setPen(QPen(col))
            text_rect = QRectF(i * item_w, 38, item_w, 18)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
