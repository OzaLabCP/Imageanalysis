"""A small, dependency-free line chart (QPainter) with an iOS-clean look."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from cellscope import theme


@dataclass
class Series:
    label: str
    color: tuple[int, int, int]
    x: np.ndarray
    y: np.ndarray


class LineChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[Series] = []
        self._x_title = ""
        self._y_title = ""
        self._empty_text = "No data yet"
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_axis_titles(self, x_title: str, y_title: str) -> None:
        self._x_title = x_title
        self._y_title = y_title
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    def set_series(self, series: list[Series]) -> None:
        self._series = [s for s in series if s.x is not None and len(s.x) > 0]
        self.update()

    def clear(self) -> None:
        self._series = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0, 0, -1, -1)
        text_secondary = theme.color("text_secondary")

        if not self._series:
            painter.setPen(QPen(text_secondary))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return

        left, right, top, bottom = 52.0, 16.0, 16.0, 40.0
        legend_h = 24.0 if len(self._series) > 1 or self._series[0].label else 0.0
        plot = QRectF(
            rect.left() + left,
            rect.top() + top,
            rect.width() - left - right,
            rect.height() - top - bottom - legend_h,
        )

        # Data ranges.
        all_x = np.concatenate([s.x for s in self._series])
        all_y = np.concatenate([s.y for s in self._series])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        # A little headroom and a zero baseline where natural.
        y_min = min(y_min, 0.0) if y_min >= 0 else y_min
        y_pad = (y_max - y_min) * 0.08
        y_max += y_pad

        def to_px(x: float, y: float) -> QPointF:
            fx = (x - x_min) / (x_max - x_min)
            fy = (y - y_min) / (y_max - y_min)
            return QPointF(plot.left() + fx * plot.width(),
                           plot.bottom() - fy * plot.height())

        # Gridlines + y labels.
        grid_pen = QPen(theme.color("chart_grid"), 1)
        font = QFont(self.font())
        font.setPointSizeF(9.0)
        painter.setFont(font)
        n_ticks = 4
        for i in range(n_ticks + 1):
            yval = y_min + (y_max - y_min) * i / n_ticks
            p = to_px(x_min, yval)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), p.y()), QPointF(plot.right(), p.y()))
            painter.setPen(QPen(text_secondary))
            label = f"{yval:.0f}" if (y_max - y_min) >= 5 else f"{yval:.1f}"
            painter.drawText(
                QRectF(rect.left(), p.y() - 8, left - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        # X axis ticks (integer-ish).
        x_span = x_max - x_min
        step = max(1, int(np.ceil(x_span / 6)))
        painter.setPen(QPen(text_secondary))
        xv = int(np.floor(x_min))
        while xv <= x_max:
            p = to_px(xv, y_min)
            painter.drawText(
                QRectF(p.x() - 20, plot.bottom() + 4, 40, 16),
                Qt.AlignmentFlag.AlignCenter,
                str(xv),
            )
            xv += step

        # Axis titles.
        if self._x_title:
            painter.setPen(QPen(text_secondary))
            painter.drawText(
                QRectF(plot.left(), rect.bottom() - legend_h - 16, plot.width(), 16),
                Qt.AlignmentFlag.AlignCenter,
                self._x_title,
            )
        if self._y_title:
            painter.save()
            painter.translate(rect.left() + 12, plot.center().y())
            painter.rotate(-90)
            painter.setPen(QPen(text_secondary))
            painter.drawText(QRectF(-plot.height() / 2, -12, plot.height(), 16),
                             Qt.AlignmentFlag.AlignCenter, self._y_title)
            painter.restore()

        # Series lines.
        for s in self._series:
            color = QColor(*s.color)
            pen = QPen(color, 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            poly = QPolygonF([to_px(float(x), float(y)) for x, y in zip(s.x, s.y)])
            painter.drawPolyline(poly)
            if len(poly) <= 30:
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                for pt in poly:
                    painter.drawEllipse(pt, 2.6, 2.6)

        # Legend.
        if legend_h > 0:
            painter.setFont(font)
            lx = plot.left()
            ly = rect.bottom() - legend_h + 4
            for s in self._series:
                color = QColor(*s.color)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QRectF(lx, ly + 2, 14, 8), 4, 4)
                painter.setPen(QPen(theme.color("text")))
                tw = painter.fontMetrics().horizontalAdvance(s.label)
                painter.drawText(QRectF(lx + 20, ly - 2, tw + 10, 18),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 s.label)
                lx += 20 + tw + 24
