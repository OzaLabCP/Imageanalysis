"""Minimal vector icons painted with QPainter (no asset files, theme-aware)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF


def paint_icon(painter: QPainter, name: str, rect: QRectF, color: QColor, weight: float = 2.0) -> None:
    """Draw a named line icon centered in ``rect``."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color, weight)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    # Work inside a padded square so all icons share visual weight.
    side = min(rect.width(), rect.height())
    pad = side * 0.16
    r = QRectF(
        rect.center().x() - side / 2 + pad,
        rect.center().y() - side / 2 + pad,
        side - 2 * pad,
        side - 2 * pad,
    )
    x, y, w, h = r.x(), r.y(), r.width(), r.height()

    if name == "wells":
        gap = w * 0.16
        cw = (w - gap) / 2
        ch = (h - gap) / 2
        for i in range(2):
            for j in range(2):
                cell = QRectF(x + i * (cw + gap), y + j * (ch + gap), cw, ch)
                painter.drawRoundedRect(cell, cw * 0.28, cw * 0.28)

    elif name == "viewer":
        painter.drawRoundedRect(r, w * 0.14, w * 0.14)
        # sun
        painter.setBrush(QBrush(color))
        rad = w * 0.11
        painter.drawEllipse(QPointF(x + w * 0.34, y + h * 0.34), rad, rad)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # mountain
        path = QPainterPath(QPointF(x + w * 0.08, y + h * 0.86))
        path.lineTo(QPointF(x + w * 0.42, y + h * 0.5))
        path.lineTo(QPointF(x + w * 0.62, y + h * 0.68))
        path.lineTo(QPointF(x + w * 0.8, y + h * 0.52))
        path.lineTo(QPointF(x + w * 0.92, y + h * 0.64))
        painter.drawPath(path)

    elif name == "cells":
        painter.drawEllipse(QRectF(x, y + h * 0.18, w * 0.6, h * 0.6))
        painter.drawEllipse(QRectF(x + w * 0.36, y + h * 0.3, w * 0.5, h * 0.5))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(x + w * 0.3, y + h * 0.48), w * 0.06, w * 0.06)
        painter.drawEllipse(QPointF(x + w * 0.61, y + h * 0.55), w * 0.05, w * 0.05)

    elif name == "results":
        base = y + h * 0.92
        bars = [(0.10, 0.45), (0.40, 0.72), (0.70, 0.30)]
        bw = w * 0.18
        for bx, bh in bars:
            painter.drawLine(QPointF(x + (bx + 0.09) * w, base),
                             QPointF(x + (bx + 0.09) * w, base - bh * h))
        # axis
        painter.drawLine(QPointF(x + w * 0.05, base), QPointF(x + w * 0.95, base))

    elif name == "play":
        painter.setBrush(QBrush(color))
        tri = QPolygonF([
            QPointF(x + w * 0.24, y + h * 0.12),
            QPointF(x + w * 0.86, y + h * 0.5),
            QPointF(x + w * 0.24, y + h * 0.88),
        ])
        painter.drawPolygon(tri)

    elif name == "pause":
        bw = w * 0.22
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(x + w * 0.22, y + h * 0.12, bw, h * 0.76), 2, 2)
        painter.drawRoundedRect(QRectF(x + w * 0.56, y + h * 0.12, bw, h * 0.76), 2, 2)

    elif name == "gear":
        cx, cy = x + w / 2, y + h / 2
        painter.drawEllipse(QPointF(cx, cy), w * 0.2, h * 0.2)
        painter.drawEllipse(QPointF(cx, cy), w * 0.34, h * 0.34)

    elif name == "layers":
        mid = QPointF(x + w / 2, y + h / 2)
        for dy in (-0.16, 0.04, 0.24):
            poly = QPolygonF([
                QPointF(mid.x(), y + (0.3 + dy) * h),
                QPointF(x + w * 0.85, y + (0.45 + dy) * h),
                QPointF(mid.x(), y + (0.6 + dy) * h),
                QPointF(x + w * 0.15, y + (0.45 + dy) * h),
            ])
            painter.drawPolygon(poly)

    elif name == "export":
        cx = x + w / 2
        painter.drawLine(QPointF(cx, y + h * 0.1), QPointF(cx, y + h * 0.62))
        arrow = QPainterPath(QPointF(cx - w * 0.16, y + h * 0.28))
        arrow.lineTo(QPointF(cx, y + h * 0.1))
        arrow.lineTo(QPointF(cx + w * 0.16, y + h * 0.28))
        painter.drawPath(arrow)
        tray = QPainterPath(QPointF(x + w * 0.16, y + h * 0.55))
        tray.lineTo(QPointF(x + w * 0.16, y + h * 0.86))
        tray.lineTo(QPointF(x + w * 0.84, y + h * 0.86))
        tray.lineTo(QPointF(x + w * 0.84, y + h * 0.55))
        painter.drawPath(tray)

    elif name == "folder":
        path = QPainterPath(QPointF(x + w * 0.08, y + h * 0.3))
        path.lineTo(QPointF(x + w * 0.4, y + h * 0.3))
        path.lineTo(QPointF(x + w * 0.48, y + h * 0.42))
        path.lineTo(QPointF(x + w * 0.92, y + h * 0.42))
        path.lineTo(QPointF(x + w * 0.92, y + h * 0.82))
        path.lineTo(QPointF(x + w * 0.08, y + h * 0.82))
        path.closeSubpath()
        painter.drawPath(path)

    elif name == "close":
        painter.drawLine(QPointF(x + w * 0.2, y + h * 0.2), QPointF(x + w * 0.8, y + h * 0.8))
        painter.drawLine(QPointF(x + w * 0.8, y + h * 0.2), QPointF(x + w * 0.2, y + h * 0.8))

    elif name == "check":
        path = QPainterPath(QPointF(x + w * 0.18, y + h * 0.52))
        path.lineTo(QPointF(x + w * 0.42, y + h * 0.74))
        path.lineTo(QPointF(x + w * 0.84, y + h * 0.26))
        painter.drawPath(path)

    elif name == "target":
        cx, cy = x + w / 2, y + h / 2
        painter.drawEllipse(QPointF(cx, cy), w * 0.34, h * 0.34)
        painter.drawLine(QPointF(cx, y), QPointF(cx, y + h * 0.16))
        painter.drawLine(QPointF(cx, y + h * 0.84), QPointF(cx, y + h))
        painter.drawLine(QPointF(x, cy), QPointF(x + w * 0.16, cy))
        painter.drawLine(QPointF(x + w * 0.84, cy), QPointF(x + w, cy))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), w * 0.07, h * 0.07)

    painter.restore()
