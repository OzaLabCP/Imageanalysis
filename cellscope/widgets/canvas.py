"""The image canvas: draws the composited frame plus cell overlays, with
zoom (wheel / pinch), pan (drag), and tap-to-select-a-cell.

Kept free of app state - the Viewer feeds it images and overlay data.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from cellscope import theme
from cellscope.colors import track_color
from cellscope.render import build_color_lut, outline_overlay


class ImageCanvas(QWidget):
    cellClicked = Signal(int)  # track ID, or -1 to clear selection
    rulerMeasured = Signal(float)  # length in image pixels

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setMouseTracking(True)
        self.grabGesture(Qt.GestureType.PinchGesture)

        self._pixmap: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._label_image: np.ndarray | None = None
        self._tracks: dict[int, np.ndarray] = {}
        self._centroids: dict[int, tuple[float, float]] = {}
        self._img_w = 0
        self._img_h = 0
        self._current_t = 0
        self._selected = -1
        self._show_outlines = True
        self._show_tracks = True
        self._show_labels = True

        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._needs_fit = True
        self._user_interacted = False  # becomes True once the user zooms/pans

        self._press_pos: QPointF | None = None
        self._panning = False

        # Ruler (scale calibration): endpoints stored in image coordinates.
        self._ruler_mode = False
        self._ruler_start: QPointF | None = None
        self._ruler_end: QPointF | None = None

    # --- inputs -----------------------------------------------------------
    def clear(self) -> None:
        self._pixmap = None
        self._overlay = None
        self._label_image = None
        self._tracks = {}
        self._centroids = {}
        self.update()

    def set_base_image(self, qimage) -> None:
        if qimage is None:
            self._pixmap = None
            self.update()
            return
        new_size = (qimage.width(), qimage.height())
        if new_size != (self._img_w, self._img_h):
            self._img_w, self._img_h = new_size
            self._needs_fit = True
        self._pixmap = QPixmap.fromImage(qimage)
        if self._needs_fit:
            self.fit_to_view()
        self.update()

    def set_label_image(self, label_image: np.ndarray | None) -> None:
        self._label_image = label_image
        if label_image is None or not self._show_outlines:
            self._overlay = None
        else:
            ids = np.unique(label_image)
            ids = [int(i) for i in ids if i > 0]
            lut = build_color_lut(ids, track_color)
            self._overlay = QPixmap.fromImage(outline_overlay(label_image, lut))
        self.update()

    def set_tracks(self, tracks: dict[int, np.ndarray]) -> None:
        self._tracks = tracks or {}
        self.update()

    def set_frame_centroids(self, centroids: dict[int, tuple[float, float]]) -> None:
        self._centroids = centroids or {}
        self.update()

    def set_current_t(self, t: int) -> None:
        self._current_t = t
        self.update()

    def set_selected_track(self, tid: int) -> None:
        self._selected = tid
        self.update()

    def set_ruler_mode(self, on: bool) -> None:
        self._ruler_mode = on
        if on:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def clear_ruler(self) -> None:
        self._ruler_start = None
        self._ruler_end = None
        self.update()

    def set_layers(self, outlines: bool, tracks: bool, labels: bool) -> None:
        rebuild = outlines != self._show_outlines
        self._show_outlines = outlines
        self._show_tracks = tracks
        self._show_labels = labels
        if rebuild:
            self.set_label_image(self._label_image)
        self.update()

    # --- view transform ---------------------------------------------------
    def fit_to_view(self) -> None:
        if self._img_w == 0 or self._img_h == 0:
            return
        avail_w = max(1, self.width())
        avail_h = max(1, self.height())
        scale = min(avail_w / self._img_w, avail_h / self._img_h) * 0.96
        self._scale = scale
        self._offset = QPointF(
            (avail_w - self._img_w * scale) / 2,
            (avail_h - self._img_h * scale) / 2,
        )
        self._needs_fit = False
        self._user_interacted = False
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        # Keep the image filling the canvas until the user zooms/pans.
        if self._needs_fit or not self._user_interacted:
            self.fit_to_view()

    def _to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._offset.x() + x * self._scale,
                       self._offset.y() + y * self._scale)

    def _to_image(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self._offset.x()) / self._scale,
                       (pt.y() - self._offset.y()) / self._scale)

    def _zoom_at(self, anchor: QPointF, factor: float) -> None:
        new_scale = max(0.05, min(40.0, self._scale * factor))
        if new_scale == self._scale:
            return
        img_pt = self._to_image(anchor)
        self._scale = new_scale
        self._offset = QPointF(anchor.x() - img_pt.x() * new_scale,
                               anchor.y() - img_pt.y() * new_scale)
        self._user_interacted = True
        self.update()

    # --- events -----------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0 + (0.0015 * delta)
        self._zoom_at(event.position(), factor)

    def event(self, ev):  # noqa: N802 - handle pinch gestures
        if ev.type() == ev.Type.Gesture:
            pinch = ev.gesture(Qt.GestureType.PinchGesture)
            if pinch is not None:
                factor = pinch.scaleFactor()
                center = self.mapFromGlobal(pinch.centerPoint().toPoint())
                self._zoom_at(QPointF(center), factor)
                return True
        return super().event(ev)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._ruler_mode and self._pixmap is not None:
            self._ruler_start = self._to_image(event.position())
            self._ruler_end = self._ruler_start
            self.update()
            return
        self._press_pos = event.position()
        self._panning = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._ruler_mode and self._ruler_start is not None \
                and (event.buttons() & Qt.MouseButton.LeftButton):
            self._ruler_end = self._to_image(event.position())
            self.update()
            return
        if self._press_pos is None:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._press_pos
            if not self._panning and (abs(delta.x()) + abs(delta.y())) > 4:
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._panning:
                self._offset += event.position() - self._press_pos
                self._press_pos = event.position()
                self._user_interacted = True
                self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._ruler_mode and self._ruler_start is not None:
            self._ruler_end = self._to_image(event.position())
            dx = self._ruler_end.x() - self._ruler_start.x()
            dy = self._ruler_end.y() - self._ruler_start.y()
            self.update()
            self.rulerMeasured.emit(math.hypot(dx, dy))
            return
        self.unsetCursor()
        was_panning = self._panning
        self._panning = False
        self._press_pos = None
        if was_panning or self._label_image is None:
            return
        # A click (no drag): hit-test the label image to select a cell.
        img_pt = self._to_image(event.position())
        ix, iy = int(img_pt.x()), int(img_pt.y())
        if 0 <= iy < self._label_image.shape[0] and 0 <= ix < self._label_image.shape[1]:
            tid = int(self._label_image[iy, ix])
            self.cellClicked.emit(tid if tid > 0 else -1)
        else:
            self.cellClicked.emit(-1)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_to_view()

    # --- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.color("canvas_bg"))

        if self._pixmap is None:
            painter.setPen(QPen(QColor("#8A8A8E")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No image loaded")
            return

        target = QRectF(self._offset.x(), self._offset.y(),
                        self._img_w * self._scale, self._img_h * self._scale)
        source = QRectF(0, 0, self._img_w, self._img_h)
        # Crisp pixels (microscopy) - no smoothing.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawPixmap(target, self._pixmap, source)

        if self._show_outlines and self._overlay is not None:
            painter.drawPixmap(target, self._overlay, source)

        if self._show_tracks and self._tracks:
            self._paint_tracks(painter)

        if self._show_labels and self._centroids:
            self._paint_labels(painter)

        self._paint_selection(painter)
        self._paint_ruler(painter)

    def _paint_tracks(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for tid, pts in self._tracks.items():
            mask = pts[:, 0] <= self._current_t
            if mask.sum() < 2:
                continue
            visible = pts[mask]
            poly = QPolygonF([self._to_widget(p[2], p[1]) for p in visible])
            selected = tid == self._selected
            color = QColor(*track_color(tid))
            width = 3.4 if selected else 1.8
            if not selected and self._selected != -1:
                color.setAlpha(90)  # dim non-selected when one is focused
            pen = QPen(color, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(poly)

    def _paint_labels(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        for tid, (cy, cx) in self._centroids.items():
            if self._selected != -1 and tid != self._selected:
                continue  # when focused, only label the selected cell
            pt = self._to_widget(cx, cy)
            painter.setPen(QPen(QColor(0, 0, 0, 170)))
            painter.drawText(QPointF(pt.x() + 7, pt.y() - 5), str(tid))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPointF(pt.x() + 6, pt.y() - 6), str(tid))

    def _paint_selection(self, painter: QPainter) -> None:
        if self._selected == -1 or self._selected not in self._centroids:
            return
        cy, cx = self._centroids[self._selected]
        pt = self._to_widget(cx, cy)
        radius = max(12.0, 16.0 * self._scale / 2)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2.4))
        painter.drawEllipse(pt, radius + 2, radius + 2)
        painter.setPen(QPen(QColor(*track_color(self._selected)), 2.4))
        painter.drawEllipse(pt, radius, radius)

    def _paint_ruler(self, painter: QPainter) -> None:
        if self._ruler_start is None or self._ruler_end is None:
            return
        a = self._to_widget(self._ruler_start.x(), self._ruler_start.y())
        b = self._to_widget(self._ruler_end.x(), self._ruler_end.y())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Line with a dark halo so it reads on any background.
        painter.setPen(QPen(QColor(0, 0, 0, 160), 4.0))
        painter.drawLine(a, b)
        accent = theme.color("accent")
        painter.setPen(QPen(accent, 2.0))
        painter.drawLine(a, b)

        for pt in (a, b):
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(accent, 2.0))
            painter.drawEllipse(pt, 4.0, 4.0)

        dx = self._ruler_end.x() - self._ruler_start.x()
        dy = self._ruler_end.y() - self._ruler_start.y()
        length_px = math.hypot(dx, dy)
        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        text = f"{length_px:.0f} px"
        font = QFont(self.font())
        font.setPointSizeF(10.0)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        box = QRectF(mid.x() - w / 2, mid.y() - 26, w, 20)
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
