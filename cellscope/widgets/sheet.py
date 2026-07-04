"""Modal bottom sheet + transient toast, both overlaid on the main window."""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellscope.widgets.controls import IconButton, card_shadow


class BottomSheet(QWidget):
    """Full-window overlay with a panel that slides up from the bottom."""

    closed = Signal()

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self.hide()

        self._scrim = QWidget(self)
        self._scrim.setObjectName("Scrim")
        self._scrim.mousePressEvent = lambda e: self.dismiss()

        self._panel = QFrame(self)
        self._panel.setObjectName("Sheet")
        card_shadow(self._panel, blur=44, dy=-6, alpha=90)

        outer = QVBoxLayout(self._panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Grabber handle.
        grabber = QLabel()
        grabber.setFixedHeight(18)
        outer.addWidget(grabber)

        header = QHBoxLayout()
        header.setContentsMargins(20, 0, 12, 6)
        self._title = QLabel("")
        self._title.setObjectName("CardTitle")
        f = QFont(self._title.font())
        f.setPointSizeF(17.0)
        f.setWeight(QFont.Weight.Bold)
        self._title.setFont(f)
        header.addWidget(self._title)
        header.addStretch(1)
        close_btn = IconButton("close", size=36)
        close_btn.clicked.connect(self.dismiss)
        header.addWidget(close_btn)
        outer.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        self._anim = QPropertyAnimation(self._panel, b"pos", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._panel_height = 320
        self._dismiss_connected = False

    # --- public API --------------------------------------------------------
    def present(self, title: str, content: QWidget) -> None:
        self._title.setText(title)
        self._scroll.setWidget(content)

        hint = content.sizeHint().height()
        chrome = 84  # grabber + header + paddings
        max_h = int(self._host.height() * 0.85)
        self._panel_height = max(200, min(hint + chrome, max_h))

        self.setGeometry(0, 0, self._host.width(), self._host.height())
        self._scrim.setGeometry(0, 0, self.width(), self.height())
        self._panel.resize(self.width(), self._panel_height)

        self.show()
        self.raise_()
        start = QPoint(0, self.height())
        end = QPoint(0, self.height() - self._panel_height)
        self._panel.move(start)
        self._anim.stop()
        if self._dismiss_connected:
            self._anim.finished.disconnect(self._on_dismissed)
            self._dismiss_connected = False
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    def dismiss(self) -> None:
        if not self.isVisible():
            return
        end = QPoint(0, self.height())
        self._anim.stop()
        self._anim.setStartValue(self._panel.pos())
        self._anim.setEndValue(end)
        if not self._dismiss_connected:
            self._anim.finished.connect(self._on_dismissed)
            self._dismiss_connected = True
        self._anim.start()

    def _on_dismissed(self) -> None:
        if self._dismiss_connected:
            self._anim.finished.disconnect(self._on_dismissed)
            self._dismiss_connected = False
        self.hide()
        self.closed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self.isVisible():
            self.setGeometry(0, 0, self._host.width(), self._host.height())
            self._scrim.setGeometry(0, 0, self.width(), self.height())
            self._panel.resize(self.width(), self._panel_height)
            # Don't fight an in-flight slide animation by hard-moving the panel.
            if self._anim.state() != QAbstractAnimation.State.Running:
                self._panel.move(0, self.height() - self._panel_height)


class Toast(QWidget):
    """Small auto-dismissing message near the bottom of the window."""

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(28,28,30,235);"
            "  color: #FFFFFF;"
            "  border-radius: 18px;"
            "  padding: 11px 20px;"
            "  font-size: 13px;"
            "}"
        )
        card_shadow(self._label, blur=24, dy=4, alpha=80)

        self._effect = QGraphicsOpacityEffect(self._label)
        self._label.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(180)
        self._hide_connected = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start_fade_out)

    def show_message(self, text: str, msec: int = 2200) -> None:
        self._label.setText(text)
        self._label.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._effect.setOpacity(0.0)
        self._fade.stop()
        if self._hide_connected:
            self._fade.finished.disconnect(self.hide)
            self._hide_connected = False
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._timer.start(msec)

    def _start_fade_out(self) -> None:
        self._fade.stop()
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        if not self._hide_connected:
            self._fade.finished.connect(self.hide)
            self._hide_connected = True
        self._fade.start()

    def _reposition(self) -> None:
        w = self._label.width()
        h = self._label.height()
        x = (self._host.width() - w) // 2
        y = self._host.height() - h - 96
        self.setGeometry(x, y, w, h)
        self._label.move(0, 0)
