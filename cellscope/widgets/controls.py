"""iOS-style building blocks: cards, switches, segmented controls, sliders, FAB."""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cellscope import theme
from cellscope.widgets import icons


def card_shadow(widget: QWidget, blur: int = 26, dy: int = 6, alpha: int = 38) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setXOffset(0)
    effect.setYOffset(dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


class Card(QFrame):
    """Rounded surface container with a soft shadow."""

    def __init__(self, parent: QWidget | None = None, shadow: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        if shadow:
            card_shadow(self)


def make_button(text: str, kind: str = "default") -> QPushButton:
    btn = QPushButton(text)
    if kind == "primary":
        btn.setObjectName("Primary")
    elif kind == "ghost":
        btn.setObjectName("Ghost")
    elif kind == "danger":
        btn.setObjectName("Danger")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setMinimumHeight(44)
    return btn


class IconButton(QPushButton):
    """Square button that paints a vector icon (no text)."""

    def __init__(self, icon_name: str, size: int = 44, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._d = size
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)

    def set_icon_name(self, name: str) -> None:
        self._icon_name = name
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._d, self._d)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        if self.isDown():
            painter.setBrush(QBrush(theme.color("surface_sunken")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 10, 10)
        col = theme.color("text") if self.isEnabled() else theme.color("text_secondary")
        icons.paint_icon(painter, self._icon_name, rect, col, weight=2.2)


class ToggleSwitch(QAbstractButton):
    """Animated iOS toggle. Checkable; emits ``toggled(bool)``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(52, 32)
        self._offset = 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = theme.color("accent")
        off = theme.color("slider_groove")
        track = QColor(on if self._offset > 0.001 else off)
        if 0.0 < self._offset < 1.0:
            track = QColor(
                int(off.red() + (on.red() - off.red()) * self._offset),
                int(off.green() + (on.green() - off.green()) * self._offset),
                int(off.blue() + (on.blue() - off.blue()) * self._offset),
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(track))
        painter.drawRoundedRect(r, r.height() / 2, r.height() / 2)

        d = r.height() - 4
        x = r.left() + 2 + self._offset * (r.width() - d - 4)
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(QRectF(x, r.top() + 2, d, d))


class SegmentedControl(QWidget):
    """A horizontal pill segmented control. Emits ``currentChanged(int)``."""

    currentChanged = Signal(int)

    def __init__(self, segments: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._segments = list(segments)
        self._current = 0
        self.setMinimumHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_segments(self, segments: list[str]) -> None:
        self._segments = list(segments)
        if self._current >= len(self._segments):
            self._current = 0
        self.update()

    def current_index(self) -> int:
        return self._current

    def set_current_index(self, index: int, emit: bool = True) -> None:
        index = max(0, min(index, len(self._segments) - 1))
        if index == self._current:
            return
        self._current = index
        self.update()
        if emit:
            self.currentChanged.emit(index)

    def _seg_width(self) -> float:
        n = max(1, len(self._segments))
        return self.width() / n

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._segments:
            return
        idx = int(event.position().x() // self._seg_width())
        self.set_current_index(idx)

    def sizeHint(self) -> QSize:
        return QSize(220, 38)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = r.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(theme.color("surface_sunken")))
        painter.drawRoundedRect(r, radius, radius)

        if not self._segments:
            return
        seg_w = self._seg_width()
        pill = QRectF(self._current * seg_w + 3, r.top() + 3, seg_w - 6, r.height() - 6)
        painter.setBrush(QBrush(theme.color("accent")))
        painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)

        font = QFont(self.font())
        font.setPointSizeF(11.0)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        for i, text in enumerate(self._segments):
            rect = QRectF(i * seg_w, r.top(), seg_w, r.height())
            painter.setPen(QPen(theme.color("accent_text") if i == self._current
                                else theme.color("text")))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class LabeledSlider(QWidget):
    """A titled slider with optional qualitative end captions. Value is 0..1.

    Emits ``valueChanged(float)``.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        title: str,
        left_caption: str = "",
        right_caption: str = "",
        value: float = 0.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel(title)
        self._title.setObjectName("SectionLabel")
        top.addWidget(self._title)
        top.addStretch(1)
        layout.addLayout(top)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(int(value * 1000))
        self._slider.valueChanged.connect(
            lambda v: self.valueChanged.emit(v / 1000.0)
        )
        layout.addWidget(self._slider)

        if left_caption or right_caption:
            caps = QHBoxLayout()
            caps.setContentsMargins(0, 0, 0, 0)
            lc = QLabel(left_caption)
            lc.setObjectName("Hint")
            rc = QLabel(right_caption)
            rc.setObjectName("Hint")
            rc.setAlignment(Qt.AlignmentFlag.AlignRight)
            caps.addWidget(lc)
            caps.addStretch(1)
            caps.addWidget(rc)
            layout.addLayout(caps)

    def value(self) -> float:
        return self._slider.value() / 1000.0

    def set_value(self, value: float) -> None:
        self._slider.setValue(int(max(0.0, min(1.0, value)) * 1000))

    def slider(self) -> QSlider:
        return self._slider


class Header(QWidget):
    """Large screen title with optional subtitle and a right-aligned action slot."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 8)
        layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        self._title = QLabel(title)
        self._title.setObjectName("Title")
        text_col.addWidget(self._title)
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("Subtitle")
        self._subtitle.setVisible(bool(subtitle))
        text_col.addWidget(self._subtitle)
        layout.addLayout(text_col)
        layout.addStretch(1)

        self._action_slot = QHBoxLayout()
        self._action_slot.setSpacing(8)
        layout.addLayout(self._action_slot)

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
        self._subtitle.setVisible(bool(text))

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def add_action(self, widget: QWidget) -> None:
        self._action_slot.addWidget(widget)


class Fab(QPushButton):
    """Floating primary action button. Position it manually in the parent's
    ``resizeEvent`` (it paints its own pill + shadow)."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Primary")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(52)
        font = QFont(self.font())
        font.setPointSizeF(14.0)
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        self.setStyleSheet(
            "QPushButton#Primary {"
            "  border-radius: 26px;"
            "  padding-left: 26px; padding-right: 26px;"
            "}"
        )
        card_shadow(self, blur=30, dy=8, alpha=70)
