"""Centralized theming: light + dark palettes and the Qt Style Sheet.

Custom-painted widgets (tab bar, switches, chart, canvas) read colors from
``palette()``; standard widgets are styled by the QSS string. Retheme the whole
app by editing this one module.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication

LIGHT: dict[str, str] = {
    "window_bg": "#F2F2F7",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F7FB",
    "surface_sunken": "#ECECF1",
    "text": "#1C1C1E",
    "text_secondary": "#8A8A8E",
    "separator": "#E3E3E8",
    "accent": "#007AFF",
    "accent_text": "#FFFFFF",
    "danger": "#FF3B30",
    "success": "#34C759",
    "canvas_bg": "#0B0B0F",
    "tabbar_bg": "#FBFBFD",
    "tabbar_inactive": "#9A9AA0",
    "slider_groove": "#D8D8DE",
    "slider_handle": "#FFFFFF",
    "scrim": "rgba(0, 0, 0, 90)",
    "chart_grid": "#E6E6EB",
    "shadow": "rgba(0, 0, 0, 38)",
}

DARK: dict[str, str] = {
    "window_bg": "#000000",
    "surface": "#1C1C1E",
    "surface_alt": "#242426",
    "surface_sunken": "#2C2C2E",
    "text": "#FFFFFF",
    "text_secondary": "#98989F",
    "separator": "#2E2E31",
    "accent": "#0A84FF",
    "accent_text": "#FFFFFF",
    "danger": "#FF453A",
    "success": "#30D158",
    "canvas_bg": "#000000",
    "tabbar_bg": "#101012",
    "tabbar_inactive": "#7C7C82",
    "slider_groove": "#3A3A3C",
    "slider_handle": "#FFFFFF",
    "scrim": "rgba(0, 0, 0, 130)",
    "chart_grid": "#2A2A2D",
    "shadow": "rgba(0, 0, 0, 120)",
}

_state = {"mode": "light", "palette": LIGHT}


def palette() -> dict[str, str]:
    """The currently active color palette."""
    return _state["palette"]


def current_mode() -> str:
    return _state["mode"]


def color(key: str) -> QColor:
    return QColor(_state["palette"][key])


def resolve_mode(mode: str, app: QApplication | None = None) -> str:
    """Resolve "system" to "light"/"dark" using the OS color scheme if available."""
    if mode in ("light", "dark"):
        return mode
    try:
        scheme = (app or QApplication.instance()).styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:
        pass
    return "light"


def apply_theme(app: QApplication, mode: str = "system") -> str:
    """Apply the theme to the whole application. Returns the resolved mode."""
    resolved = resolve_mode(mode, app)
    _state["mode"] = resolved
    _state["palette"] = DARK if resolved == "dark" else LIGHT

    font = QFont()
    font.setPointSizeF(10.5)
    app.setFont(font)

    app.setStyleSheet(build_qss(_state["palette"]))
    return resolved


def build_qss(p: dict[str, str]) -> str:
    return f"""
* {{
    outline: 0;
}}

QWidget {{
    background-color: {p['window_bg']};
    color: {p['text']};
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
}}

QToolTip {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['separator']};
    border-radius: 8px;
    padding: 6px 8px;
}}

/* --- cards / surfaces ------------------------------------------------- */
QFrame#Card {{
    background-color: {p['surface']};
    border: 1px solid {p['separator']};
    border-radius: 16px;
}}
QFrame#Sheet {{
    background-color: {p['surface']};
    border-top-left-radius: 22px;
    border-top-right-radius: 22px;
    border: none;
}}
QWidget#Scrim {{
    background-color: {p['scrim']};
}}

/* --- headings --------------------------------------------------------- */
QLabel#Title {{
    font-size: 26px;
    font-weight: 700;
    color: {p['text']};
}}
QLabel#Subtitle {{
    font-size: 13px;
    color: {p['text_secondary']};
}}
QLabel#SectionLabel {{
    font-size: 12px;
    font-weight: 600;
    color: {p['text_secondary']};
}}
QLabel#CardTitle {{
    font-size: 15px;
    font-weight: 600;
}}
QLabel#Hint {{
    color: {p['text_secondary']};
    font-size: 13px;
}}

/* --- buttons ---------------------------------------------------------- */
QPushButton {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['separator']};
    border-radius: 12px;
    padding: 9px 16px;
    min-height: 26px;
    font-size: 14px;
}}
QPushButton:hover {{
    background-color: {p['surface_alt']};
}}
QPushButton:pressed {{
    background-color: {p['surface_sunken']};
}}
QPushButton:disabled {{
    color: {p['text_secondary']};
}}

QPushButton#Primary {{
    background-color: {p['accent']};
    color: {p['accent_text']};
    border: none;
    font-weight: 600;
    padding: 11px 18px;
}}
QPushButton#Primary:hover {{
    background-color: {p['accent']};
}}
QPushButton#Primary:disabled {{
    background-color: {p['slider_groove']};
    color: {p['text_secondary']};
}}

QPushButton#Ghost {{
    background-color: transparent;
    border: none;
    color: {p['accent']};
    font-weight: 600;
}}
QPushButton#Ghost:hover {{
    background-color: {p['surface_alt']};
}}

QPushButton#Danger {{
    background-color: transparent;
    border: 1px solid {p['danger']};
    color: {p['danger']};
}}

/* --- sliders ---------------------------------------------------------- */
QSlider::groove:horizontal {{
    height: 6px;
    background: {p['slider_groove']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {p['accent']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {p['slider_handle']};
    border: 1px solid {p['separator']};
    width: 24px;
    height: 24px;
    margin: -10px 0;
    border-radius: 12px;
}}
QSlider::handle:horizontal:hover {{
    border: 1px solid {p['accent']};
}}
QSlider:disabled {{
}}
QSlider::sub-page:horizontal:disabled {{
    background: {p['slider_groove']};
}}

/* --- tables ----------------------------------------------------------- */
QTableWidget, QTableView {{
    background-color: {p['surface']};
    alternate-background-color: {p['surface_alt']};
    border: 1px solid {p['separator']};
    border-radius: 14px;
    gridline-color: transparent;
    selection-background-color: {p['accent']};
    selection-color: {p['accent_text']};
    font-size: 13px;
}}
QTableView::item {{
    padding: 6px 8px;
    border: none;
}}
QHeaderView::section {{
    background-color: {p['surface_alt']};
    color: {p['text_secondary']};
    border: none;
    border-bottom: 1px solid {p['separator']};
    padding: 8px;
    font-weight: 600;
    font-size: 12px;
}}
QTableCornerButton::section {{
    background-color: {p['surface_alt']};
    border: none;
}}

/* --- lists ------------------------------------------------------------ */
QListWidget {{
    background-color: transparent;
    border: none;
}}

/* --- text input ------------------------------------------------------- */
QLineEdit {{
    background-color: {p['surface_sunken']};
    color: {p['text']};
    border: 1px solid {p['separator']};
    border-radius: 10px;
    padding: 9px 12px;
    min-height: 22px;
    selection-background-color: {p['accent']};
    selection-color: {p['accent_text']};
}}
QLineEdit:focus {{
    border: 1px solid {p['accent']};
}}

/* --- combo box -------------------------------------------------------- */
QComboBox {{
    background-color: {p['surface']};
    border: 1px solid {p['separator']};
    border-radius: 10px;
    padding: 6px 12px;
    min-height: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {p['surface']};
    border: 1px solid {p['separator']};
    border-radius: 10px;
    selection-background-color: {p['accent']};
    selection-color: {p['accent_text']};
}}

/* --- scroll areas ----------------------------------------------------- */
QScrollArea {{
    background-color: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p['slider_groove']};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {p['tabbar_inactive']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {p['slider_groove']};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QProgressBar {{
    border: none;
    background-color: {p['slider_groove']};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {p['accent']};
    border-radius: 4px;
}}
"""
