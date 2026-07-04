"""Plate-map editor: tap wells on a plate grid and assign condition names.

The conditions flow into the Wells gallery, the Results grouping, and the CSV
export (a ``condition`` column), so output can be grouped by treatment.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellscope import theme
from cellscope.widgets.controls import make_button


class WellChip(QWidget):
    clicked = Signal(str)

    def __init__(self, well_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.well_id = well_id
        self._condition = ""
        self._color = None
        self._selected = False
        self.setFixedSize(62, 48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, condition: str, color, selected: bool) -> None:
        self._condition = condition
        self._color = color
        self._selected = selected
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.well_id)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)

        if self._color is not None:
            base = QColor(*self._color)
            fill = QColor(base)
            fill.setAlpha(60)
            p.setBrush(fill)
            p.setPen(QPen(base, 1.4))
        else:
            p.setBrush(theme.color("surface_sunken"))
            p.setPen(QPen(theme.color("separator"), 1.2))
        p.drawRoundedRect(r, 9, 9)

        if self._selected:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(theme.color("accent"), 2.4))
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 10, 10)

        f = QFont(self.font())
        f.setPointSizeF(10.0)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QPen(theme.color("text")))
        if self._condition:
            p.drawText(QRectF(r.left(), r.top() + 2, r.width(), r.height() * 0.55),
                       Qt.AlignmentFlag.AlignCenter, self.well_id)
            f2 = QFont(self.font())
            f2.setPointSizeF(7.5)
            p.setFont(f2)
            p.setPen(QPen(theme.color("text_secondary")))
            elided = p.fontMetrics().elidedText(
                self._condition, Qt.TextElideMode.ElideRight, int(r.width() - 6))
            p.drawText(QRectF(r.left() + 3, r.center().y(), r.width() - 6, r.height() * 0.45),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, elided)
        else:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.well_id)


class ConditionChip(QPushButton):
    def __init__(self, name: str, color, parent: QWidget | None = None) -> None:
        super().__init__(name, parent)
        self.name = name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(38)
        pal = theme.palette()
        self.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            f"  background-color: {pal['surface']};"
            f"  border: 1px solid {pal['separator']};"
            f"  border-left: 7px solid rgb{tuple(color)};"
            "  border-radius: 9px; padding: 7px 12px; font-weight: 600;"
            "}"
            f"QPushButton:hover {{ background-color: {pal['surface_alt']}; }}"
        )


class PlateMapEditor(QWidget):
    def __init__(self, state, shell, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.shell = shell
        self._selected: set[str] = set()
        self._chips: dict[str, WellChip] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 4, 18, 22)
        root.setSpacing(14)

        intro = QLabel("Tap wells to select them, then assign a condition. "
                       "Conditions are saved to your exported data.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        root.addWidget(intro)

        sel_row = QHBoxLayout()
        self._sel_label = QLabel("0 wells selected")
        self._sel_label.setObjectName("SectionLabel")
        sel_row.addWidget(self._sel_label)
        sel_row.addStretch(1)
        all_btn = make_button("Select all", "ghost")
        all_btn.setMinimumHeight(34)
        all_btn.clicked.connect(self._select_all)
        none_btn = make_button("Clear selection", "ghost")
        none_btn.setMinimumHeight(34)
        none_btn.clicked.connect(self._clear_selection)
        sel_row.addWidget(all_btn)
        sel_row.addWidget(none_btn)
        root.addLayout(sel_row)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)
        root.addWidget(self._grid_host, 0, Qt.AlignmentFlag.AlignHCenter)

        assign_label = QLabel("Assign condition")
        assign_label.setObjectName("SectionLabel")
        root.addWidget(assign_label)

        new_row = QHBoxLayout()
        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("New condition name (e.g. PURExpress + sfGFP)")
        self._name_field.returnPressed.connect(self._apply_new)
        new_row.addWidget(self._name_field, 1)
        self._apply_btn = make_button("Apply", "primary")
        self._apply_btn.setMinimumHeight(40)
        self._apply_btn.clicked.connect(self._apply_new)
        new_row.addWidget(self._apply_btn)
        root.addLayout(new_row)

        self._existing_label = QLabel("Existing conditions (tap to apply)")
        self._existing_label.setObjectName("Hint")
        root.addWidget(self._existing_label)
        self._existing_host = QWidget()
        self._existing_grid = QGridLayout(self._existing_host)
        self._existing_grid.setContentsMargins(0, 0, 0, 0)
        self._existing_grid.setHorizontalSpacing(8)
        self._existing_grid.setVerticalSpacing(8)
        root.addWidget(self._existing_host)

        self._remove_btn = make_button("Remove condition from selected", "danger")
        self._remove_btn.clicked.connect(self._remove)
        root.addWidget(self._remove_btn)

        done = make_button("Done", "ghost")
        done.clicked.connect(self.shell.dismiss_sheet)
        root.addWidget(done, 0, Qt.AlignmentFlag.AlignRight)

        state.conditionsChanged.connect(self._refresh)
        self._build_grid()
        self._refresh()
        self._update_selection_ui()

    # --- grid -------------------------------------------------------------
    def _build_grid(self) -> None:
        wells = self.state.wells
        if not wells:
            return
        rows = sorted({w.row for w in wells})
        cols = sorted({w.col for w in wells})
        row_pos = {r: i for i, r in enumerate(rows)}
        col_pos = {c: i for i, c in enumerate(cols)}

        # Column headers (numbers) and row headers (letters).
        for c in cols:
            lbl = QLabel(str(c + 1))
            lbl.setObjectName("Hint")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lbl, 0, col_pos[c] + 1)
        for r in rows:
            lbl = QLabel(chr(ord("A") + r) if r < 26 else str(r + 1))
            lbl.setObjectName("Hint")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lbl, row_pos[r] + 1, 0)

        for w in wells:
            chip = WellChip(w.well_id)
            chip.clicked.connect(self._toggle_well)
            self._chips[w.well_id] = chip
            self._grid.addWidget(chip, row_pos[w.row] + 1, col_pos[w.col] + 1)

    # --- interactions -----------------------------------------------------
    def _toggle_well(self, well_id: str) -> None:
        if well_id in self._selected:
            self._selected.discard(well_id)
        else:
            self._selected.add(well_id)
        self._refresh()
        self._update_selection_ui()

    def _select_all(self) -> None:
        self._selected = {w.well_id for w in self.state.wells}
        self._refresh()
        self._update_selection_ui()

    def _clear_selection(self) -> None:
        self._selected.clear()
        self._refresh()
        self._update_selection_ui()

    def _apply_new(self) -> None:
        name = self._name_field.text().strip()
        if not self._selected:
            self.shell.toast("Select wells first")
            return
        if not name:
            self.shell.toast("Type a condition name")
            return
        self.state.set_condition(self._selected, name)
        self._name_field.clear()

    def _apply_existing(self, name: str) -> None:
        if not self._selected:
            self.shell.toast("Select wells first")
            return
        self.state.set_condition(self._selected, name)

    def _remove(self) -> None:
        if not self._selected:
            self.shell.toast("Select wells first")
            return
        self.state.clear_conditions(self._selected)

    # --- refresh ----------------------------------------------------------
    def _update_selection_ui(self) -> None:
        n = len(self._selected)
        self._sel_label.setText(f"{n} well{'s' if n != 1 else ''} selected")
        self._apply_btn.setEnabled(n > 0)
        self._remove_btn.setEnabled(n > 0)

    def _refresh(self) -> None:
        for wid, chip in self._chips.items():
            cond = self.state.condition_of(wid)
            color = self.state.condition_color(cond) if cond else None
            chip.set_state(cond, color, wid in self._selected)
        self._rebuild_existing()

    def _rebuild_existing(self) -> None:
        while self._existing_grid.count():
            item = self._existing_grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        conditions = self.state.distinct_conditions()
        self._existing_label.setVisible(bool(conditions))
        for i, name in enumerate(conditions):
            chip = ConditionChip(name, self.state.condition_color(name))
            chip.clicked.connect(lambda _=False, n=name: self._apply_existing(n))
            self._existing_grid.addWidget(chip, i // 2, i % 2)

    def sizeHint(self) -> QSize:
        return QSize(420, 560)
