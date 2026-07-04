"""Capture the ruler line on the canvas and the Set-scale sheet (offscreen)."""

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from cellscope import config, theme

OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def pump(app, s=0.3):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(app, pred, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


def main():
    config.set_value("first_run_done", True)
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app, "light")
    from cellscope.app import MainWindow

    window = MainWindow()
    window.resize(460, 900)
    window.show()
    pump(app, 0.3)
    st = window.state
    wait(app, lambda: st.current_array is not None)
    window.show_viewer()
    pump(app, 0.4)

    viewer = window._viewer
    canvas = viewer._canvas
    viewer._toggle_ruler()  # enter ruler mode

    # Simulate a drawn line in image coordinates.
    a, b = QPointF(120, 150), QPointF(420, 330)
    canvas._ruler_start = a
    canvas._ruler_end = b
    canvas.update()
    pump(app, 0.3)
    window.grab().save(str(OUT / "ruler_line.png"), "PNG")

    length = math.hypot(b.x() - a.x(), b.y() - a.y())
    viewer._on_ruler_measured(length)
    pump(app, 0.5)
    window.grab().save(str(OUT / "ruler_sheet.png"), "PNG")
    print(f"saved ruler_line.png, ruler_sheet.png; line={length:.0f}px")
    window.close()


if __name__ == "__main__":
    main()
