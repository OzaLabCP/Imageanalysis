"""Capture the plate-map editor and the gallery with conditions (offscreen)."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    pump(app, 0.4)

    st.set_condition(["A1", "A2"], "PURExpress + sfGFP")
    st.set_condition(["A3"], "PURExpress + CX43-GFP")
    st.set_condition(["B1", "B2"], "Cytosols (+control)")
    pump(app, 0.2)

    window.show_wells()
    pump(app, 0.3)
    window.grab().save(str(OUT / "pm_wells.png"), "PNG")

    window._wells._open_platemap()
    pump(app, 0.5)
    # select a couple wells to show selection state
    window._sheet._scroll.widget()  # ensure built
    editor = window._sheet._scroll.widget()
    editor._toggle_well("B3")
    editor._toggle_well("A3")
    pump(app, 0.3)
    window.grab().save(str(OUT / "pm_editor.png"), "PNG")
    print("saved pm_wells.png, pm_editor.png; conditions:", st.distinct_conditions())
    window.close()


if __name__ == "__main__":
    main()
