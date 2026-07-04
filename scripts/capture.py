"""Render real screenshots of the app offscreen (no display needed).

Saves PNGs of each tab in both light and dark themes to .preview/.
Run:  python scripts/capture.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from cellscope import config, theme  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def pump(app, seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_until(app, predicate, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def capture_theme(app, mode: str) -> None:
    theme.apply_theme(app, mode)
    from cellscope.app import MainWindow

    window = MainWindow()
    window.resize(460, 880)
    window.show()
    pump(app, 0.3)

    state = window.state
    wait_until(app, lambda: state.current_array is not None)
    pump(app, 0.4)  # let thumbnails render

    window.show_wells()
    pump(app, 0.3)
    window.grab().save(str(OUT / f"{mode}_1_wells.png"), "PNG")

    window._wells._open_well(state.wells[0].well_id)
    window.show_viewer()
    state.set_t(8)
    pump(app, 0.2)
    state.start_analysis(state.wells[0].well_id)
    wait_until(app, lambda: state.analysis_for(state.wells[0].well_id) is not None)
    window._viewer._refresh_canvas()
    pump(app, 0.3)
    window.grab().save(str(OUT / f"{mode}_2_viewer.png"), "PNG")

    state.set_selected_track(window._viewer.state.analysis_for(state.wells[0].well_id).track_ids()[0])
    window.show_cells()
    pump(app, 0.3)
    window.grab().save(str(OUT / f"{mode}_3_cells.png"), "PNG")

    window.show_results()
    pump(app, 0.3)
    window.grab().save(str(OUT / f"{mode}_4_results.png"), "PNG")

    window.close()
    print(f"[{mode}] saved 4 screenshots")


def main() -> int:
    config.set_value("first_run_done", True)
    app = QApplication.instance() or QApplication(sys.argv)
    capture_theme(app, "light")
    capture_theme(app, "dark")
    print(f"Screenshots in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
