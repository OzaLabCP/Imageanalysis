"""Load a real folder into the app, render raw + detected views, print stats.

Usage:  python scripts/capture_dataset.py "<folder>"
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from cellscope import config, theme  # noqa: E402
from cellscope.analysis import AnalysisSettings  # noqa: E402
from cellscope.data.folder_loader import FolderLoader  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def pump(app, s=0.25):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(app, pred, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    return False


def main():
    folder = sys.argv[1]
    config.set_value("first_run_done", True)
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app, "light")
    from cellscope.app import MainWindow

    window = MainWindow()
    window.resize(460, 900)
    window.show()
    pump(app, 0.3)

    state = window.state
    state.set_loader(FolderLoader(folder))
    assert wait(app, lambda: state.current_array is not None), "well never loaded"
    well = state.current_well_id
    info = state.current_well_info()
    print(f"well {well}: T={info.n_time} C={info.n_channels} {info.height}x{info.width}")

    window.show_viewer()
    state.set_t(info.n_time // 2)
    pump(app, 0.4)
    window.grab().save(str(OUT / "ds_raw.png"), "PNG")

    t0 = time.time()
    state.start_analysis(well)
    ok = wait(app, lambda: state.analysis_for(well) is not None, timeout=240)
    print(f"analysis finished={ok} in {time.time() - t0:.1f}s")
    if ok:
        wa = state.analysis_for(well)
        print("n_tracks:", wa.n_tracks)
        print("counts_per_frame:", wa.counts_per_frame.tolist())
        window._viewer._refresh_canvas()
        pump(app, 0.4)
        window.grab().save(str(OUT / "ds_detected.png"), "PNG")
    window.close()


if __name__ == "__main__":
    main()
