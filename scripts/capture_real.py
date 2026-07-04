"""Capture the app loaded from a REAL folder of TIFFs (offscreen)."""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tifffile
from PySide6.QtWidgets import QApplication

from cellscope import config, theme
from cellscope.data.folder_loader import FolderLoader

OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def pump(app, s=0.25):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(app, pred, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


def blob(cx, cy, amp=4500):
    yy, xx = np.mgrid[0:256, 0:256]
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 9.0 ** 2)))
    return (amp * g).astype(np.uint16)


def main():
    config.set_value("first_run_done", True)
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app, "light")
    from cellscope.app import MainWindow

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(3)
        centers = rng.uniform(40, 216, size=(12, 2))
        for t in range(10):
            frame = np.full((256, 256), 60, dtype=np.uint16)
            for i, (cy, cx) in enumerate(centers):
                frame = np.maximum(frame, blob(cx + 1.5 * t, cy + (i % 3 - 1) * t))
            tifffile.imwrite(str(Path(td) / f"A1_t{t:02d}.tif"), frame)

        window = MainWindow()
        window.resize(460, 880)
        window.show()
        pump(app, 0.3)
        window.state.set_loader(FolderLoader(td))
        wait(app, lambda: window.state.current_array is not None)
        pump(app, 0.3)

        window.show_viewer()
        window.state.start_analysis(window.state.current_well_id)
        wait(app, lambda: window.state.analysis_for(window.state.current_well_id) is not None)
        window._viewer._refresh_canvas()
        pump(app, 0.3)
        window.grab().save(str(OUT / "real_viewer.png"), "PNG")
        print("saved real_viewer.png; channels:", window.state.loader.channel_names)
        window.close()


if __name__ == "__main__":
    main()
