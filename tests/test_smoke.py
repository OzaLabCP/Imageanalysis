"""End-to-end smoke test of the whole app, headless (Qt 'offscreen' platform).

Drives the real workflow: load demo plate -> open a well -> scrub time ->
detect cells (on a worker thread) -> inspect overlays/cells -> export CSV + PNG.

Run:  python tests/test_smoke.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from cellscope import config, theme  # noqa: E402


def wait_until(app, predicate, timeout=20.0):
    start = time.time()
    while time.time() - start < timeout:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def main() -> int:
    # Avoid the first-run welcome timer interfering with the headless run.
    config.set_value("first_run_done", True)

    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app, "light")

    # compose_rgb pulls in Qt (QImage), so it lives here rather than in the
    # Qt-free analysis suite. A NaN pixel must not black out the whole channel.
    from cellscope.render import compose_rgb
    nan_frame = np.zeros((1, 16, 16), dtype=np.float32)
    nan_frame[0, 4:12, 4:12] = 200.0
    nan_frame[0, 0, 0] = np.nan
    rgb = compose_rgb(nan_frame, [(255, 255, 255)], [True], 0.5, 0.5)
    assert rgb.dtype == np.uint8 and rgb.max() > 0
    print("[ok] compose_rgb tolerates NaN pixels")

    from cellscope.app import MainWindow

    window = MainWindow()
    window.resize(480, 860)
    window.show()
    app.processEvents()

    state = window.state
    assert state.loader is not None
    assert len(state.wells) == 6, state.wells
    first_well = state.wells[0].well_id

    # Wait for the first well's pixel array to load on the worker thread.
    assert wait_until(app, lambda: state.current_array is not None), "well never loaded"
    assert state.current_array.shape == (20, 1, 2, 512, 512)
    print(f"[ok] demo plate loaded; current well {state.current_well_id}")

    # Open a well from the gallery and switch to the Viewer.
    window._wells._open_well(first_well)
    window.show_viewer()
    app.processEvents()

    # Scrub time and render the canvas.
    state.set_t(5)
    app.processEvents()
    canvas_pix = window._viewer._canvas.grab()
    assert canvas_pix.width() > 0 and canvas_pix.height() > 0
    print("[ok] viewer renders a frame")

    # Run detection on a background thread and wait for results.
    state.start_analysis(first_well)
    assert wait_until(app, lambda: state.analysis_for(first_well) is not None, timeout=30.0), \
        "analysis never finished"
    wa = state.analysis_for(first_well)
    assert wa.n_tracks >= 10, wa.n_tracks
    print(f"[ok] detection found {wa.n_tracks} cells across {wa.n_time} frames")

    # Overlays should now be present on the canvas for the current frame.
    window._viewer._refresh_canvas()
    app.processEvents()
    assert window._viewer._canvas._label_image is not None
    assert window._viewer._canvas._tracks

    # Select a cell (as if tapped) and confirm Cells tab builds a detail view.
    some_track = wa.track_ids()[0]
    state.set_selected_track(some_track)
    window.show_cells()
    app.processEvents()
    assert window._cells._rows, "cells list empty"
    print(f"[ok] cells list shows {len(window._cells._rows)} cells; detail for cell {some_track}")

    # Results tab: chart + table populated.
    window.show_results()
    app.processEvents()
    results = window._results
    assert results._table.rowCount() == len(wa.measurements), (
        results._table.rowCount(), len(wa.measurements))
    assert results._chart._series, "chart has no series"
    print(f"[ok] results table has {results._table.rowCount()} rows")

    # Ruler calibration rescales existing measurements (area x4, lengths x2).
    a_before = sum(m.area_um2 for m in wa.measurements)
    f_before = sum(m.feret_diameter_um for m in wa.measurements)
    p_before = sum(m.perimeter_um for m in wa.measurements)
    maj_before = sum(m.length_major_um for m in wa.measurements)
    state.set_pixel_size_um(state.pixel_size_um * 2.0)
    a_after = sum(m.area_um2 for m in wa.measurements)
    f_after = sum(m.feret_diameter_um for m in wa.measurements)
    p_after = sum(m.perimeter_um for m in wa.measurements)
    maj_after = sum(m.length_major_um for m in wa.measurements)
    assert abs(a_after / a_before - 4.0) < 0.01, (a_before, a_after)
    assert abs(f_after / f_before - 2.0) < 0.01, (f_before, f_after)
    assert abs(p_after / p_before - 2.0) < 0.01, (p_before, p_after)
    assert abs(maj_after / maj_before - 2.0) < 0.01, (maj_before, maj_after)
    print("[ok] ruler calibration rescaled measurements (area x4, lengths x2)")

    # Race: calibrate WHILE an analysis is in flight. done() must reconcile the
    # freshly-computed result to the current scale. Deterministic because the
    # worker's done() callback is queued and cannot run until we pump events.
    state.set_pixel_size_um(1.0)
    third = state.wells[2].well_id
    state.start_analysis(third)           # dispatched at scale 1.0
    state.set_pixel_size_um(2.0)          # calibrate before done() is delivered
    assert wait_until(app, lambda: state.analysis_for(third) is not None, timeout=30)
    assert abs(state.analysis_for(third).pixel_size_um - 2.0) < 1e-9, \
        state.analysis_for(third).pixel_size_um
    for _wid, r in state.results.items():
        assert abs(r.pixel_size_um - state.pixel_size_um) < 1e-9, (_wid, r.pixel_size_um)
    print("[ok] calibration-during-analysis race reconciled")

    # Assign a plate-map condition and confirm it flows into the export.
    state.set_condition([first_well], "Treatment A")
    assert state.condition_of(first_well) == "Treatment A"
    assert state.distinct_conditions() == ["Treatment A"]

    # Export CSV + PNG (bypassing the file dialog) and verify the files.
    with tempfile.TemporaryDirectory() as td:
        csv_path = str(Path(td) / "measurements.csv")
        png_path = str(Path(td) / "chart.png")
        results.write_csv(csv_path, [first_well])
        ok_png = results.write_png(png_path)
        assert Path(csv_path).exists() and Path(csv_path).stat().st_size > 0
        lines = Path(csv_path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(wa.measurements) + 1, (len(lines), len(wa.measurements))
        header = lines[0].split(",")
        assert "condition" in header and "feret_diameter_um" in header, header
        cond_col = header.index("condition")
        assert lines[1].split(",")[cond_col] == "Treatment A", lines[1]
        assert ok_png and Path(png_path).exists() and Path(png_path).stat().st_size > 0
        print(f"[ok] exported CSV ({len(lines)} lines, condition+diameter cols) "
              f"and PNG ({Path(png_path).stat().st_size} bytes)")

    # Exercise a second well + comparison selection.
    second = state.wells[1].well_id
    state.toggle_well_selected(first_well)
    state.toggle_well_selected(second)
    state.start_analysis(second)
    assert wait_until(app, lambda: state.analysis_for(second) is not None, timeout=30.0)
    results._rebuild()
    app.processEvents()
    assert len(results._chart._series) >= 2, "comparison chart should overlay wells"
    print("[ok] two-well comparison chart built")

    # Fast preview: shrink the target so the 512px demo triggers 1/2 downsampling.
    state.preview_target = 200
    state.set_preview(True)
    assert state.current_downsample() == 2, state.current_downsample()
    fourth = state.wells[3].well_id
    state.set_current_well(fourth)
    assert wait_until(app, lambda: state.current_array is not None
                      and state.current_array.shape[-1] == 256), \
        (None if state.current_array is None else state.current_array.shape)
    state.start_analysis(fourth)
    assert wait_until(app, lambda: state.analysis_for(fourth) is not None, timeout=30)
    assert state.analysis_for(fourth).downsample == 2
    state.set_preview(False)
    state.preview_target = 700
    print("[ok] fast preview: 1/2 load + analysis, restored to full")

    # --- real TIFF folder through the same UI path -----------------------
    import tifffile
    from cellscope.data.folder_loader import FolderLoader

    with tempfile.TemporaryDirectory() as td:
        yy, xx = np.mgrid[0:64, 0:64]
        for frame in range(6):
            blob = np.exp(-(((xx - (18 + 3 * frame)) ** 2 + (yy - 32) ** 2) / (2 * 7.0 ** 2)))
            tifffile.imwrite(str(Path(td) / f"A1_t{frame}.tif"),
                             (4000 * blob + 50).astype(np.uint16))
        state.set_loader(FolderLoader(td))
        real_well = state.wells[0].well_id
        assert wait_until(app, lambda: state.current_array is not None), "real well never loaded"
        assert state.current_array.shape[0] == 6, state.current_array.shape
        window.show_wells()
        app.processEvents()
        assert window._wells._cards, "gallery did not rebuild for real folder"

        state.start_analysis(real_well)
        assert wait_until(app, lambda: state.analysis_for(real_well) is not None, timeout=30.0)
        assert state.analysis_for(real_well).n_tracks >= 1
        print(f"[ok] real TIFF folder loaded + analyzed ({state.analysis_for(real_well).n_tracks} cell)")

    window.close()
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
