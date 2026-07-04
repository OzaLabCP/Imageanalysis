"""Headless checks for the analysis pipeline (no Qt / no display needed).

Run directly:  python tests/test_analysis.py
Or with pytest: pytest tests/test_analysis.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis import AnalysisSettings, run_analysis  # noqa: E402
from cellscope.analysis.segmentation import segment_frame  # noqa: E402
from cellscope.data.mock import MockLoader  # noqa: E402
from cellscope.render import compose_rgb  # noqa: E402


def test_mock_dataset_shape():
    loader = MockLoader()
    wells = loader.list_wells()
    assert [w.well_id for w in wells] == ["A1", "A2", "A3", "B1", "B2", "B3"]
    arr = loader.get_well("A1")
    assert arr.shape == (20, 1, 2, 512, 512)
    assert arr.dtype == np.uint16
    assert arr.min() >= 0
    assert arr.max() > 100  # cells are bright


def test_wells_differ():
    loader = MockLoader()
    a1 = loader.get_well("A1")
    a2 = loader.get_well("A2")
    assert not np.array_equal(a1, a2)


def test_pipeline_produces_real_results():
    loader = MockLoader()
    progress = []
    wa = run_analysis(loader, "A1", AnalysisSettings(), progress_cb=progress.append)

    # Found a plausible number of cells.
    assert wa.n_tracks >= 10
    assert len(wa.measurements) > 100
    assert wa.counts_per_frame.shape == (20,)
    assert wa.counts_per_frame.min() > 0

    # Progress is monotonic and finishes at 100.
    assert progress == sorted(progress)
    assert progress[-1] == 100

    # Track-ID label images are valued by track ID (0 = background).
    ids_in_image = set(np.unique(wa.track_label_images)) - {0}
    assert ids_in_image.issubset(set(wa.tracks.keys()))

    # The reporter channel ramps over time: for a long-lived cell, late-frame
    # mean reporter intensity should exceed early-frame.
    longest = max(wa.tracks.items(), key=lambda kv: len(kv[1]))[0]
    ms = sorted(wa.measurements_for_track(longest), key=lambda m: m.frame)
    assert len(ms) >= 10
    early = np.mean([m.mean_intensity[1] for m in ms[:3]])
    late = np.mean([m.mean_intensity[1] for m in ms[-3:]])
    assert late > early

    # Areas and diameters are positive in real units; Feret >= equivalent
    # diameter for EVERY cell (max caliper is never smaller than the
    # area-equivalent disk) - this invariant only holds with the edge-correct
    # Feret computation.
    assert all(m.area_um2 > 0 for m in ms)
    assert all(m.feret_diameter_um > 0 for m in ms)
    for m in wa.measurements:
        assert m.feret_diameter_um >= m.equiv_diameter_um - 1e-6, (
            m.feret_diameter_um, m.equiv_diameter_um)


def test_pixel_size_override_scales_measurements():
    loader = MockLoader()  # loader pixel size 0.65 um/px
    base = run_analysis(loader, "A1", AnalysisSettings())
    doubled = run_analysis(loader, "A1", AnalysisSettings(),
                           pixel_size_um=loader.pixel_size_um * 2.0)
    # Same segmentation -> pixel counts identical, but real-unit area scales x4
    # and diameters x2 when the pixel size doubles.
    assert sum(m.area_px for m in base.measurements) == \
        sum(m.area_px for m in doubled.measurements)
    a_base = sum(m.area_um2 for m in base.measurements)
    a_dbl = sum(m.area_um2 for m in doubled.measurements)
    assert abs(a_dbl / a_base - 4.0) < 0.01
    f_base = sum(m.feret_diameter_um for m in base.measurements)
    f_dbl = sum(m.feret_diameter_um for m in doubled.measurements)
    assert abs(f_dbl / f_base - 2.0) < 0.01


def test_downsample_preserves_real_units():
    loader = MockLoader()
    assert loader.get_well("A1", downsample=2).shape[-1] == \
        loader.get_well("A1").shape[-1] // 2

    full = run_analysis(loader, "A1", AnalysisSettings())
    preview = run_analysis(loader, "A1", AnalysisSettings(), downsample=2)
    assert preview.downsample == 2
    # Same cells, measured at 1/2 resolution with the pixel size scaled x2:
    # mean cell area in REAL microns stays in the same ballpark.
    fa = float(np.mean([m.area_um2 for m in full.measurements]))
    pa = float(np.mean([m.area_um2 for m in preview.measurements]))
    assert 0.6 < pa / fa < 1.6, (fa, pa)


def test_feret_diameter_of_known_shape():
    # A 21x21 solid square spans 21 px edge-to-edge; the true Feret (max caliper)
    # is the corner-to-corner diagonal of that extent = 21*sqrt(2) px.
    from cellscope.analysis.quantify import measure_frame
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[10:31, 10:31] = 1  # 21x21 block
    inten = np.ones((1, 40, 40), dtype=np.float32)
    m = measure_frame(lab, inten, pixel_size_um=0.5)[1]
    assert m["area_px"] == 21 * 21
    expected = 21 * np.sqrt(2) * 0.5  # edge-to-edge diagonal, in microns
    assert abs(m["feret_diameter_um"] - expected) < 0.05, m["feret_diameter_um"]
    # Feret must be >= the equivalent-disk diameter for this non-circular shape.
    assert m["feret_diameter_um"] >= m["equiv_diameter_um"]


def test_sensitivity_changes_detection():
    loader = MockLoader()
    strict = run_analysis(loader, "A2", AnalysisSettings(sensitivity=0.2))
    loose = run_analysis(loader, "A2", AnalysisSettings(sensitivity=0.85))
    strict_area = sum(m.area_px for m in strict.measurements)
    loose_area = sum(m.area_px for m in loose.measurements)
    # The control must actually do something...
    assert strict_area != loose_area
    # ...and a looser threshold passes strictly more foreground pixels, so the
    # total detected area cannot shrink (cell COUNT can fall as cells merge).
    assert loose_area >= strict_area


def test_sensitivity_sign_stable_on_negative_data():
    # A background-subtracted image: a brighter region on a NEGATIVE background.
    img = np.full((64, 64), -50.0, dtype=np.float32)
    img[24:40, 24:40] = -10.0
    strict = segment_frame(img, sensitivity=0.1, smoothing=0.0, min_size=1)
    loose = segment_frame(img, sensitivity=0.9, smoothing=0.0, min_size=1)
    # Direction must hold regardless of the sign of the data.
    assert (loose > 0).sum() >= (strict > 0).sum()
    # Strict must not flood the whole image (the inversion bug did exactly that).
    assert 0 < (strict > 0).sum() < img.size


def test_compose_rgb_handles_nan():
    frame = np.zeros((1, 16, 16), dtype=np.float32)
    frame[0, 4:12, 4:12] = 200.0
    frame[0, 0, 0] = np.nan
    rgb = compose_rgb(frame, [(255, 255, 255)], [True], 0.5, 0.5)
    assert rgb.dtype == np.uint8
    assert rgb.max() > 0  # one NaN pixel must not black out the channel


def test_mock_small_size_ok():
    loader = MockLoader(size=64, n_wells=2, n_time=3)
    arr = loader.get_well(loader.list_wells()[0].well_id)
    assert arr.shape == (3, 1, 2, 64, 64)


def test_mock_rejects_tiny_size():
    try:
        MockLoader(size=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError for size=8")


if __name__ == "__main__":
    t0 = time.time()
    test_mock_dataset_shape()
    test_wells_differ()
    test_pipeline_produces_real_results()
    test_downsample_preserves_real_units()
    test_pixel_size_override_scales_measurements()
    test_feret_diameter_of_known_shape()
    test_sensitivity_changes_detection()
    test_sensitivity_sign_stable_on_negative_data()
    test_compose_rgb_handles_nan()
    test_mock_small_size_ok()
    test_mock_rejects_tiny_size()
    print(f"All analysis checks passed in {time.time() - t0:.1f}s")
