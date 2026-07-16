"""Tests for the pluggable segmentation engines (threshold + optional Cellpose).

The actual Cellpose GPU run is validated on a GPU machine; here we verify the
plumbing: engine selection, threshold-engine equivalence, and that requesting
Cellpose when it isn't installed fails clearly.

Run:  python tests/test_engines.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis import AnalysisSettings, available_engines, cellpose_available
from cellscope.analysis.engines import ThresholdEngine, get_engine
from cellscope.analysis.segmentation import relabel_min_size, segment_frame


def test_threshold_always_available():
    assert "threshold" in available_engines()
    assert get_engine("threshold").name == "threshold"


def test_relabel_min_size():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[2:5, 2:5] = 7      # 9 px
    lab[10:16, 10:16] = 3  # 36 px
    out = relabel_min_size(lab, min_size=20)
    # small object dropped; the surviving one renumbered to 1
    assert set(np.unique(out)) == {0, 1}
    assert int((out == 1).sum()) == 36


def test_threshold_engine_matches_segment_frame():
    rng = np.random.default_rng(0)
    frames = [((rng.random((48, 48)) < 0.02) * 5000).astype(np.uint16) for _ in range(3)]
    s = AnalysisSettings(min_size=1, smoothing=1.0)
    engine_out = ThresholdEngine().segment_stack(frames, s)
    for f, lab in zip(frames, engine_out):
        direct = segment_frame(f, sensitivity=s.sensitivity,
                               smoothing=s.smoothing, min_size=s.min_size)
        assert np.array_equal(lab, direct)


def test_settings_copy_carries_engine():
    s = AnalysisSettings(engine="cellpose", cellpose_diameter=42.0, cellpose_gpu=False)
    c = s.copy()
    assert c.engine == "cellpose"
    assert c.cellpose_diameter == 42.0 and c.cellpose_gpu is False


def test_cellpose_selection_when_absent_errors_clearly():
    if cellpose_available():
        print("[skip] cellpose is installed; graceful-absence path not exercised")
        return
    assert "cellpose" not in available_engines()
    try:
        get_engine("cellpose")
    except RuntimeError as exc:
        assert "not installed" in str(exc).lower()
        return
    raise AssertionError("expected RuntimeError when cellpose is not installed")


if __name__ == "__main__":
    test_threshold_always_available()
    test_relabel_min_size()
    test_threshold_engine_matches_segment_frame()
    test_settings_copy_carries_engine()
    test_cellpose_selection_when_absent_errors_clearly()
    print("All engine checks passed")
