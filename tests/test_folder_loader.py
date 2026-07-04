"""Tests for the real TIFF/JPG folder reader, using generated image files.

Run:  python tests/test_folder_loader.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tifffile  # noqa: E402
from PIL import Image  # noqa: E402

from cellscope.analysis import AnalysisSettings, run_analysis  # noqa: E402
from cellscope.data.folder_loader import (  # noqa: E402
    FolderLoader,
    NoImagesFoundError,
    _parse_tokens,
)


def _blob(h=64, w=64, cx=32, cy=32, amp=4000):
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 7.0 ** 2)))
    return (amp * g + 50).astype(np.uint16)


def test_no_images_raises():
    with tempfile.TemporaryDirectory() as td:
        try:
            FolderLoader(td)
        except NoImagesFoundError:
            return
    raise AssertionError("expected NoImagesFoundError on empty folder")


def test_multidim_tiff_is_one_well():
    with tempfile.TemporaryDirectory() as td:
        # (T, C, Y, X) OME-TIFF hyperstack.
        data = np.stack([
            np.stack([_blob(cx=20 + 2 * t), _blob(cx=40, amp=2000 + 100 * t)])
            for t in range(5)
        ]).astype(np.uint16)  # shape (5, 2, 64, 64)
        tifffile.imwrite(str(Path(td) / "A1_movie.tif"), data, ome=True,
                         metadata={"axes": "TCYX"})
        loader = FolderLoader(td)
        wells = loader.list_wells()
        assert len(wells) == 1, [w.well_id for w in wells]
        w = wells[0]
        assert (w.n_time, w.n_channels) == (5, 2), (w.n_time, w.n_channels)
        arr = loader.get_well(w.well_id)
        assert arr.shape == (5, 1, 2, 64, 64), arr.shape


def test_named_planes_group_into_wells():
    with tempfile.TemporaryDirectory() as td:
        for well in ("A1", "A2"):
            for t in range(3):
                for c in range(2):
                    img = _blob(cx=24 + 3 * t + 5 * c)
                    tifffile.imwrite(str(Path(td) / f"{well}_t{t}_c{c}.tif"), img)
        loader = FolderLoader(td)
        ids = sorted(w.well_id for w in loader.list_wells())
        assert ids == ["A1", "A2"], ids
        info = loader.get_well_info("A1")
        assert (info.n_time, info.n_channels) == (3, 2), (info.n_time, info.n_channels)
        arr = loader.get_well("A1")
        assert arr.shape == (3, 1, 2, 64, 64), arr.shape
        # The two channels must differ (c0 vs c1 blobs at different positions).
        assert not np.array_equal(arr[0, 0, 0], arr[0, 0, 1])


def test_fluor_named_channels():
    with tempfile.TemporaryDirectory() as td:
        for fl in ("DAPI", "GFP"):
            tifffile.imwrite(str(Path(td) / f"A1_{fl}.tif"), _blob())
        loader = FolderLoader(td)
        assert sorted(loader.channel_names) == ["Dapi", "Gfp"], loader.channel_names
        info = loader.get_well_info(loader.list_wells()[0].well_id)
        assert info.n_channels == 2


def test_plain_rgb_jpgs_are_one_time_series():
    with tempfile.TemporaryDirectory() as td:
        for i in range(4):
            rgb = np.zeros((48, 48, 3), dtype=np.uint8)
            rgb[10:30, 10 + i:30 + i, 1] = 220  # moving green blob
            Image.fromarray(rgb, "RGB").save(str(Path(td) / f"frame{i:03d}.jpg"))
        loader = FolderLoader(td)
        wells = loader.list_wells()
        assert len(wells) == 1, [w.well_id for w in wells]
        w = wells[0]
        assert w.n_time == 4, w.n_time
        assert w.n_channels == 3, w.n_channels
        arr = loader.get_well(w.well_id)
        assert arr.shape == (4, 1, 3, 48, 48), arr.shape
        # Green channel carries signal; red is near-empty (JPEG is lossy, so
        # allow small compression artifacts rather than exactly 0).
        assert arr[0, 0, 1].max() > 100
        assert arr[0, 0, 0].max() < arr[0, 0, 1].max() // 2


def test_single_grayscale_tiff():
    with tempfile.TemporaryDirectory() as td:
        tifffile.imwrite(str(Path(td) / "snapshot.tif"), _blob())
        loader = FolderLoader(td)
        w = loader.list_wells()[0]
        assert (w.n_time, w.n_z, w.n_channels) == (1, 1, 1)
        arr = loader.get_well(w.well_id)
        assert arr.shape == (1, 1, 1, 64, 64), arr.shape


def test_plain_multipage_tiff_keeps_all_frames():
    # A metadata-less multi-page TIFF (default tifffile.imwrite of a 3-D array,
    # axes 'QYX') must be read as a stack, not collapsed into one plane.
    with tempfile.TemporaryDirectory() as td:
        data = np.stack([_blob(cx=16 + 3 * t) for t in range(10)]).astype(np.uint16)
        tifffile.imwrite(str(Path(td) / "timelapse.tif"), data)  # no ome/imagej
        loader = FolderLoader(td)
        w = loader.list_wells()[0]
        assert w.n_time == 10, w.n_time
        arr = loader.get_well(w.well_id)
        assert arr.shape == (10, 1, 1, 64, 64), arr.shape
        # frames must be distinct (blob moved), i.e. not scrambled/duplicated
        assert not np.array_equal(arr[0, 0, 0], arr[9, 0, 0])


def test_row_c_channel_disambiguation():
    # Row-C wells must not take their channel index from the row digit.
    assert _parse_tokens("C3_c2")["well"] == "C3"
    assert _parse_tokens("C3_c2")["c"] == 2
    assert _parse_tokens("well_C3_channel2")["well"] == "C3"
    assert _parse_tokens("well_C3_channel2")["c"] == 2
    # A bare, prefixed 'C#' with no other structure is a channel, not a well.
    assert _parse_tokens("img_c1")["well"] is None
    assert _parse_tokens("img_c1")["c"] == 1
    # Non-C rows are unaffected.
    assert _parse_tokens("B4_c2")["well"] == "B4"
    assert _parse_tokens("B4_c2")["c"] == 2


def test_row_c_two_channel_well():
    with tempfile.TemporaryDirectory() as td:
        for c in (1, 2):
            tifffile.imwrite(str(Path(td) / f"C3_c{c}.tif"), _blob(cx=20 + 8 * c))
        loader = FolderLoader(td)
        assert [w.well_id for w in loader.list_wells()] == ["C3"]
        assert loader.get_well_info("C3").n_channels == 2


def test_nested_folders_not_hidden_by_stray_top_image():
    with tempfile.TemporaryDirectory() as td:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(str(Path(td) / "overview.jpg"))
        sub = Path(td) / "plate"
        sub.mkdir()
        for t in range(5):
            tifffile.imwrite(str(sub / f"A1_t{t}.tif"), _blob(cx=18 + 3 * t))
        loader = FolderLoader(td)
        ids = [w.well_id for w in loader.list_wells()]
        assert "A1" in ids, ids
        assert loader.get_well_info("A1").n_time == 5


def test_pipeline_runs_on_real_files():
    with tempfile.TemporaryDirectory() as td:
        # A small time series with a moving blob, named with time tokens.
        for t in range(6):
            tifffile.imwrite(str(Path(td) / f"A1_t{t}.tif"), _blob(cx=20 + 3 * t))
        loader = FolderLoader(td)
        wa = run_analysis(loader, loader.list_wells()[0].well_id,
                          AnalysisSettings(min_size=10))
        assert wa.n_tracks >= 1, wa.n_tracks
        assert len(wa.measurements) >= 1
        assert wa.counts_per_frame.shape == (6,)


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_no_images_raises()
    test_multidim_tiff_is_one_well()
    test_named_planes_group_into_wells()
    test_fluor_named_channels()
    test_plain_rgb_jpgs_are_one_time_series()
    test_single_grayscale_tiff()
    test_plain_multipage_tiff_keeps_all_frames()
    test_row_c_channel_disambiguation()
    test_row_c_two_channel_well()
    test_nested_folders_not_hidden_by_stray_top_image()
    test_pipeline_runs_on_real_files()
    print(f"All folder-loader checks passed in {time.time() - t0:.1f}s")
