"""Tests for the Cephla/Squid acquisition reader.

Builds a small synthetic acquisition (numbered timepoint folders,
<region>_<fov>_<z>_<channel>.tiff) and checks the loader assembles it into
correct (T, Z, C, Y, X) positions. Also does a metadata-only check against a
real acquisition folder if one is present (no heavy pixel load).

Run:  python tests/test_cephla_loader.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis import AnalysisSettings, run_analysis  # noqa: E402
from cellscope.data.cephla_loader import CephlaLoader  # noqa: E402

REGIONS = ["B2", "B3"]
FOVS = ["0", "1"]
CHANNELS = ["Fluorescence_488_nm_Ex", "Fluorescence_638_nm_Ex"]
N_T = 3


def _blob(cx, amp):
    yy, xx = np.mgrid[0:32, 0:32]
    g = np.exp(-(((xx - cx) ** 2 + (yy - 16) ** 2) / (2 * 4.0 ** 2)))
    return (amp * g + 20).astype(np.uint16)


def _build_acquisition(root: Path):
    (root / "acquisition parameters.json").write_text(json.dumps({
        "Nt": N_T, "Nz": 1, "sensor_pixel_size_um": 1.85,
        "objective": {"magnification": 20.0, "name": "20x"},
    }))
    (root / "coordinates.csv").write_text("region,x (mm),y (mm),z (mm)\n")
    for t in range(N_T):
        tp = root / str(t)
        tp.mkdir()
        (tp / ".done").write_text("")
        for region in REGIONS:
            for fov in FOVS:
                for ci, chan in enumerate(CHANNELS):
                    # blob moves with time; amplitude differs per channel
                    img = _blob(cx=10 + 3 * t + int(fov), amp=3000 + 800 * ci)
                    tifffile.imwrite(str(tp / f"{region}_{fov}_0_{chan}.tiff"), img)


def test_detection_and_structure():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)
        assert CephlaLoader.looks_like(root)

        loader = CephlaLoader(str(root))
        ids = [w.well_id for w in loader.list_wells()]
        assert ids == ["B2-0", "B2-1", "B3-0", "B3-1"], ids
        assert loader.channel_names == ["488 nm", "638 nm"], loader.channel_names
        # 488 -> greenish, 638 -> reddish
        assert loader.channel_colors[0][1] > loader.channel_colors[0][0]
        assert loader.channel_colors[1][0] > loader.channel_colors[1][1]
        # pixel size = sensor 1.85 / mag 20
        assert abs(loader.pixel_size_um - 1.85 / 20) < 1e-9, loader.pixel_size_um

        w = loader.get_well_info("B2-0")
        assert (w.n_time, w.n_z, w.n_channels) == (N_T, 1, 2)


def test_assembly_axes_correct():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)
        loader = CephlaLoader(str(root))
        arr = loader.get_well("B2-1")
        assert arr.shape == (N_T, 1, 2, 32, 32), arr.shape
        # channel 1 (638) is brighter than channel 0 (488) by construction
        assert arr[0, 0, 1].max() > arr[0, 0, 0].max()
        # blob moves over time -> frames differ
        assert not np.array_equal(arr[0, 0, 0], arr[2, 0, 0])
        # FOV 1 blob sits one pixel right of FOV 0
        other = loader.get_well("B2-0")
        assert not np.array_equal(arr[0, 0, 0], other[0, 0, 0])


def test_pipeline_runs_on_cephla_position():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)
        loader = CephlaLoader(str(root))
        wa = run_analysis(loader, "B3-0", AnalysisSettings(min_size=8))
        assert wa.n_tracks >= 1
        assert wa.counts_per_frame.shape == (N_T,)
        # pixel size flows into real-unit measurements
        assert wa.pixel_size_um == loader.pixel_size_um


def test_downsampled_position():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)
        loader = CephlaLoader(str(root))
        arr = loader.get_well("B2-0", downsample=2)
        assert arr.shape == (N_T, 1, 2, 16, 16), arr.shape  # 32 -> 16


def test_thumbnail_uses_nav_cache():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)
        cache = root / "tile_cache"
        cache.mkdir()
        for region in REGIONS:
            for fov in FOVS:
                for wl in ("488", "638"):
                    np.save(str(cache / f"nav_{region}_fov_{fov}_ch_{wl}_75.npy"),
                            (np.ones((75, 75)) * int(wl)).astype(np.uint16))
        loader = CephlaLoader(str(root))
        thumb = loader.get_thumbnail("B2-0")
        assert thumb.shape == (2, 75, 75), thumb.shape
        assert thumb[0].mean() == 488 and thumb[1].mean() == 638


def test_thumbnail_fallback_without_cache():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_acquisition(root)  # no tile_cache
        loader = CephlaLoader(str(root))
        thumb = loader.get_thumbnail("B2-0", max_size=16)
        assert thumb.shape[0] == 2  # 2 channels, read from the first timepoint
        assert thumb.shape[1] <= 32 and thumb.shape[2] <= 32


def _real_folder_metadata_check():
    """If the real acquisition folder is present, verify metadata only."""
    real = Path(r"C:\Users\joza\OneDrive - Cal Poly\Oza Lab - Shared 2"
                r"\synthetic-cell-imaging\tiff_files_for_imaging"
                r"\2026.06.26-gm-ppk2_2026-06-26_12-56-06.391058")
    if not real.exists():
        print("[skip] real Cephla folder not present")
        return
    assert CephlaLoader.looks_like(real)
    loader = CephlaLoader(str(real))
    wells = loader.list_wells()
    print(f"[real] {len(wells)} positions, channels={loader.channel_names}, "
          f"pixel={loader.pixel_size_um:.4f} um/px, "
          f"T={wells[0].n_time} {wells[0].height}x{wells[0].width}")
    assert len(wells) == 96
    assert wells[0].n_time == 24
    assert abs(loader.pixel_size_um - 1.85 / 20) < 1e-6


if __name__ == "__main__":
    test_detection_and_structure()
    test_assembly_axes_correct()
    test_pipeline_runs_on_cephla_position()
    test_downsampled_position()
    test_thumbnail_uses_nav_cache()
    test_thumbnail_fallback_without_cache()
    _real_folder_metadata_check()
    print("All Cephla-loader checks passed")
