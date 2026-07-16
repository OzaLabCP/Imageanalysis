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
    # Every length metric scales linearly with the pixel size too.
    for attr in ("length_major_um", "length_minor_um", "perimeter_um"):
        b = sum(getattr(m, attr) for m in base.measurements)
        d = sum(getattr(m, attr) for m in doubled.measurements)
        assert b > 0 and abs(d / b - 2.0) < 0.01, (attr, b, d)


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


def test_extended_metrics_present_and_sane():
    loader = MockLoader(size=128, n_wells=1, n_time=3)
    wa = run_analysis(loader, loader.list_wells()[0].well_id, AnalysisSettings())
    n_chan = len(loader.channel_names)
    assert wa.measurements
    for m in wa.measurements:
        # Morphometrics: major >= minor > 0, eccentricity in [0, 1), perimeter > 0.
        assert m.length_major_um >= m.length_minor_um > 0, m
        assert 0.0 <= m.eccentricity < 1.0 + 1e-9, m.eccentricity
        assert m.perimeter_um > 0, m.perimeter_um
        # Per-channel intensity stats, one value per channel, correctly ordered.
        for stat in (m.mean_intensity, m.max_intensity, m.min_intensity, m.std_intensity):
            assert len(stat) == n_chan, (len(stat), n_chan)
        for c in range(n_chan):
            assert m.min_intensity[c] <= m.mean_intensity[c] <= m.max_intensity[c], (c, m)
            assert m.std_intensity[c] >= 0.0


def test_parquet_exact_schema():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        print("  (skipped parquet schema test: pyarrow not installed)")
        return
    import tempfile
    from cellscope.export import parquet_columns, write_measurements_parquet

    loader = MockLoader(size=96, n_wells=1, n_time=3, n_channels=2)
    loader._channel_names = ["GFP", "Alexa Fluor 647"]
    wid = loader.list_wells()[0].well_id
    wa = run_analysis(loader, wid, AnalysisSettings())

    # The first 18 columns are the fixed reference layout (a notebook built on
    # that schema still reads by name); `fov` and `condition` are appended so the
    # per-cell key (Dataset, Well, fov, Timepoint, Label) is unique across the
    # FOVs pooled into a well (Label restarts at 1 per FOV).
    reference18 = [
        "Label", "Diameter (Equivalent) (um)", "Diameter (Feret) (um)",
        "Length Major (um)", "Length Minor (um)", "Perimeter (um)",
        "Intensity Mean (GFP)", "Intensity Mean (Alexa Fluor 647)",
        "Intensity Max (GFP)", "Intensity Max (Alexa Fluor 647)",
        "Intensity STD (GFP)", "Intensity STD (Alexa Fluor 647)",
        "Intensity Min (GFP)", "Intensity Min (Alexa Fluor 647)",
        "Eccentricity", "Dataset", "Timepoint", "Well",
    ]
    expected = reference18 + ["fov", "condition", "segment"]
    cols = parquet_columns(loader.channel_names)
    assert cols == expected, cols
    # The reference schema must remain the exact 18-column prefix.
    assert cols[:18] == reference18

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "p.parquet")
        n = write_measurements_parquet(path, [(wid, "ctrl", wa)], dataset="src.zarr")
        assert n == len(wa.measurements)
        t = pq.read_table(path)
        assert t.schema.names == expected, t.schema.names
        types = {nm: str(ty) for nm, ty in zip(t.schema.names, t.schema.types)}
        assert types["Label"] == "int64" and types["Timepoint"] == "int64"
        assert types["Dataset"] == "string" and types["Well"] == "string"
        assert types["fov"] == "string" and types["condition"] == "string"
        assert types["segment"] == "int64"
        assert types["Eccentricity"] == "double"
        # Timepoint is 0-based (unlike the tidy CSV's 1-based `time`).
        assert min(t.column("Timepoint").to_pylist()) == 0
        # Well is the region (FOV suffix stripped); demo wells have no FOV.
        assert set(t.column("Well").to_pylist()) == {wid}
        # condition is threaded through from the (well, condition, wa) item.
        assert set(t.column("condition").to_pylist()) == {"ctrl"}
        # A gap-free time course is one segment (0).
        assert set(t.column("segment").to_pylist()) == {0}


def test_provenance_and_device():
    from cellscope.analysis import resolve_device
    from cellscope.provenance import build_run_metadata

    # resolve_device must never raise, even with no torch / no GPU.
    off = resolve_device(False)
    assert off["device"] == "cpu" and not off["fell_back"]
    on = resolve_device(True)
    assert on["device"] in ("cpu", "cuda", "mps") and "detail" in on

    loader = MockLoader(size=64, n_wells=1, n_time=2)
    wa = run_analysis(loader, loader.list_wells()[0].well_id, AnalysisSettings())
    meta = build_run_metadata(
        source="/data/exp", engine="cellpose", settings=wa.settings,
        pixel_size_um=0.65, downsample=1, channel_names=loader.channel_names,
        gpu=on, positions=["A1"], output_format="parquet",
        created_utc="2026-07-16T00:00:00Z",
    )
    for key in ("cellscope_version", "engine", "gpu", "settings",
                "pixel_size_um", "channel_names", "n_positions", "created_utc"):
        assert key in meta, key
    assert meta["engine"] == "cellpose" and meta["n_positions"] == 1


def test_run_analysis_accepts_prefetched_array():
    # Passing a prefetched array (batch overlaps disk I/O with compute) must give
    # identical results to letting run_analysis load it.
    loader = MockLoader(size=96, n_wells=1, n_time=2)
    wid = loader.list_wells()[0].well_id
    a = run_analysis(loader, wid, AnalysisSettings())
    b = run_analysis(loader, wid, AnalysisSettings(),
                     array=loader.get_well(wid, downsample=1))
    assert a.n_tracks == b.n_tracks
    assert len(a.measurements) == len(b.measurements)
    assert abs(sum(m.area_um2 for m in a.measurements)
               - sum(m.area_um2 for m in b.measurements)) < 1e-6


def test_cephla_pixel_size_tube_lens_correction():
    from cellscope.data.cephla_loader import _pixel_size_from_params
    # Real Squid params: 20x nominal, but a 50 mm tube lens vs its 180 mm design
    # => effective 5.56x => 1.85 / 5.56 = 0.333 um/px (matches acquisition.yaml).
    params = {"sensor_pixel_size_um": 1.85, "tube_lens_mm": 50,
              "objective": {"magnification": 20.0, "tube_lens_f_mm": 180.0}}
    assert abs(_pixel_size_from_params(params) - 0.333) < 0.002
    # No tube-lens info -> naive sensor / magnification.
    assert abs(_pixel_size_from_params(
        {"sensor_pixel_size_um": 1.85, "objective": {"magnification": 20.0}}) - 0.0925) < 1e-3
    # Explicit pixel size always wins; missing data -> 1.0.
    assert _pixel_size_from_params(
        {"pixel_size_um": 0.5, "sensor_pixel_size_um": 1.85}) == 0.5
    assert _pixel_size_from_params({}) == 1.0


def test_cephla_flat_single_timepoint_folder():
    import json
    import tempfile

    import tifffile
    from cellscope.data.cephla_loader import CephlaLoader

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # A flat, single-timepoint Cephla export: images + metadata together,
        # no numbered timepoint subfolders (the layout a single download yields).
        params = {"sensor_pixel_size_um": 1.85, "tube_lens_mm": 50,
                  "objective": {"magnification": 20.0, "tube_lens_f_mm": 180.0}}
        (p / "acquisition parameters.json").write_text(json.dumps(params))
        img = np.full((80, 80), 120, dtype=np.uint16)
        img[30:50, 30:50] = 4000
        for region in ("H2", "H3"):
            for fov in (0, 1):
                for chan in ("Fluorescence_488_nm_Ex", "Fluorescence_638_nm_Ex"):
                    tifffile.imwrite(str(p / f"{region}_{fov}_0_{chan}.tiff"), img)

        assert CephlaLoader.looks_like(p) is True
        ld = CephlaLoader(str(p))
        assert ld.channel_names == ["488 nm", "638 nm"], ld.channel_names
        assert abs(ld.pixel_size_um - 0.333) < 0.002, ld.pixel_size_um  # tube-lens corrected
        assert sorted(w.well_id for w in ld.list_wells()) == ["H2-0", "H2-1", "H3-0", "H3-1"]
        assert ld.get_well("H2-0").shape == (1, 1, 2, 80, 80)  # (T, Z, C, Y, X)

    # A folder that is NOT Cephla-named must not be hijacked.
    with tempfile.TemporaryDirectory() as d2:
        p2 = Path(d2)
        for name in ("photo.tif", "scan_a.tif", "img001.tif"):
            tifffile.imwrite(str(p2 / name), np.zeros((8, 8), dtype=np.uint16))
        assert CephlaLoader.looks_like(p2) is False


def test_analyze_report():
    try:
        import pandas as pd
        import matplotlib  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        print("  (skipped analyze report test: pandas/matplotlib not installed)")
        return
    import os
    import tempfile

    from cellscope.analyze import run

    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        rows = []
        # H2 grows a high-green subpopulation over time; H3 stays low.
        for well, fmax in (("H2", 0.5), ("H3", 0.05)):
            for t in range(3):
                n = 500
                resp = rng.random(n) < fmax * (t / 2.0)
                green = np.where(resp, rng.lognormal(8.6, 0.3, n), rng.lognormal(7.2, 0.4, n))
                for gv in green:
                    rows.append((well, t, float(gv), 400.0, 9.0, 0.5))
        pd.DataFrame(rows, columns=[
            "Well", "Timepoint", "Intensity Mean (488 nm)",
            "Intensity Mean (638 nm)", "Diameter (Equivalent) (um)", "Eccentricity"
        ]).to_parquet(os.path.join(d, "m.parquet"))

        info = run(os.path.join(d, "m.parquet"), os.path.join(d, "report"))
        assert info["timepoints"] == [0, 1, 2]
        assert info["responders"] > 0
        for f in ("index.html", "responder_fraction.png",
                  "group_timepoint_summary.csv", "responder_characteristics.csv"):
            assert os.path.exists(os.path.join(d, "report", f)), f
        # The subpopulation must be detected as larger in H2 than H3 at the last timepoint.
        s = pd.read_csv(os.path.join(d, "report", "group_timepoint_summary.csv"))
        last = s[s["Timepoint"] == 2].set_index("group")["pct_responders"]
        assert last["H2"] > last["H3"] + 10, last.to_dict()


def test_segment_map_marks_gaps():
    from types import SimpleNamespace

    from cellscope.export import segment_map

    def wa(frames):
        return SimpleNamespace(measurements=[SimpleNamespace(frame=f) for f in frames])

    # A gap-free run is one segment.
    assert segment_map(wa([0, 0, 1, 2, 2])) == {0: 0, 1: 0, 2: 0}
    # A missing timepoint (3, then 5) starts a new segment - so grouping by
    # (position, segment, Label) can't join a track across the gap.
    m = segment_map(wa([0, 1, 3, 5, 6]))
    assert m == {0: 0, 1: 0, 3: 1, 5: 2, 6: 2}, m
    # Two cells the tracker split across a gap land in different segments.
    assert m[3] != m[5]


def test_qc_report():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError:
        print("  (skipped qc test: pandas/pyarrow not installed)")
        return
    import os
    import tempfile

    from cellscope.qc import format_issues, qc_report

    with tempfile.TemporaryDirectory() as d:
        # A clean table: unique keys, no NaN/saturation, contiguous timepoints.
        clean = pd.DataFrame({
            "Dataset": ["ds"] * 6,
            "Well": ["H2"] * 6,
            "fov": ["0"] * 3 + ["1"] * 3,
            "Timepoint": [0, 1, 2] * 2,
            "Label": [1, 1, 1, 1, 1, 1],
            "Intensity Mean (488 nm)": [100.0, 110.0, 120.0, 90.0, 95.0, 100.0],
            "Intensity Max (488 nm)": [500.0, 510.0, 520.0, 490.0, 495.0, 500.0],
        })
        p_clean = os.path.join(d, "clean.parquet")
        clean.to_parquet(p_clean)
        rep = qc_report(p_clean, os.path.join(d, "qc_clean.json"))
        assert rep["ok"], rep["issues"]
        assert rep["duplicate_key_rows"] == 0
        assert os.path.exists(os.path.join(d, "qc_clean.json"))
        assert format_issues(rep) == "QC: no issues found."

        # A corrupt table that mimics every silent failure mode at once:
        #  * no 'fov' column -> (Well, Timepoint, Label) collides across FOVs,
        #  * a NaN (missing/blank channel) intensity row,
        #  * a saturated cell, and
        #  * a gap in the timepoint sequence (0, 2 - missing 1).
        bad = pd.DataFrame({
            "Dataset": ["ds"] * 4,
            "Well": ["H2", "H2", "H2", "H2"],
            "Timepoint": [0, 0, 2, 2],
            "Label": [1, 1, 2, 2],  # (Well, Timepoint, Label) duplicated -> collision
            "Intensity Mean (488 nm)": [100.0, float("nan"), 120.0, 130.0],
            "Intensity Max (488 nm)": [500.0, 510.0, 70000.0, 520.0],  # >saturation
        })
        p_bad = os.path.join(d, "bad.parquet")
        bad.to_parquet(p_bad)
        rep2 = qc_report(p_bad)
        assert not rep2["ok"]
        blob = " ".join(rep2["issues"]).lower()
        assert "fov" in blob            # missing-fov key-collision warning
        assert rep2["duplicate_key_rows"] > 0
        assert "blank channel" in blob or "missing" in blob  # NaN intensity
        assert "saturation" in blob     # saturated cell
        assert "gap" in blob            # 0,2 timepoint gap
        assert "QC:" in format_issues(rep2)

        # Coverage matrix: 3 positions all reach TP0; only 1 reaches TP2 -> the
        # under-covered timepoint must be reported as data and flagged.
        cov = pd.DataFrame({
            "Well": ["H2", "H2", "H3", "H2"],
            "fov": ["0", "1", "0", "0"],
            "Timepoint": [0, 0, 0, 2],
            "Label": [1, 1, 1, 1],
            "Intensity Mean (488 nm)": [100.0, 100.0, 100.0, 100.0],
        })
        p_cov = os.path.join(d, "cov.parquet")
        cov.to_parquet(p_cov)
        repc = qc_report(p_cov)
        assert repc["positions"] == 3
        assert repc["positions_per_timepoint"] == {0: 3, 2: 1}, repc["positions_per_timepoint"]
        assert any("<50%" in m for m in repc["issues"]), repc["issues"]


def test_fog_plot_script():
    try:
        import pandas as pd
        import matplotlib  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        print("  (skipped fog_plot test: pandas/matplotlib/pyarrow not installed)")
        return
    import importlib.util
    import os
    import tempfile

    fp_path = Path(__file__).resolve().parents[1] / "scripts" / "fog_plot.py"
    spec = importlib.util.spec_from_file_location("fog_plot", fp_path)
    fog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fog)

    # Case-insensitive column resolution (so --facet-by Condition finds condition).
    df0 = pd.DataFrame({"Well": ["H2"], "condition": ["A"], "Timepoint": [0]})
    assert fog._resolve_column(df0, "Condition") == "condition"
    assert fog._resolve_column(df0, "nope") is None

    # Non-positive counting (invisible on a log axis): zeros + negatives + NaN.
    assert fog._count_nonpositive([1.0, 0.0, -3.0, float("nan"), 5.0]) == 3

    with tempfile.TemporaryDirectory() as d:
        # Uneven coverage: 3 positions (H2-0/H2-1/H3-0) all reach TP0, but only one
        # reaches TP1 - the exact "partial late timepoint" trap.
        rows = []
        for (well, fov), tmax in {("H2", "0"): 1, ("H2", "1"): 0, ("H3", "0"): 0}.items():
            for t in range(tmax + 1):
                for _ in range(20):
                    rows.append((well, fov, "Drug with a very long condition name",
                                 t, 100.0 + t, 0.0 if _ == 0 else 500.0))
        df = pd.DataFrame(rows, columns=[
            "Well", "fov", "condition", "Timepoint",
            "Intensity Mean (488 nm)", "Intensity Mean (638 nm)"])
        p = os.path.join(d, "m.parquet")
        df.to_parquet(p)

        per, total = fog._coverage(df)
        assert total == 3 and per == {0: 3, 1: 1}, (per, total)

        # A channel with exact-zero cells (638) is flagged, not silently dropped.
        assert fog._count_nonpositive(df["Intensity Mean (638 nm)"].to_numpy(float)) > 0

        # Faceting by the (long-named, case-different) condition renders and exits 0.
        out = os.path.join(d, "fog.png")
        rc = fog.main([p, "-o", out, "--facet-by", "Condition",
                       "--channel", "Intensity Mean (638 nm)", "--control",
                       "Drug with a very long condition name"])
        assert rc == 0 and os.path.exists(out)

        # An unknown --facet-by errors loudly (exit 2) instead of silently pooling.
        assert fog.main([p, "-o", out, "--facet-by", "Treatment"]) == 2
        # An unknown --channel errors loudly too.
        assert fog.main([p, "-o", out, "--channel", "Intensity Mean (999 nm)"]) == 2


def test_cephla_multi_timepoint_folders():
    import json
    import tempfile

    import tifffile
    from cellscope.data.cephla_loader import CephlaLoader

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        params = {"sensor_pixel_size_um": 1.85, "tube_lens_mm": 50,
                  "objective": {"magnification": 20.0, "tube_lens_f_mm": 180.0}}
        img = np.full((72, 72), 120, dtype=np.uint16)
        img[24:44, 24:44] = 4000
        # Numbered timepoint folders, each with images + metadata inside (not at root).
        for tp in ("0", "1", "7"):
            sub = root / tp
            sub.mkdir()
            (sub / "acquisition parameters.json").write_text(json.dumps(params))
            for region in ("H2", "H3"):
                for chan in ("Fluorescence_488_nm_Ex", "Fluorescence_638_nm_Ex"):
                    tifffile.imwrite(str(sub / f"{region}_0_0_{chan}.tiff"), img)

        assert CephlaLoader.looks_like(root) is True
        ld = CephlaLoader(str(root))
        assert ld.channel_names == ["488 nm", "638 nm"]
        assert abs(ld.pixel_size_um - 0.333) < 0.002  # metadata found inside a timepoint folder
        # Three timepoint folders -> a 3-frame time axis.
        assert ld.get_well("H2-0").shape[0] == 3


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
    test_extended_metrics_present_and_sane()
    test_parquet_exact_schema()
    test_provenance_and_device()
    test_run_analysis_accepts_prefetched_array()
    test_cephla_pixel_size_tube_lens_correction()
    test_cephla_flat_single_timepoint_folder()
    test_cephla_multi_timepoint_folders()
    test_segment_map_marks_gaps()
    test_qc_report()
    test_fog_plot_script()
    test_analyze_report()
    test_mock_small_size_ok()
    test_mock_rejects_tiny_size()
    print(f"All analysis checks passed in {time.time() - t0:.1f}s")
