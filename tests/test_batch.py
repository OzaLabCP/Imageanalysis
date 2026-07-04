"""Tests for the headless parallel batch runner.

Runs the CLI in subprocesses (real spawn / no Qt) against the built-in mock
plate, so no image files are needed.

Run:  python tests/test_batch.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", "cellscope.batch", *args],
        capture_output=True, text=True, cwd=str(ROOT), **kw,
    )


def test_batch_is_headless():
    # Importing the batch module must NOT pull in Qt (it has to run on a
    # display-less compute node).
    code = ("import sys, cellscope.batch; "
            "assert 'PySide6' not in sys.modules, sorted(m for m in sys.modules "
            "if 'PySide6' in m); print('headless-ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "headless-ok" in r.stdout


def test_list():
    r = _run(["__mock__", "--list"])
    assert r.returncode == 0, r.stderr
    assert "A1" in r.stdout and "B3" in r.stdout
    assert "6 positions" in r.stdout


def test_parallel_run_and_combine():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        r = _run(["__mock__", "-o", str(out), "-j", "2",
                  "--positions", "A1,A2,A3", "--downsample", "2", "--combine"])
        assert r.returncode == 0, r.stderr + r.stdout
        # Per-position CSVs.
        for wid in ("A1", "A2", "A3"):
            assert (out / f"{wid}.csv").exists(), r.stdout
        # Combined CSV with the region column and real headers.
        combined = out / "all_measurements.csv"
        assert combined.exists()
        lines = combined.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        for col in ("well", "region", "condition", "feret_diameter_um",
                    "mean_Nuclei", "mean_Reporter"):
            assert col in header, header
        assert len(lines) > 1
        # region is derived from the well id (A1 has no fov -> region A1).
        region_col = header.index("region")
        assert lines[1].split(",")[region_col] in ("A1", "A2", "A3")
        assert "3 analyzed" in r.stdout


def test_resume_skips_existing():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        a = _run(["__mock__", "-o", str(out), "-j", "1",
                  "--positions", "A1", "--downsample", "2"])
        assert a.returncode == 0, a.stderr
        b = _run(["__mock__", "-o", str(out), "-j", "1",
                  "--positions", "A1", "--downsample", "2", "--resume"])
        assert b.returncode == 0, b.stderr
        assert "1 skipped" in b.stdout, b.stdout


if __name__ == "__main__":
    test_batch_is_headless()
    print("[ok] headless (no Qt)")
    test_list()
    print("[ok] --list")
    test_parallel_run_and_combine()
    print("[ok] parallel run + combined CSV with region column")
    test_resume_skips_existing()
    print("[ok] --resume skips existing")
    print("All batch checks passed")
