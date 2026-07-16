"""Tests for the cellscope-downsample tool.

Builds a small synthetic Cephla acquisition, downsamples it, and checks that the
copy is smaller, keeps its structure, and reports the correctly-scaled pixel size.

Run:  python tests/test_downsample.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellscope.data.cephla_loader import CephlaLoader  # noqa: E402
from cellscope.downsample import downsample_array  # noqa: E402


def test_block_mean_shape_and_intensity():
    a = np.full((60, 90), 100, dtype=np.uint16)
    d = downsample_array(a, 3)
    assert d.shape == (20, 30)
    # Block-mean of a constant field preserves the value (mean intensity intact).
    assert d.mean() == 100
    # RGB handled per-channel.
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    dr = downsample_array(rgb, 2)
    assert dr.shape == (20, 20, 3)
    assert dr[..., 1].mean() == 200 and dr[..., 0].mean() == 0


def _build_cephla(root: Path):
    (root / "acquisition parameters.json").write_text(json.dumps({
        "Nt": 2, "Nz": 1, "sensor_pixel_size_um": 1.85,
        "objective": {"magnification": 20.0},
    }))
    (root / "coordinates.csv").write_text("region,x (mm),y (mm)\nB2,1,1\n")
    for t in range(2):
        tp = root / str(t)
        tp.mkdir()
        for region in ("B2", "B3"):
            for fov in ("0", "1"):
                for chan in ("Fluorescence_488_nm_Ex", "Fluorescence_638_nm_Ex"):
                    tifffile.imwrite(str(tp / f"{region}_{fov}_0_{chan}.tiff"),
                                     (np.random.default_rng(0).integers(
                                         0, 4000, (64, 64))).astype(np.uint16))


def test_downsample_cephla_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "full"
        dst = Path(td) / "small"
        src.mkdir()
        _build_cephla(src)

        r = subprocess.run(
            [sys.executable, "-m", "cellscope.downsample", str(src), str(dst),
             "--factor", "2", "-j", "2"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert r.returncode == 0, r.stderr + r.stdout
        assert "2x smaller" in r.stdout or "x smaller" in r.stdout

        # Structure preserved: same relative image paths.
        src_imgs = sorted(p.relative_to(src) for p in src.rglob("*.tiff"))
        dst_imgs = sorted(p.relative_to(dst) for p in dst.rglob("*.tiff"))
        assert src_imgs == dst_imgs and len(dst_imgs) == 16

        # Each image is half-size.
        one = tifffile.imread(str(dst / "0" / "B2_0_0_Fluorescence_488_nm_Ex.tiff"))
        assert one.shape == (32, 32), one.shape

        # Metadata copied; pixel size embedded and scaled x2.
        params = json.loads((dst / "acquisition parameters.json").read_text())
        assert abs(params["pixel_size_um"] - (1.85 / 20) * 2) < 1e-9
        assert params["cellscope_downsample_factor"] == 2

        # The loader reads the reduced copy with the correct effective pixel size.
        loader = CephlaLoader(str(dst))
        assert abs(loader.pixel_size_um - (1.85 / 20) * 2) < 1e-9
        w = loader.list_wells()[0]
        assert (w.height, w.width) == (32, 32)


if __name__ == "__main__":
    test_block_mean_shape_and_intensity()
    print("[ok] block-mean shape + intensity + RGB")
    test_downsample_cephla_end_to_end()
    print("[ok] downsample Cephla end-to-end (structure, size, scaled pixel size)")
    print("All downsample checks passed")
