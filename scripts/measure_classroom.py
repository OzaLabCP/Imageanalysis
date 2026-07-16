"""Measure object sizes in the accessible 'classroom' Cephla acquisition."""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis.segmentation import segment_frame
from cellscope.render import compose_rgb, outline_overlay, build_color_lut
from cellscope.colors import track_color

FOLDER = Path(r"C:\Users\joza\OneDrive - Cal Poly\Oza Lab - Shared 2"
              r"\synthetic-cell-imaging\tiff_files_for_imaging"
              r"\classroom.04.24.26_2026-04-24_10-43-41.036911")
USER_FILE = FOLDER / "0" / "C2_7_0_Fluorescence_488_nm_Ex.tiff"
OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def pixel_size():
    try:
        p = json.loads((FOLDER / "acquisition parameters.json").read_text(encoding="utf-8"))
    except Exception:
        return 0.0925, "assumed 20x"
    if p.get("pixel_size_um"):
        return float(p["pixel_size_um"]), "explicit"
    s = p.get("sensor_pixel_size_um")
    obj = p.get("objective") or {}
    mag = (obj.get("magnification") if isinstance(obj, dict) else None) or p.get("magnification")
    if s and mag:
        return float(s) / float(mag), f"{s}um / {mag}x"
    return 0.0925, "fallback"


def measure(path, px):
    img = tifffile.imread(str(path))
    labels = segment_frame(img, sensitivity=0.5, smoothing=1.5, min_size=12)
    counts = np.bincount(labels.ravel())
    sizes = counts[1:]
    sizes = sizes[sizes > 0]
    diam_px = 2.0 * np.sqrt(sizes / np.pi) if sizes.size else np.array([])
    return img, labels, diam_px


def main():
    px, how = pixel_size()
    print(f"pixel size: {px:.4f} um/px ({how})")

    files = [USER_FILE]
    # add a spread of other fields from folder 0
    others = sorted((FOLDER / "0").glob("*_Fluorescence_488_nm_Ex.tiff"))
    for f in others:
        if f != USER_FILE and len(files) < 6:
            files.append(f)

    all_d = []
    overlay_saved = False
    for f in files:
        try:
            img, labels, diam = measure(f, px)
        except Exception as exc:
            print(f"  skip {f.name}: {exc}")
            continue
        if diam.size:
            all_d.append(diam)
            print(f"  {f.name}: {diam.size} objects, median {np.median(diam):.1f} px "
                  f"({np.median(diam)*px:.2f} um), 90th {np.percentile(diam,90):.1f} px")
        else:
            print(f"  {f.name}: 0 objects (range {int(img.min())}-{int(img.max())})")
        if not overlay_saved and diam.size:
            _save_overlay(img, labels, f.stem)
            overlay_saved = True

    if not all_d:
        print("No objects detected in any file.")
        return
    d = np.concatenate(all_d)
    p25, p50, p75, p90 = np.percentile(d, [25, 50, 75, 90])
    print(f"\n=== {d.size} objects across {len(all_d)} fields ===")
    print(f"  median   {p50:5.1f} px  ({p50*px:.2f} um)")
    print(f"  25-75th  {p25:.1f}-{p75:.1f} px  ({p25*px:.2f}-{p75*px:.2f} um)")
    print(f"  90th     {p90:5.1f} px  ({p90*px:.2f} um)")
    factor = max(1, int(round(p50 / 45.0)))
    print(f"\nSuggested downsample: 1/{factor} -> median ~{p50/factor:.0f} px, "
          f"90th ~{p90/factor:.0f} px")
    if p50 < 60:
        print("NOTE: objects already small; do NOT downsample much (you'd lose them).")


def _save_overlay(img, labels, stem):
    # crop a central 800x800 region so cells are visible at reasonable scale
    h, w = img.shape
    y0, x0 = h//2 - 400, w//2 - 400
    crop = img[y0:y0+800, x0:x0+800]
    lab = labels[y0:y0+800, x0:x0+800]
    rgb = compose_rgb(crop[None], [(90, 230, 110)], [True], 0.5, 0.65)
    ids = [int(i) for i in np.unique(lab) if i > 0]
    if ids:
        lut = build_color_lut(ids, track_color)
        ov = outline_overlay(lab.astype(np.int32), lut)
        ov = np.array(Image.fromqimage(ov)) if hasattr(Image, "fromqimage") else None
    # simpler: draw outlines by boundary
    boundary = np.zeros(lab.shape, bool)
    boundary[:-1] |= lab[:-1] != lab[1:]
    boundary[1:] |= lab[1:] != lab[:-1]
    boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    boundary[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    boundary &= lab > 0
    rgb[boundary] = [255, 60, 60]
    Image.fromarray(rgb).save(str(OUT / "classroom_detect.png"))
    print(f"  (saved detection overlay of {stem} central 800px crop)")


if __name__ == "__main__":
    main()
