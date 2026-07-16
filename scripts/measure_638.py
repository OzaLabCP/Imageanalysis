"""Look at and measure the 638 nm 'red circles' channel of one field."""

import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis.segmentation import otsu_threshold
from cellscope.render import compose_rgb

FILE = Path(r"C:\Users\joza\OneDrive - Cal Poly\Oza Lab - Shared 2"
            r"\synthetic-cell-imaging\tiff_files_for_imaging"
            r"\classroom.04.24.26_2026-04-24_10-43-41.036911\0"
            r"\C2_6_0_Fluorescence_638_nm_Ex.tiff")
PX = 0.0925
OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def detect(img):
    """Threshold + fill ring interiors + drop specks -> labels."""
    sm = ndi.gaussian_filter(img.astype(np.float32), 1.5)
    mask = sm > otsu_threshold(sm)
    filled = ndi.binary_opening(ndi.binary_fill_holes(mask), iterations=1)
    labels, _ = ndi.label(filled)
    return labels


def diameters(labels):
    bbox_d, equiv_d = [], []
    for sl in ndi.find_objects(labels):
        if sl is None:
            continue
        area = int(np.sum(labels[sl] > 0))
        if area < 30:  # ignore residual noise
            continue
        bbox_d.append(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
        equiv_d.append(2.0 * np.sqrt(area / np.pi))
    return np.array(bbox_d), np.array(equiv_d)


def main():
    files = [FILE]
    for f in sorted(FILE.parent.glob("*_Fluorescence_638_nm_Ex.tiff")):
        if f != FILE and len(files) < 4:
            files.append(f)

    all_bbox, all_equiv = [], []
    for i, f in enumerate(files):
        try:
            img = tifffile.imread(str(f))
        except Exception as exc:
            print(f"  skip {f.name}: {exc}")
            continue
        labels = detect(img)
        bd, ed = diameters(labels)
        all_bbox.append(bd)
        all_equiv.append(ed)
        print(f"  {f.name}: {bd.size} circles, median outer "
              f"{np.median(bd):.0f}px ({np.median(bd)*PX:.1f}um)" if bd.size
              else f"  {f.name}: 0 circles")
        if i == 0:  # save a full-field overlay of the user's file to verify
            rgb = compose_rgb(img[None], [(255, 70, 70)], [True], 0.5, 0.66)
            b = np.zeros(labels.shape, bool)
            b[:-1] |= labels[:-1] != labels[1:]
            b[1:] |= labels[1:] != labels[:-1]
            b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
            b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            b &= labels > 0
            rgb[ndi.binary_dilation(b, iterations=2)] = [90, 230, 255]
            Image.fromarray(rgb[::4, ::4]).save(str(OUT / "c638_detect.png"))

    bbox = np.concatenate(all_bbox) if all_bbox else np.array([])
    equiv = np.concatenate(all_equiv) if all_equiv else np.array([])
    if not bbox.size:
        print("no circles detected")
        return
    print(f"\n=== {bbox.size} red circles across {len(all_bbox)} fields ===")
    for name, d in (("outer (bbox)", bbox), ("area-equivalent", equiv)):
        p10, p25, p50, p75, p90 = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"  {name:16s} median {p50:5.0f}px ({p50*PX:4.1f}um)   "
              f"10-90th {p10:.0f}-{p90:.0f}px ({p10*PX:.1f}-{p90*PX:.1f}um)")
    print("saved c638_full.png and c638_detect.png (full field /4 + outlines)")


if __name__ == "__main__":
    main()
