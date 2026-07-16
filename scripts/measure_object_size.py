"""Measure the typical object size in a real Cephla acquisition.

Reads a few fields, detects objects with the threshold segmenter, and reports
the distribution of object diameters (pixels and microns), so we can choose a
sensible downsample factor for Cellpose (which wants cells < ~100 px across).
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis.segmentation import segment_frame
from cellscope.data.cephla_loader import CephlaLoader

REAL = (r"C:\Users\joza\OneDrive - Cal Poly\Oza Lab - Shared 2"
        r"\synthetic-cell-imaging\tiff_files_for_imaging"
        r"\2026.06.26-gm-ppk2_2026-06-26_12-56-06.391058")

# A spread of fields across the plate; extras in case some fail to read.
FIELDS = ["B2-0", "B4-7", "B6-15", "B3-3", "B5-10", "B7-1"]


def main():
    if not CephlaLoader.looks_like(Path(REAL)):
        print("Real folder not accessible right now (OneDrive). Cannot measure.")
        return
    loader = CephlaLoader(REAL)
    px = loader.pixel_size_um
    tmid = loader._timepoints[len(loader._timepoints) // 2]
    print(f"pixel size: {px:.4f} um/px; image {loader._height}x{loader._width}; "
          f"channels {loader.channel_names}")

    all_diam_px = []
    read_ok = 0
    for wid in FIELDS:
        if wid not in loader._positions:
            continue
        pos = loader._positions[wid]
        for ci, chan in enumerate(loader._channel_keys):
            path = tmid / f"{pos.region}_{pos.fov}_{loader._z_values[0]}_{chan}.tiff"
            try:
                img = tifffile.imread(str(path))
            except Exception as exc:
                print(f"  skip {path.name}: {exc}")
                continue
            read_ok += 1
            labels = segment_frame(img, sensitivity=0.5, smoothing=1.5, min_size=12)
            counts = np.bincount(labels.ravel())
            sizes = counts[1:]  # drop background
            sizes = sizes[sizes > 0]
            if sizes.size == 0:
                print(f"  {wid} {loader.channel_names[ci]}: 0 objects")
                continue
            diam_px = 2.0 * np.sqrt(sizes / np.pi)
            all_diam_px.append(diam_px)
            print(f"  {wid} {loader.channel_names[ci]}: {sizes.size} objects, "
                  f"median diam {np.median(diam_px):.1f} px "
                  f"({np.median(diam_px)*px:.2f} um)")
        if read_ok >= 6:  # enough samples; stop hitting flaky I/O
            break

    if not all_diam_px:
        print("\nCould not read any images (OneDrive). Tell me the approx object "
              "diameter in microns and I'll pick the factor.")
        return

    d = np.concatenate(all_diam_px)
    p25, p50, p75, p90 = np.percentile(d, [25, 50, 75, 90])
    print(f"\n=== object diameter across {read_ok} fields, {d.size} objects ===")
    print(f"  median {p50:.1f} px  ({p50*px:.2f} um)")
    print(f"  25-75th {p25:.1f}-{p75:.1f} px  ({p25*px:.2f}-{p75*px:.2f} um)")
    print(f"  90th {p90:.1f} px ({p90*px:.2f} um)")
    # Recommend a factor that lands the median around ~45 px.
    target = 45.0
    factor = max(1, int(round(p50 / target)))
    landed = p50 / factor
    print(f"\nRecommended downsample: 1/{factor}  -> median ~{landed:.0f} px "
          f"({'good for Cellpose' if 20 <= landed <= 90 else 'reconsider'}). "
          f"90th percentile then ~{p90/factor:.0f} px.")
    if p50 < 60:
        print("NOTE: objects are already small; downsampling much would destroy "
              "them. Keep near full resolution.")


if __name__ == "__main__":
    main()
