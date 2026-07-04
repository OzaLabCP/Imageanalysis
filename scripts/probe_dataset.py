"""Probe a real image folder: report file info and how FolderLoader reads it.

Usage:  python scripts/probe_dataset.py "<folder>"
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.data.folder_loader import FolderLoader, _parse_tokens  # noqa: E402


def main():
    folder = sys.argv[1]
    files = sorted(Path(folder).glob("*"))
    imgs = [f for f in files if f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
    print(f"folder: {folder}")
    print(f"image files: {len(imgs)}")

    f0 = imgs[0]
    with Image.open(str(f0)) as im:
        print(f"first file: {f0.name}  mode={im.mode}  size={im.size}")
        arr = np.asarray(im)
    print(f"  array shape={arr.shape} dtype={arr.dtype} range=({int(arr.min())},{int(arr.max())})")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = (np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 1], arr[..., 2]))
        print(f"  RGB grayscale (R==G==B)? {gray}")

    print("\ntoken parse of each filename:")
    for f in imgs:
        print(f"  {f.stem:>12} -> {_parse_tokens(f.stem)}")

    print("\n--- FolderLoader result ---")
    loader = FolderLoader(folder)
    print("name:", loader.name)
    print("channels:", loader.channel_names, loader.channel_colors)
    print("pixel_size_um:", loader.pixel_size_um)
    for w in loader.list_wells():
        print(f"  well {w.well_id}: T={w.n_time} Z={w.n_z} C={w.n_channels} "
              f"YX={w.height}x{w.width}")
    w0 = loader.list_wells()[0]
    a = loader.get_well(w0.well_id)
    print(f"assembled well {w0.well_id}: shape={a.shape} dtype={a.dtype} "
          f"range=({int(a.min())},{int(a.max())})")


if __name__ == "__main__":
    main()
