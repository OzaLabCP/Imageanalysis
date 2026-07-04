"""Verify Cephla gallery thumbnails (nav cache) render on real data, and time them."""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.data.cephla_loader import CephlaLoader
from cellscope.render import compose_rgb

REAL = (r"C:\Users\joza\OneDrive - Cal Poly\Oza Lab - Shared 2"
        r"\synthetic-cell-imaging\tiff_files_for_imaging"
        r"\2026.06.26-gm-ppk2_2026-06-26_12-56-06.391058")
OUT = Path(__file__).resolve().parents[1] / ".preview"
OUT.mkdir(exist_ok=True)


def main():
    loader = CephlaLoader(REAL)
    t0 = time.time()
    thumbs = {wid: loader.get_thumbnail(wid) for wid in
              ["B2-0", "B2-8", "B4-3", "B7-15"]}
    dt = time.time() - t0
    print(f"4 thumbnails in {dt*1000:.0f} ms  (shapes {[t.shape for t in thumbs.values()]})")

    # Compose a 2x2 contact sheet.
    tiles = []
    for wid, frame in thumbs.items():
        rgb = compose_rgb(frame, loader.channel_colors, [True] * frame.shape[0], 0.5, 0.62)
        tiles.append(np.asarray(Image.fromarray(rgb).resize((260, 260), Image.BILINEAR)))
    top = np.concatenate(tiles[:2], axis=1)
    bot = np.concatenate(tiles[2:], axis=1)
    Image.fromarray(np.concatenate([top, bot], axis=0)).save(str(OUT / "cephla_thumbs.png"))
    print("saved cephla_thumbs.png")


if __name__ == "__main__":
    main()
