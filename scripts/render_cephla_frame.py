"""Render one frame of a real Cephla position (2 channels) to a thumbnail."""

import sys
from pathlib import Path

import numpy as np
import tifffile
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
    pos = loader._positions["B2-0"]
    tp = loader._timepoints[len(loader._timepoints) // 2]  # a middle timepoint
    planes = []
    for chan in loader._channel_keys:
        p = tp / f"{pos.region}_{pos.fov}_{loader._z_values[0]}_{chan}.tiff"
        planes.append(tifffile.imread(str(p)))
    frame = np.stack(planes)  # (C, Y, X)
    print("frame", frame.shape, frame.dtype,
          "ch ranges", [(int(c.min()), int(c.max())) for c in frame])
    rgb = compose_rgb(frame, loader.channel_colors, [True] * len(planes), 0.5, 0.62)
    img = Image.fromarray(rgb).resize((700, 700), Image.BILINEAR)
    img.save(str(OUT / "cephla_B2-0.png"))
    print("saved cephla_B2-0.png")


if __name__ == "__main__":
    main()
