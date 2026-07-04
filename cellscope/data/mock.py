"""Synthetic plate generator used for Milestone 1.

Produces a plausible multi-well, multi-timepoint, multi-channel dataset of
moving Gaussian-blob "cells" so the entire pipeline (segmentation, tracking,
quantification, charts, export) runs on real, non-faked data.

Design goals that make the downstream analysis meaningful:
  * cells DRIFT frame-to-frame (linear velocity + a small random walk) so the
    tracker has genuine motion to follow,
  * the reporter channel RAMPS over time per cell so quantification yields a
    real intensity curve,
  * every well differs (seeded per well), so the gallery and per-well results
    look distinct.
"""

from __future__ import annotations

import threading

import numpy as np

from cellscope.colors import channel_colors_for, channel_names_for
from cellscope.data.loader import DatasetLoader, WellInfo


class MockLoader(DatasetLoader):
    def __init__(
        self,
        n_wells: int = 6,
        n_time: int = 20,
        n_z: int = 1,
        n_channels: int = 2,
        size: int = 512,
        n_cells: int = 30,
        seed: int = 1234,
        pixel_size_um: float = 0.65,
        plate_cols: int = 3,
    ) -> None:
        if size < 16:
            raise ValueError("MockLoader size must be at least 16 pixels")
        self._n_wells = n_wells
        self._n_time = n_time
        self._n_z = n_z
        self._n_channels = n_channels
        self._size = size
        self._n_cells = n_cells
        self._seed = seed
        self._pixel = pixel_size_um
        self._plate_cols = plate_cols

        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

        self._wells: list[WellInfo] = []
        for i in range(n_wells):
            row = i // plate_cols
            col = i % plate_cols
            well_id = f"{chr(ord('A') + row)}{col + 1}"
            self._wells.append(
                WellInfo(
                    well_id=well_id,
                    row=row,
                    col=col,
                    n_time=n_time,
                    n_z=n_z,
                    n_channels=n_channels,
                    height=size,
                    width=size,
                )
            )

        self._channel_names = channel_names_for(n_channels)
        self._channel_colors = channel_colors_for(n_channels)

    # --- DatasetLoader interface ------------------------------------------
    def list_wells(self) -> list[WellInfo]:
        return list(self._wells)

    @property
    def name(self) -> str:
        return "Demo plate (synthetic)"

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def channel_colors(self) -> list[tuple[int, int, int]]:
        return list(self._channel_colors)

    @property
    def pixel_size_um(self) -> float:
        return self._pixel

    def get_well(self, well_id: str, downsample: int = 1) -> np.ndarray:
        with self._lock:
            full = self._cache.get(well_id)
        if full is None:
            # Generate outside the lock (CPU-bound, independent per well).
            index = next(i for i, w in enumerate(self._wells) if w.well_id == well_id)
            full = self._generate_well(index)
            with self._lock:
                full = self._cache.setdefault(well_id, full)
        if downsample and downsample > 1:
            return np.ascontiguousarray(full[:, :, :, ::downsample, ::downsample])
        return full

    # --- synthesis ---------------------------------------------------------
    def _generate_well(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed + index * 9973)
        size = self._size
        n_time = self._n_time
        n_z = self._n_z
        n_chan = self._n_channels

        # Vary the cell count per well so wells look genuinely different.
        n_cells = int(np.clip(self._n_cells + rng.integers(-6, 7), 8, self._n_cells + 8))

        # Keep the spawn margin well inside the image for small sizes.
        margin = float(min(36.0, max(2.0, size / 4.0)))
        start = rng.uniform(margin, size - margin, size=(n_cells, 2))  # (y, x)
        velocity = rng.normal(0.0, 1.1, size=(n_cells, 2))             # px / frame drift
        sigma = rng.uniform(5.0, 11.0, size=n_cells)                   # cell radius

        # Per-channel base amplitudes.
        amp_nuclei = rng.uniform(170.0, 250.0, size=n_cells)
        amp_reporter = rng.uniform(30.0, 140.0, size=n_cells)

        # Reporter dynamics: each cell ramps up over time at its own rate/phase.
        ramp_rate = rng.uniform(0.5, 1.4, size=n_cells)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)

        # Cumulative random walk so motion is not a perfectly straight line.
        steps = rng.normal(0.0, 0.55, size=(n_time, n_cells, 2))
        walk = np.cumsum(steps, axis=0)

        arr = np.zeros((n_time, n_z, n_chan, size, size), dtype=np.float32)

        denom = max(1, n_time - 1)
        for t in range(n_time):
            pos = start + velocity * t + walk[t]
            pos = np.clip(pos, 4.0, size - 5.0)
            for c in range(n_cells):
                cy, cx = float(pos[c, 0]), float(pos[c, 1])
                s = float(sigma[c])

                # Channel 0: nuclei, near-constant with slight breathing.
                if n_chan >= 1:
                    breathe = 1.0 + 0.05 * np.sin(0.5 * t + phase[c])
                    self._add_blob(arr[t, 0, 0], cy, cx, s, amp_nuclei[c] * breathe)

                # Channel 1: reporter, ramps up over the time course.
                if n_chan >= 2:
                    frac = t / denom
                    level = 0.25 + 0.75 * frac * ramp_rate[c]
                    level += 0.12 * np.sin(1.3 * t + phase[c])
                    level = max(0.05, level)
                    self._add_blob(arr[t, 0, 1], cy, cx, s * 0.9, amp_reporter[c] * level)

                # Any further channels: faint, slowly varying.
                for ch in range(2, n_chan):
                    self._add_blob(
                        arr[t, 0, ch], cy, cx, s,
                        60.0 * (0.5 + 0.5 * np.sin(0.7 * t + ch + phase[c])),
                    )

        # Replicate single Z generation across any extra Z slices (2D + time M1).
        if n_z > 1:
            for z in range(1, n_z):
                arr[:, z] = arr[:, 0]

        # Background offset + read noise so segmentation has a real threshold to find.
        arr += 8.0
        arr += rng.normal(0.0, 3.0, size=arr.shape).astype(np.float32)
        np.clip(arr, 0.0, None, out=arr)

        return arr.clip(0, 65535).astype(np.uint16)

    @staticmethod
    def _add_blob(plane: np.ndarray, cy: float, cx: float, sigma: float, amp: float) -> None:
        """Add one Gaussian blob into a 2-D plane using a local bounding box."""
        size_y, size_x = plane.shape
        r = int(np.ceil(sigma * 3.0))
        y0 = max(0, int(cy) - r)
        y1 = min(size_y, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(size_x, int(cx) + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        ly, lx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-(((ly - cy) ** 2 + (lx - cx) ** 2) / (2.0 * sigma * sigma)))
        plane[y0:y1, x0:x1] += (amp * g).astype(plane.dtype)
