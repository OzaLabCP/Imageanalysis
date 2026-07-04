"""The dataset loading boundary.

Everything above this line (views, analysis) talks only to ``DatasetLoader``.
Today the only implementation is the synthetic ``MockLoader``. Later, real
readers built on ``tifffile`` / ``aicsimageio`` implement the same interface and
drop in without any UI changes.

The canonical in-memory layout for one well is a 5-D array:
``(Time, Z, Channel, Y, X)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WellInfo:
    """Lightweight description of one well / position on the plate."""

    well_id: str          # e.g. "A1"
    row: int              # 0-based plate row
    col: int              # 0-based plate column
    n_time: int
    n_z: int
    n_channels: int
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (self.n_time, self.n_z, self.n_channels, self.height, self.width)


class DatasetLoader(ABC):
    """Abstract source of plate image data.

    Implementations must be safe to call ``get_well`` from a worker thread.
    """

    # --- structure ---------------------------------------------------------
    @abstractmethod
    def list_wells(self) -> list[WellInfo]:
        """Return metadata for every well, in display order."""

    @abstractmethod
    def get_well(self, well_id: str, downsample: int = 1) -> np.ndarray:
        """Return the ``(T, Z, C, Y, X)`` array for one well.

        ``downsample`` (>= 1) decimates Y/X by that factor for a fast, lower
        resolution preview. May be slow (disk / synthesis); callers run it off
        the UI thread. Implementations should cache per (well, downsample).
        """

    # --- metadata ----------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-friendly name of the experiment (shown in the UI)."""

    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def channel_colors(self) -> list[tuple[int, int, int]]:
        """Display tint (R, G, B 0-255) for each channel."""

    @property
    @abstractmethod
    def pixel_size_um(self) -> float:
        """Physical size of one pixel in microns (for area in real units)."""

    # --- convenience -------------------------------------------------------
    def well_ids(self) -> list[str]:
        return [w.well_id for w in self.list_wells()]

    def get_well_info(self, well_id: str) -> WellInfo:
        for w in self.list_wells():
            if w.well_id == well_id:
                return w
        raise KeyError(f"No such well: {well_id!r}")

    def get_thumbnail(self, well_id: str, max_size: int = 200) -> np.ndarray:
        """Return a small ``(C, y, x)`` preview of a well's first frame.

        Default implementation loads a downsampled well; loaders with a cheaper
        source (e.g. an on-disk nav cache) should override this to avoid reading
        full-resolution data just for a thumbnail.
        """
        info = self.get_well_info(well_id)
        ds = max(1, int(min(info.height, info.width) // max_size))
        arr = self.get_well(well_id, downsample=ds)
        return arr[0, 0]
