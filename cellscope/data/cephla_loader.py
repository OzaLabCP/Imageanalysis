"""Reader for Cephla / Squid multi-point time-lapse acquisitions.

These acquisitions are laid out as::

    <experiment>/
      acquisition parameters.json      # Nt, Nz, sensor pixel size, magnification
      coordinates.csv                  # region -> stage positions (FOVs)
      0/  1/  2/ ... 23/               # one numbered folder PER TIMEPOINT
        <region>_<fov>_<z>_<channel>.tiff

So the timepoint lives in the folder name, the channel is a descriptive string
(e.g. ``Fluorescence_488_nm_Ex``), and each region (well) is imaged at several
fields of view. CellScope models each **(region, FOV)** as one position: a
(Time, Z, Channel, Y, X) stack that can be viewed, tracked, and quantified.

The physical pixel size is read straight from the acquisition metadata
(sensor pixel size / magnification), so measurements come out in real microns
with no ruler calibration needed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellscope.data.loader import DatasetLoader, WellInfo

logger = logging.getLogger(__name__)

_WAVELENGTH_COLORS = [
    (405, (150, 130, 255)),
    (470, (90, 200, 255)),
    (488, (90, 230, 110)),
    (515, (140, 230, 90)),
    (561, (230, 220, 70)),
    (594, (255, 170, 70)),
    (638, (255, 80, 80)),
    (730, (255, 90, 200)),
    (750, (255, 90, 200)),
]


def _channel_color(name: str) -> tuple[int, int, int]:
    m = re.search(r"(\d{3,4})", name)
    if m:
        wl = int(m.group(1))
        nearest = min(_WAVELENGTH_COLORS, key=lambda kv: abs(kv[0] - wl))
        if abs(nearest[0] - wl) <= 40:
            return nearest[1]
    low = name.lower()
    if any(k in low for k in ("bf", "bright", "phase", "dic", "trans")):
        return (220, 220, 220)
    return (200, 200, 200)


def _channel_label(chan_key: str) -> str:
    """Turn 'Fluorescence_488_nm_Ex' into a short '488 nm' label."""
    m = re.search(r"(\d{3,4})\D*nm", chan_key, re.I)
    if m:
        return f"{m.group(1)} nm"
    return chan_key.replace("_", " ").strip()


@dataclass
class _Position:
    region: str
    fov: str


class CephlaLoader(DatasetLoader):
    @staticmethod
    def looks_like(folder: Path) -> bool:
        folder = Path(folder)
        if (folder / "acquisition parameters.json").exists():
            return True
        if (folder / "coordinates.csv").exists() and any(
            d.is_dir() and d.name.isdigit() for d in folder.iterdir()
        ):
            return True
        return False

    def __init__(self, folder: str) -> None:
        self._folder = Path(folder)
        self._name = self._folder.name or str(self._folder)
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_limit = 2  # positions are large; keep only a couple resident

        # Timepoint folders, numerically ordered.
        self._timepoints = sorted(
            (d for d in self._folder.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )
        if not self._timepoints:
            raise ValueError("No numbered timepoint folders found")

        params = self._read_params()
        self._pixel = self._compute_pixel_size(params)

        # Parse the first timepoint's filenames to learn regions/FOVs/Z/channels.
        regions, fovs_by_region, z_values, channels = self._scan(self._timepoints[0])
        if not channels:
            raise ValueError("No Cephla-style image files found")
        self._z_values = z_values
        self._channel_keys = channels
        self._channel_names = [_channel_label(c) for c in channels]
        self._channel_colors = [_channel_color(c) for c in channels]
        self._channel_wl = [
            (re.search(r"(\d{3,4})", c).group(1) if re.search(r"(\d{3,4})", c) else None)
            for c in channels
        ]

        self._height, self._width, self._dtype = self._probe_shape(regions, fovs_by_region)

        # One well per (region, FOV). Grid rows = regions, columns = FOV index.
        self._positions: dict[str, _Position] = {}
        self._wells: list[WellInfo] = []
        for r_idx, region in enumerate(regions):
            for fov_idx, fov in enumerate(fovs_by_region[region]):
                well_id = f"{region}-{fov}"
                self._positions[well_id] = _Position(region, fov)
                # FOV tokens are usually numeric, but non-numeric ones must not
                # crash the whole load - fall back to the enumeration index.
                try:
                    col = int(fov)
                except (TypeError, ValueError):
                    col = fov_idx
                self._wells.append(WellInfo(
                    well_id=well_id, row=r_idx, col=col,
                    n_time=len(self._timepoints), n_z=len(z_values),
                    n_channels=len(channels), height=self._height, width=self._width,
                ))

    # --- metadata ---------------------------------------------------------
    def _read_params(self) -> dict:
        path = self._folder / "acquisition parameters.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _compute_pixel_size(self, params: dict) -> float:
        # An explicit pixel_size_um wins (e.g. written by cellscope-downsample so
        # a reduced-resolution copy reports its true effective pixel size).
        explicit = params.get("pixel_size_um")
        try:
            if explicit and float(explicit) > 0:
                return float(explicit)
        except (TypeError, ValueError):
            pass
        sensor = params.get("sensor_pixel_size_um")
        obj = params.get("objective") or {}
        mag = obj.get("magnification") if isinstance(obj, dict) else None
        if not mag:
            mag = params.get("magnification")
        try:
            if sensor and mag:
                return float(sensor) / float(mag)
        except (TypeError, ValueError):
            pass
        return 1.0

    def _scan(self, tp_folder: Path):
        regions: list[str] = []
        fovs_by_region: dict[str, list[str]] = {}
        z_set: set[str] = set()
        channels: list[str] = []
        for f in sorted(tp_folder.iterdir(), key=lambda p: p.name):
            if f.suffix.lower() not in (".tif", ".tiff"):
                continue
            parts = f.stem.split("_", 3)
            if len(parts) < 4:
                continue
            region, fov, z, chan = parts[0], parts[1], parts[2], parts[3]
            if region not in fovs_by_region:
                fovs_by_region[region] = []
                regions.append(region)
            if fov not in fovs_by_region[region]:
                fovs_by_region[region].append(fov)
            z_set.add(z)
            if chan not in channels:
                channels.append(chan)
        regions.sort(key=_natural_key)
        for region in fovs_by_region:
            fovs_by_region[region].sort(key=lambda v: (len(v), v))
        z_values = sorted(z_set, key=lambda v: (len(v), v))
        return regions, fovs_by_region, z_values, channels

    def _probe_shape(self, regions, fovs_by_region):
        import tifffile
        region = regions[0]
        fov = fovs_by_region[region][0]
        path = self._timepoints[0] / f"{region}_{fov}_{self._z_values[0]}_{self._channel_keys[0]}.tiff"
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            shape = series.shape
            return int(shape[-2]), int(shape[-1]), np.dtype(series.dtype)

    # --- DatasetLoader interface -----------------------------------------
    def list_wells(self) -> list[WellInfo]:
        return list(self._wells)

    @property
    def name(self) -> str:
        return self._name

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
        key = (well_id, int(max(1, downsample)))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        arr = self._assemble(self._positions[well_id], int(max(1, downsample)))
        with self._lock:
            self._cache[key] = arr
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            return arr

    def get_thumbnail(self, well_id: str, max_size: int = 200) -> np.ndarray:
        # Prefer the acquisition's on-disk nav cache (tiny, instant); only fall
        # back to reading the first timepoint's images if it is missing.
        pos = self._positions[well_id]
        cache_dir = self._folder / "tile_cache"
        if cache_dir.is_dir():
            navs = []
            for wl in self._channel_wl:
                if wl is None:
                    navs = []
                    break
                matches = list(cache_dir.glob(f"nav_{pos.region}_fov_{pos.fov}_ch_{wl}_*.npy"))
                if not matches:
                    navs = []
                    break
                navs.append(matches[0])
            if len(navs) == len(self._channel_keys):
                # The nav cache is the intended cheap source; let a read error
                # propagate (the gallery card just stays blank) rather than fall
                # through to an expensive full-resolution read.
                return np.stack([np.load(str(p)) for p in navs])

        import tifffile
        ds = max(1, int(min(self._height, self._width) // max_size))
        planes = []
        for chan in self._channel_keys:
            p = self._timepoints[0] / f"{pos.region}_{pos.fov}_{self._z_values[0]}_{chan}.tiff"
            if not p.exists():
                planes.append(np.zeros((1, 1), dtype=self._dtype))
                continue
            img = tifffile.imread(str(p))
            if img.ndim == 3:
                img = img[..., 0]
            planes.append(img[::ds, ::ds])
        return np.stack(planes)

    def _assemble(self, pos: _Position, downsample: int = 1) -> np.ndarray:
        import tifffile
        ds = max(1, int(downsample))
        n_t = len(self._timepoints)
        n_z = len(self._z_values)
        n_c = len(self._channel_keys)
        yh = (self._height + ds - 1) // ds
        xw = (self._width + ds - 1) // ds
        arr = np.zeros((n_t, n_z, n_c, yh, xw), dtype=self._dtype)
        for ti, tp in enumerate(self._timepoints):
            for zi, z in enumerate(self._z_values):
                for ci, chan in enumerate(self._channel_keys):
                    path = tp / f"{pos.region}_{pos.fov}_{z}_{chan}.tiff"
                    if not path.exists():
                        logger.debug("missing plane, left blank: %s", path)
                        continue
                    try:
                        plane = tifffile.imread(str(path))
                    except Exception as exc:
                        logger.warning("unreadable plane, left blank: %s (%s)", path, exc)
                        continue
                    if plane.ndim == 3:
                        plane = plane[..., 0]
                    if ds > 1:
                        plane = plane[::ds, ::ds]
                    h = min(yh, plane.shape[0])
                    w = min(xw, plane.shape[1])
                    arr[ti, zi, ci, :h, :w] = plane[:h, :w]
        return arr


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]
