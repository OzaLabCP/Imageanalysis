"""Read a folder of real TIFF / JPG images as a plate experiment.

This is the first real reader behind ``DatasetLoader``. It is built to "just
work" on the messy ways biologists actually save images, with no configuration:

  * A single multi-dimensional TIFF per well (OME-TIFF or ImageJ hyperstack):
    the file's own T/Z/C/Y/X axes are used. Each such file becomes one well.
  * A folder of 2-D image files named with tokens it recognizes:
    well (A1..H12 or well_<name>), time (t<N>/time<N>/frame<N>), channel
    (c<N>/ch<N>/channel<N> or fluor names like DAPI/GFP/RFP), z (z<N>).
    Files are grouped into wells and stacked along those axes.
  * A plain folder of frames with no tokens: treated as ONE well whose images
    are a time series, ordered by natural filename sort.

RGB images contribute three channels (Red/Green/Blue) unless a channel token is
present, in which case they are converted to grayscale for that channel.

Pixel data is read lazily: scanning only reads headers; full arrays are built
(and cached) when a well is opened.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cellscope.data.loader import DatasetLoader, WellInfo

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
TIFF_EXTS = {".tif", ".tiff"}

# Tokens may be separated (A1_t3_c0) or concatenated (3B4T0). The lookbehind
# (?<![A-Za-z]) lets a token start right after a digit (a numeric prefix) while
# still rejecting mid-word matches (e.g. the "t" in "patient"). Longer keywords
# are listed first so "time"/"channel" win over the bare "t"/"c".
WELL_RE = re.compile(r"(?<![A-Za-z])(?:well[_\-]?)?([A-P])(\d{1,2})(?![0-9])", re.I)
TIME_RE = re.compile(r"(?<![A-Za-z])(?:time|frame|tp|t)[_\-]?(\d{1,4})", re.I)
Z_RE = re.compile(r"(?<![A-Za-z])(?:zstep|slice|z)[_\-]?(\d{1,4})", re.I)
CHAN_RE = re.compile(r"(?<![A-Za-z])(?:channel|ch|c)[_\-]?(\d{1,3})(?![0-9])", re.I)

FLUOR_COLORS = {
    "dapi": (80, 120, 255), "hoechst": (80, 120, 255), "cfp": (0, 200, 255),
    "gfp": (90, 230, 110), "fitc": (90, 230, 110), "yfp": (180, 230, 60),
    "rfp": (255, 90, 90), "tritc": (255, 120, 60), "cy3": (255, 120, 60),
    "mcherry": (255, 90, 90), "texas": (255, 110, 70),
    "cy5": (255, 90, 200), "alexa647": (255, 90, 200), "farred": (255, 90, 200),
    "bf": (210, 210, 210), "brightfield": (210, 210, 210), "dic": (210, 210, 210),
    "trans": (210, 210, 210), "phase": (210, 210, 210),
    "gray": (235, 235, 235), "grey": (235, 235, 235),
    "red": (255, 90, 90), "green": (90, 230, 110), "blue": (90, 140, 255),
}
FLUOR_NAMES = ["dapi", "hoechst", "cfp", "gfp", "fitc", "yfp", "rfp", "tritc",
               "cy3", "cy5", "mcherry", "texas", "brightfield", "dic", "phase"]


class NoImagesFoundError(Exception):
    """Raised when a chosen folder contains no readable images."""


_PLATE_ID_RE = re.compile(r"^([A-P])(\d{1,2})$", re.I)


def _plate_position(well_id: str, fallback_index: int) -> tuple[int, int]:
    """Map a plate-style well id (e.g. 'B4') to (row, col); else stack vertically."""
    m = _PLATE_ID_RE.match(well_id.strip())
    if m:
        return ord(m.group(1).upper()) - ord("A"), int(m.group(2)) - 1
    return fallback_index, 0


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", name)]


@dataclass
class _PlanePlacement:
    t: int
    z: int
    c: int          # target channel index, or -1 for "split RGB across channels"
    path: str
    ext: str


@dataclass
class _WellPlan:
    well_id: str
    n_time: int
    n_z: int
    n_channels: int
    height: int
    width: int
    dtype: np.dtype
    channel_names: list[str]
    nd_path: str | None = None
    nd_axes: str | None = None
    planes: list[_PlanePlacement] = field(default_factory=list)


# --- header probing ------------------------------------------------------
@dataclass
class _Probe:
    path: str
    ext: str
    height: int
    width: int
    samples: int          # color samples per pixel (3 = RGB)
    dtype: np.dtype
    is_stack: bool        # has internal T/Z/C > 1
    axes: str | None      # tifffile axes string for stacks
    nd_t: int
    nd_z: int
    nd_c: int


def _probe_file(path: Path) -> _Probe | None:
    ext = path.suffix.lower()
    try:
        if ext in TIFF_EXTS:
            return _probe_tiff(path)
        return _probe_pil(path)
    except Exception:
        return None


def _map_axes(axes: str, shape) -> list:
    """Map each tifffile axis char to a canonical TZCYX slot, or None to drop.

    Y/X and real T/Z/C keep their identity. RGB samples (S) become C only if
    there is no real C axis. Any OTHER axis with length > 1 (e.g. tifffile's
    'Q'/'I' on a plain multi-page stack) is a genuine stack dimension and is
    assigned to the first free T/Z/C slot instead of being silently dropped.
    """
    axes = axes.upper()
    mapped: list = [None] * len(axes)
    used: set = set()
    for i, a in enumerate(axes):
        if a in "YXTZC":
            mapped[i] = a
            used.add(a)
    has_c = "C" in used
    for i, a in enumerate(axes):
        if mapped[i] is not None:
            continue
        if a == "S":
            if not has_c:
                mapped[i] = "C"
                used.add("C")
                has_c = True
            continue  # drop redundant samples when a real C already exists
        if shape[i] <= 1:
            continue  # singleton unknown axis -> drop
        for slot in "TZC":
            if slot not in used:
                mapped[i] = slot
                used.add(slot)
                break
    return mapped


def _axis_sizes(axes: str, shape) -> dict:
    sizes = {"T": 1, "Z": 1, "C": 1, "Y": 1, "X": 1}
    for m, n in zip(_map_axes(axes, shape), shape):
        if m is not None:
            sizes[m] = n
    return sizes


def _probe_tiff(path: Path) -> _Probe | None:
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        axes = series.axes.upper()
        shape = series.shape
        szmap = dict(zip(axes, shape))
        height = szmap.get("Y", 1)
        width = szmap.get("X", 1)
        samples = szmap.get("S", 1)

        # A stack has a real T/Z/C axis > 1, OR any non-sample extra axis > 1
        # (e.g. 'Q'/'I' for a metadata-less multi-page TIFF). Pure RGB 'S'
        # alone does NOT make it a stack (single RGB images stay planes).
        is_stack = any(
            (a in "TZC" and n > 1) or (a not in "YXSTZC" and n > 1)
            for a, n in zip(axes, shape)
        )
        if is_stack:
            sizes = _axis_sizes(axes, shape)
            nd_t, nd_z, nd_c = sizes["T"], sizes["Z"], sizes["C"]
        else:
            nd_t = nd_z = nd_c = 1

        dtype = np.dtype(series.dtype)
        return _Probe(str(path), path.suffix.lower(), height, width,
                      samples, dtype, is_stack, axes, nd_t, nd_z, nd_c)


def _probe_pil(path: Path) -> _Probe | None:
    from PIL import Image

    with Image.open(str(path)) as im:
        width, height = im.size
        mode = im.mode
    samples = 3 if mode in ("RGB", "RGBA", "P") else 1
    dtype = np.dtype(np.uint8)
    return _Probe(str(path), path.suffix.lower(), height, width,
                  samples, dtype, False, None, 1, 1, 1)


# --- token parsing -------------------------------------------------------
def _parse_tokens(stem: str) -> dict:
    # [A-P] already excludes the t/z/s prefixes; only plate row 'C' collides with
    # the channel prefix 'c'. We (a) never take the channel value from the well's
    # own row token, and (b) only reinterpret a leading 'C#' as a channel when it
    # is truly the sole token in an otherwise structureless name (e.g. img_c1).
    well = None
    well_start = None
    m = WELL_RE.search(stem)
    if m:
        letter = m.group(1).upper()
        col = int(m.group(2))
        if 1 <= col <= 24:
            well = f"{letter}{col}"
            well_start = m.start(1)  # position of the row letter, not any 'well' prefix

    t = _first_int(TIME_RE, stem)
    z = _first_int(Z_RE, stem)
    fluor = _detect_fluor(stem)

    chan_matches = list(CHAN_RE.finditer(stem))

    def overlaps_well(cm):
        return well_start is not None and abs(cm.start() - well_start) <= 1

    overlap = [cm for cm in chan_matches if overlaps_well(cm)]
    others = [cm for cm in chan_matches if not overlaps_well(cm)]

    if (well and well[0] == "C" and overlap and not others
            and well_start > 0 and t is None and z is None and fluor is None
            and not re.search(r"well", stem, re.I)):
        # A prefixed, isolated 'C#' with no other structure is really a channel.
        return {"well": None, "t": t, "z": z,
                "c": int(overlap[0].group(1)), "fluor": fluor}

    c = int(others[0].group(1)) if others else None
    return {"well": well, "t": t, "z": z, "c": c, "fluor": fluor}


def _first_int(regex: re.Pattern, text: str):
    m = regex.search(text)
    return int(m.group(1)) if m else None


def _detect_fluor(stem: str) -> str | None:
    low = stem.lower()
    for name in FLUOR_NAMES:
        if name in low:
            return name
    return None


def _channel_color(name: str, index: int) -> tuple[int, int, int]:
    low = name.lower().replace(" ", "")
    for key, color in FLUOR_COLORS.items():
        if key in low:
            return color
    from cellscope.colors import channel_colors_for
    return channel_colors_for(index + 1)[index]


# --- reading -------------------------------------------------------------
def _read_image(path: str, ext: str) -> np.ndarray:
    if ext in TIFF_EXTS:
        import tifffile
        return tifffile.imread(path)
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _to_tzcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reshape an array with tifffile ``axes`` into canonical (T, Z, C, Y, X).

    Unknown stack axes are remapped to T/Z/C (see ``_map_axes``) so no frames
    are lost; only singleton unknown axes are dropped.
    """
    mapped = _map_axes(axes, arr.shape)
    # Drop axes mapped to None (take index 0), from the last axis inward so
    # earlier indices stay valid.
    for i in range(len(mapped) - 1, -1, -1):
        if mapped[i] is None:
            arr = np.take(arr, 0, axis=i)
    kept = [m for m in mapped if m is not None]  # parallel to arr's axes now
    desired = [a for a in "TZCYX" if a in kept]
    arr = np.transpose(arr, [kept.index(a) for a in desired])
    for idx, a in enumerate("TZCYX"):
        if a not in kept:
            arr = np.expand_dims(arr, idx)
    return arr


def _to_gray(plane: np.ndarray) -> np.ndarray:
    if plane.ndim == 3:
        if plane.shape[2] <= 4:  # real color samples (RGB/RGBA)
            return plane[..., :3].mean(axis=2)
        return plane[..., 0]  # unexpected trailing axis -> take first, never average frames
    return plane


def _is_grayscale_rgb(path: str) -> bool:
    """True if an RGB(A) file is really grayscale (R == G == B everywhere)."""
    try:
        img = _read_image(path, Path(path).suffix.lower())
    except Exception:
        return False
    if img.ndim != 3 or img.shape[2] < 3:
        return False
    s = img[::4, ::4, :3]  # subsample for speed
    return bool(np.array_equal(s[..., 0], s[..., 1]) and np.array_equal(s[..., 1], s[..., 2]))


def _fit_into(dst_plane: np.ndarray, src: np.ndarray) -> None:
    """Place ``src`` into ``dst_plane`` (top-left), cropping/padding as needed."""
    h = min(dst_plane.shape[0], src.shape[0])
    w = min(dst_plane.shape[1], src.shape[1])
    dst_plane[:h, :w] = src[:h, :w]


class FolderLoader(DatasetLoader):
    def __init__(self, folder: str) -> None:
        self._folder = Path(folder)
        self._name = self._folder.name or str(self._folder)
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._pixel = 1.0

        files = self._discover()
        if not files:
            raise NoImagesFoundError(
                f"No TIFF/JPG images found in {self._folder}"
            )

        probes = [p for p in (_probe_file(f) for f in files) if p is not None]
        if not probes:
            raise NoImagesFoundError(
                f"Found files in {self._folder} but none could be read as images"
            )

        self._plans: dict[str, _WellPlan] = {}
        self._build_plans(probes)

        # Unify channels across wells so the UI's channel controls are consistent.
        global_c = max(p.n_channels for p in self._plans.values())
        richest = max(self._plans.values(), key=lambda p: p.n_channels)
        names = list(richest.channel_names)
        while len(names) < global_c:
            names.append(f"Channel {len(names) + 1}")
        self._channel_names = names[:global_c]
        self._channel_colors = [_channel_color(n, i) for i, n in enumerate(self._channel_names)]
        self._global_c = global_c

        self._pixel = self._infer_pixel_size(probes)

        self._wells: list[WellInfo] = []
        for i, (well_id, plan) in enumerate(sorted(self._plans.items(),
                                                   key=lambda kv: _natural_key(kv[0]))):
            row, col = _plate_position(well_id, i)
            self._wells.append(WellInfo(
                well_id=well_id, row=row, col=col,
                n_time=plan.n_time, n_z=plan.n_z, n_channels=global_c,
                height=plan.height, width=plan.width,
            ))

    # --- discovery & planning --------------------------------------------
    def _discover(self) -> list[Path]:
        if not self._folder.exists():
            return []
        # Always search recursively so nested per-well/per-position exports are
        # found even when a stray overview image sits at the top level. Ordered
        # by relative path so a plain folder of frames keeps its filename order.
        found = [
            p for p in self._folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        found.sort(key=lambda p: _natural_key(str(p.relative_to(self._folder))))
        return found

    def _build_plans(self, probes: list[_Probe]) -> None:
        stacks = [p for p in probes if p.is_stack]
        planes = [p for p in probes if not p.is_stack]

        used_ids: set[str] = set()
        for pr in stacks:
            stem = Path(pr.path).stem
            tok = _parse_tokens(stem)
            base = tok["well"] or stem
            well_id = self._unique_id(base, used_ids)
            is_rgb = pr.samples >= 3 and "C" not in (pr.axes or "")
            names = self._nd_channel_names(stem, pr.nd_c, is_rgb)
            self._plans[well_id] = _WellPlan(
                well_id=well_id, n_time=pr.nd_t, n_z=pr.nd_z, n_channels=pr.nd_c,
                height=pr.height, width=pr.width, dtype=pr.dtype,
                channel_names=names, nd_path=pr.path, nd_axes=pr.axes,
            )

        if planes:
            self._plan_planes(planes, used_ids)

    def _plan_planes(self, planes: list[_Probe], used_ids: set[str]) -> None:
        parsed = []
        for pr in planes:
            tok = _parse_tokens(Path(pr.path).stem)
            parsed.append((pr, tok))

        any_well = any(tok["well"] for _, tok in parsed)
        groups: dict[str, list] = {}
        if any_well:
            for pr, tok in parsed:
                groups.setdefault(tok["well"] or "unknown", []).append((pr, tok))
        else:
            groups[self._name or "well"] = parsed

        for base_id, items in groups.items():
            well_id = self._unique_id(base_id, used_ids)
            self._plans[well_id] = self._plan_one_well(well_id, items)

    def _plan_one_well(self, well_id: str, items: list) -> _WellPlan:
        height = max(pr.height for pr, _ in items)
        width = max(pr.width for pr, _ in items)
        dtype = np.result_type(*[pr.dtype for pr, _ in items])

        has_c = any(tok["c"] is not None for _, tok in items)
        fluors = sorted({tok["fluor"] for _, tok in items if tok["fluor"]})
        any_rgb = any(pr.samples >= 3 for pr, _ in items)

        if has_c:
            ctokens = sorted({tok["c"] for _, tok in items if tok["c"] is not None})
            cmap = {tok: i for i, tok in enumerate(ctokens)}
            n_channels = len(ctokens)
            rgb_split = False
            chan_names = self._token_channel_names(items, ctokens, cmap)

            def chan_of(tok):
                return cmap.get(tok["c"], 0)
        elif len(fluors) > 1:
            # Channels named only by fluorophore (e.g. A1_DAPI.tif, A1_GFP.tif).
            fmap = {f: i for i, f in enumerate(fluors)}
            n_channels = len(fluors)
            rgb_split = False
            chan_names = [f.title() for f in fluors]

            def chan_of(tok):
                return fmap.get(tok["fluor"], 0)
        elif any_rgb:
            sample = next((pr.path for pr, _ in items if pr.samples >= 3), None)
            if sample is not None and _is_grayscale_rgb(sample):
                # Color file that is really grayscale -> one channel.
                n_channels = 1
                rgb_split = False
                chan_names = ["Gray"]

                def chan_of(_tok):
                    return 0
            else:
                n_channels = 3
                rgb_split = True
                chan_names = ["Red", "Green", "Blue"]

                def chan_of(_tok):
                    return -1
        else:
            n_channels = 1
            rgb_split = False
            chan_names = ["Gray"]

            def chan_of(_tok):
                return 0

        zs = sorted({(tok["z"] if tok["z"] is not None else 0) for _, tok in items})
        zindex = {z: i for i, z in enumerate(zs)}
        n_z = len(zs)

        has_t = any(tok["t"] is not None for _, tok in items)
        placements: list[_PlanePlacement] = []

        if has_t:
            ts = sorted({(tok["t"] if tok["t"] is not None else 0) for _, tok in items})
            tindex = {t: i for i, t in enumerate(ts)}
            n_time = len(ts)
            for pr, tok in items:
                ti = tindex[tok["t"] if tok["t"] is not None else 0]
                zi = zindex[tok["z"] if tok["z"] is not None else 0]
                placements.append(_PlanePlacement(ti, zi, chan_of(tok), pr.path, pr.ext))
        else:
            # No time token: enumerate files within each (z, channel) bucket.
            buckets: dict[tuple, list] = {}
            for pr, tok in sorted(items, key=lambda it: _natural_key(Path(it[0].path).name)):
                zi = zindex[tok["z"] if tok["z"] is not None else 0]
                buckets.setdefault((zi, chan_of(tok)), []).append((pr, tok))
            n_time = max(len(v) for v in buckets.values())
            for (zi, ci), bucket in buckets.items():
                for ti, (pr, _tok) in enumerate(bucket):
                    placements.append(_PlanePlacement(ti, zi, ci, pr.path, pr.ext))

        return _WellPlan(
            well_id=well_id, n_time=n_time, n_z=n_z, n_channels=n_channels,
            height=height, width=width, dtype=dtype,
            channel_names=chan_names, planes=placements,
        )

    @staticmethod
    def _unique_id(base: str, used: set[str]) -> str:
        candidate = base
        n = 2
        while candidate in used:
            candidate = f"{base} ({n})"
            n += 1
        used.add(candidate)
        return candidate

    def _nd_channel_names(self, stem: str, n_c: int, is_rgb: bool = False) -> list[str]:
        fluor = _detect_fluor(stem)
        if is_rgb and n_c == 3 and fluor is None:
            return ["Red", "Green", "Blue"]  # genuine RGB samples only
        if n_c == 1 and fluor:
            return [fluor.title()]
        # A real multi-channel fluorescence stack has no per-channel name in the
        # single filename, so use neutral labels rather than Red/Green/Blue.
        return [f"Channel {i + 1}" for i in range(n_c)]

    def _token_channel_names(self, items, ctokens, cindex) -> list[str]:
        names = [f"Channel {i + 1}" for i in range(len(ctokens))]
        for _pr, tok in items:
            if tok["c"] is not None and tok["fluor"]:
                names[cindex[tok["c"]]] = tok["fluor"].title()
        return names

    def _infer_pixel_size(self, probes: list[_Probe]) -> float:
        for pr in probes:
            if pr.ext not in TIFF_EXTS:
                continue
            try:
                import tifffile
                with tifffile.TiffFile(pr.path) as tf:
                    page = tf.pages[0]
                    tags = page.tags
                    if "XResolution" in tags and tags["XResolution"].value:
                        num, den = tags["XResolution"].value
                        if num:
                            res = num / den  # pixels per unit
                            unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
                            per_px = (1.0 / res) if res else 1.0
                            if int(unit) == 3:      # centimeter
                                return per_px * 10_000.0
                            if int(unit) == 2:      # inch
                                return per_px * 25_400.0
                            return per_px
            except Exception:
                continue
        return 1.0

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
        with self._lock:
            full = self._cache.get(well_id)
        if full is None:
            # Assemble outside the lock (heavy disk I/O), then store under it.
            full = self._assemble(self._plans[well_id])
            with self._lock:
                full = self._cache.setdefault(well_id, full)
        if downsample and downsample > 1:
            return np.ascontiguousarray(full[:, :, :, ::downsample, ::downsample])
        return full

    def _assemble(self, plan: _WellPlan) -> np.ndarray:
        if plan.nd_path is not None:
            raw = _read_image(plan.nd_path, plan.nd_path[plan.nd_path.rfind("."):].lower())
            arr = _to_tzcyx(raw, plan.nd_axes or "YX")
            arr = self._pad_channels(arr)
            return np.ascontiguousarray(arr)

        out = np.zeros(
            (plan.n_time, plan.n_z, self._global_c, plan.height, plan.width),
            dtype=plan.dtype,
        )
        for pl in plan.planes:
            try:
                img = _read_image(pl.path, pl.ext)
            except Exception:
                continue
            if pl.c == -1:  # split RGB across channels
                if img.ndim == 2:
                    _fit_into(out[pl.t, pl.z, 0], img)
                else:
                    for k in range(min(3, img.shape[2], self._global_c)):
                        _fit_into(out[pl.t, pl.z, k], img[..., k])
            else:
                _fit_into(out[pl.t, pl.z, min(pl.c, self._global_c - 1)], _to_gray(img))
        return out

    def _pad_channels(self, arr: np.ndarray) -> np.ndarray:
        t, z, c, y, x = arr.shape
        if c == self._global_c:
            return arr
        padded = np.zeros((t, z, self._global_c, y, x), dtype=arr.dtype)
        padded[:, :, :min(c, self._global_c)] = arr[:, :, :min(c, self._global_c)]
        return padded
