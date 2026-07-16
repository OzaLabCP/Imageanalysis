"""Per-cell measurements.

For each labeled region in a frame, compute area (pixels and microns squared),
size as equivalent and Feret (max-caliper) diameter, major/minor axis length,
perimeter, eccentricity, centroid, and per-channel intensity statistics (mean,
total, max, std, min). Uses vectorized ``scipy.ndimage`` region statistics plus
a small amount of per-region moment math; no scikit-image dependency.

Swap for richer feature extraction later; ``measure_frame`` is the contract.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist

try:  # ConvexHull is optional; we fall back to a brute-force max distance.
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError
except Exception:  # pragma: no cover
    ConvexHull = None

    class QhullError(Exception):
        pass


# Pixel-corner offsets so the caliper spans the object's true extent (edge to
# edge) rather than the distance between pixel centers, which undercounts by ~1px.
_CORNERS = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])

# scikit-image's perimeter estimator weights. A border image is convolved with
# the kernel below; each resulting code maps to a boundary-length contribution
# (orthogonal steps = 1, diagonal = sqrt(2), corners = (1 + sqrt(2)) / 2).
_PERIM_KERNEL = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]])
_PERIM_WEIGHTS = np.zeros(50, dtype=np.float64)
_PERIM_WEIGHTS[[5, 7, 15, 17, 25, 27]] = 1.0
_PERIM_WEIGHTS[[21, 33]] = np.sqrt(2.0)
_PERIM_WEIGHTS[[13, 23]] = (1.0 + np.sqrt(2.0)) / 2.0
_EROSION_STRUCT = ndi.generate_binary_structure(2, 1)


def _feret_max_px(coords: np.ndarray) -> float:
    """Maximum caliper (Feret) distance across a region, measured edge-to-edge.

    Hull the pixel centers first (cheap), then expand only those few extreme
    points to their pixel corners so the caliper measures true object extent
    (matches scikit-image feret_diameter_max) without hulling every pixel.
    """
    if coords.shape[0] == 0:
        return 0.0
    pts = coords
    if ConvexHull is not None and coords.shape[0] > 3:
        try:
            pts = coords[ConvexHull(coords).vertices]
        except (QhullError, ValueError):
            pts = coords  # collinear/degenerate -> use all points
    corners = (pts[:, None, :] + _CORNERS[None, :, :]).reshape(-1, 2)
    return float(pdist(corners).max()) if corners.shape[0] > 1 else 1.0


def _axis_and_eccentricity(coords: np.ndarray) -> tuple[float, float, float]:
    """Major/minor axis length (pixels) and eccentricity for a region.

    Uses the eigenvalues of the region's second central moments, the same
    formulation scikit-image uses (``axis = 4 * sqrt(eigenvalue)``), so an
    equivalent ellipse is fitted to the pixel cloud. Returns ``(major, minor,
    eccentricity)`` with ``eccentricity`` in ``[0, 1)`` (0 = circular).
    """
    n = coords.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    ys = coords[:, 0].astype(np.float64)
    xs = coords[:, 1].astype(np.float64)
    dy = ys - ys.mean()
    dx = xs - xs.mean()
    cyy = float((dy * dy).mean())
    cxx = float((dx * dx).mean())
    cyx = float((dy * dx).mean())
    # Eigenvalues of [[cyy, cyx], [cyx, cxx]].
    tr_half = 0.5 * (cyy + cxx)
    disc = max(0.0, tr_half * tr_half - (cyy * cxx - cyx * cyx))
    root = np.sqrt(disc)
    l1 = max(tr_half + root, 0.0)  # larger eigenvalue
    l2 = max(tr_half - root, 0.0)  # smaller eigenvalue
    major = 4.0 * np.sqrt(l1)
    minor = 4.0 * np.sqrt(l2)
    ecc = float(np.sqrt(1.0 - l2 / l1)) if l1 > 0 else 0.0
    return float(major), float(minor), ecc


def _perimeter_px(mask: np.ndarray) -> float:
    """scikit-image-style perimeter (in pixels) of a binary region patch."""
    m = mask.astype(bool)
    if not m.any():
        return 0.0
    eroded = ndi.binary_erosion(m, structure=_EROSION_STRUCT, border_value=0)
    border = (m & ~eroded).astype(np.int32)
    conv = ndi.convolve(border, _PERIM_KERNEL, mode="constant", cval=0)
    hist = np.bincount(conv.ravel(), minlength=50)[:50]
    return float(hist @ _PERIM_WEIGHTS)


def measure_frame(
    label_image: np.ndarray,
    intensity_cyx: np.ndarray,
    pixel_size_um: float,
) -> dict[int, dict]:
    """Measure every label in one frame.

    Parameters
    ----------
    label_image : ``(Y, X)`` int array. Nonzero values identify cells. The label
        value is used as the cell key (so passing a track-ID label image yields
        measurements keyed by track ID).
    intensity_cyx : ``(C, Y, X)`` array of raw intensities.
    pixel_size_um : microns per pixel.

    Returns
    -------
    dict mapping label/track ID -> measurement dict with keys
    ``area_px``, ``area_um2``, ``equiv_diameter_um``, ``feret_diameter_um``,
    ``length_major_um``, ``length_minor_um``, ``perimeter_um``,
    ``eccentricity``, ``centroid_y``, ``centroid_x``, and the per-channel lists
    ``mean_intensity``, ``total_intensity``, ``max_intensity``,
    ``std_intensity``, ``min_intensity``.
    """
    ids = np.unique(label_image)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}

    n_channels = intensity_cyx.shape[0]
    px = float(pixel_size_um)
    px_area = px ** 2

    counts = np.bincount(label_image.ravel())
    centroids = ndi.center_of_mass(
        np.ones_like(label_image, dtype=np.float32), label_image, index=ids
    )
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)

    means = np.zeros((ids.size, n_channels), dtype=np.float64)
    totals = np.zeros((ids.size, n_channels), dtype=np.float64)
    maxes = np.zeros((ids.size, n_channels), dtype=np.float64)
    stds = np.zeros((ids.size, n_channels), dtype=np.float64)
    mins = np.zeros((ids.size, n_channels), dtype=np.float64)
    # ndimage computes stats over the full label range internally; labels not in
    # our (present-only) ``ids`` divide 0/0 and warn harmlessly - silence that.
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in range(n_channels):
            chan = intensity_cyx[c].astype(np.float64)
            if not np.any(chan):
                # An all-zero plane is a missing / not-acquired channel image
                # (the loader zero-fills gaps). Emitting 0.0 would poison any
                # mean, so record it as absent (NaN) rather than as data.
                means[:, c] = totals[:, c] = np.nan
                maxes[:, c] = mins[:, c] = stds[:, c] = np.nan
                continue
            totals[:, c] = ndi.sum_labels(chan, label_image, index=ids)
            means[:, c] = ndi.mean(chan, label_image, index=ids)
            maxes[:, c] = ndi.maximum(chan, label_image, index=ids)
            mins[:, c] = ndi.minimum(chan, label_image, index=ids)
            stds[:, c] = ndi.standard_deviation(chan, label_image, index=ids)

    # Feret/axis/perimeter need each region's pixel coordinates; find_objects
    # gives a bounding box per label so we only scan the local patch.
    slices = ndi.find_objects(label_image)

    out: dict[int, dict] = {}
    for i, label in enumerate(ids):
        label = int(label)
        area_px = int(counts[label])
        equiv_diam_um = 2.0 * np.sqrt(area_px / np.pi) * px

        sl = slices[label - 1] if 0 < label <= len(slices) else None
        if sl is not None:
            patch = label_image[sl] == label
            coords = np.argwhere(patch)
            feret_um = _feret_max_px(coords) * px
            major_px, minor_px, ecc = _axis_and_eccentricity(coords)
            perim_um = _perimeter_px(patch) * px
        else:
            feret_um = equiv_diam_um
            major_px = minor_px = 0.0
            ecc = 0.0
            perim_um = 0.0

        out[label] = {
            "area_px": area_px,
            "area_um2": area_px * px_area,
            "equiv_diameter_um": float(equiv_diam_um),
            "feret_diameter_um": float(feret_um),
            "length_major_um": float(major_px * px),
            "length_minor_um": float(minor_px * px),
            "perimeter_um": float(perim_um),
            "eccentricity": float(ecc),
            "centroid_y": float(centroids[i, 0]),
            "centroid_x": float(centroids[i, 1]),
            "mean_intensity": [float(v) for v in means[i]],
            "total_intensity": [float(v) for v in totals[i]],
            "max_intensity": [float(v) for v in maxes[i]],
            "std_intensity": [float(v) for v in stds[i]],
            "min_intensity": [float(v) for v in mins[i]],
        }
    return out
