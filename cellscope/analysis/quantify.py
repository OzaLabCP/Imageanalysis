"""Per-cell measurements.

For each labeled region in a frame, compute area (pixels and microns squared),
size as equivalent and Feret (max-caliper) diameter, centroid, and mean / total
intensity in every channel. Uses vectorized ``scipy.ndimage`` region statistics.

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
    ``centroid_y``, ``centroid_x``, ``mean_intensity`` (list per channel),
    ``total_intensity`` (list per channel).
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
    for c in range(n_channels):
        chan = intensity_cyx[c].astype(np.float64)
        totals[:, c] = ndi.sum(chan, label_image, index=ids)
        means[:, c] = ndi.mean(chan, label_image, index=ids)

    # Feret diameter needs each region's pixel coordinates; find_objects gives a
    # bounding box per label so we only scan the local patch.
    slices = ndi.find_objects(label_image)

    out: dict[int, dict] = {}
    for i, label in enumerate(ids):
        label = int(label)
        area_px = int(counts[label])
        equiv_diam_um = 2.0 * np.sqrt(area_px / np.pi) * px

        sl = slices[label - 1] if 0 < label <= len(slices) else None
        if sl is not None:
            coords = np.argwhere(label_image[sl] == label)
            feret_um = _feret_max_px(coords) * px
        else:
            feret_um = equiv_diam_um

        out[label] = {
            "area_px": area_px,
            "area_um2": area_px * px_area,
            "equiv_diameter_um": float(equiv_diam_um),
            "feret_diameter_um": float(feret_um),
            "centroid_y": float(centroids[i, 0]),
            "centroid_x": float(centroids[i, 1]),
            "mean_intensity": [float(v) for v in means[i]],
            "total_intensity": [float(v) for v in totals[i]],
        }
    return out
