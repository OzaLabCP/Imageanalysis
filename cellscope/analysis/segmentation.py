"""Cell detection (segmentation).

Milestone 1 uses a real but simple pipeline:
    Gaussian smooth -> threshold (Otsu, nudged by a single Sensitivity control)
    -> connected-component labeling -> drop tiny specks.

Returns an int32 label image where 0 is background and 1..k are cells. Labels
are renumbered to be contiguous (1..k) so downstream code can index by label.

Swap this whole module for Cellpose/StarDist later; the signature is the contract.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def otsu_threshold(image: np.ndarray) -> float:
    """Otsu's method implemented on a 256-bin histogram (no scikit-image dep)."""
    flat = image[np.isfinite(image)].ravel()
    if flat.size == 0:
        return 0.0
    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmax <= vmin:
        return vmax
    hist, edges = np.histogram(flat, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return vmax

    weight0 = np.cumsum(hist)
    weight1 = total - weight0
    sum0 = np.cumsum(hist * centers)
    sum_total = sum0[-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean0 = sum0 / weight0
        mean1 = (sum_total - sum0) / weight1
        between = weight0 * weight1 * (mean0 - mean1) ** 2
    between[~np.isfinite(between)] = 0.0
    idx = int(np.argmax(between))
    return float(centers[idx])


def segment_frame(
    image2d: np.ndarray,
    sensitivity: float = 0.5,
    smoothing: float = 1.5,
    min_size: int = 25,
) -> np.ndarray:
    """Detect cells in one 2-D image.

    Parameters
    ----------
    sensitivity : 0..1. 0.5 is plain Otsu. Higher detects more / dimmer cells
        (lower threshold); lower is stricter. This is the only user-facing knob.
    smoothing : Gaussian sigma in pixels applied before thresholding.
    min_size : remove connected components smaller than this many pixels.
    """
    img = image2d.astype(np.float32)
    if smoothing and smoothing > 0:
        img = ndi.gaussian_filter(img, sigma=float(smoothing))

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.int32)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        # Flat image: nothing to detect.
        return np.zeros(img.shape, dtype=np.int32)

    threshold = otsu_threshold(img)
    # Nudge the threshold by sensitivity in NORMALIZED units (its position
    # between the image min and max), so the control direction is stable
    # regardless of the absolute sign of the data (real readers may subtract
    # background and produce negative values). sensitivity 0.0 -> stricter
    # (higher threshold), 1.0 -> looser (lower threshold).
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    pos = (threshold - vmin) / (vmax - vmin)
    factor = float(2.0 ** ((0.5 - sensitivity) * 2.0))
    pos = float(np.clip(pos * factor, 0.0, 1.0))
    threshold = vmin + pos * (vmax - vmin)

    mask = img > threshold

    labels, n = ndi.label(mask)
    if n == 0:
        return labels.astype(np.int32)

    if min_size and min_size > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = sizes >= int(min_size)
        # Renumber kept labels to a contiguous 1..k range.
        remap = np.zeros(n + 1, dtype=np.int32)
        kept_ids = np.nonzero(keep)[0]
        remap[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
        labels = remap[labels]

    return labels.astype(np.int32)


def centroids_of(labels: np.ndarray) -> np.ndarray:
    """Geometric centroids of each label, as an ``(k, 2)`` array of ``(y, x)``.

    Row ``i`` corresponds to label value ``i + 1`` (labels are contiguous 1..k).
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    coms = ndi.center_of_mass(np.ones_like(labels, dtype=np.float32), labels, index=ids)
    return np.asarray(coms, dtype=np.float64).reshape(-1, 2)
