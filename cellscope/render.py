"""Turn numpy image data into Qt images for the canvas and thumbnails.

Pure rendering helpers shared by the Viewer canvas and the Wells gallery so
channel compositing looks identical everywhere. No app state here.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def _apply_brightness_contrast(norm: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Map a 0..1 normalized image through qualitative brightness/contrast.

    brightness 0.5 and contrast 0.5 are neutral (a straight auto-range).
    """
    gain = 0.25 + 3.5 * float(contrast)          # contrast 0.5 -> 2.0x
    offset = (float(brightness) - 0.5) * 1.2      # brightness shifts midpoint
    out = (norm - 0.5) * gain + 0.5 + offset
    return out


def compose_rgb(
    frame_cyx: np.ndarray,
    channel_colors,
    visible,
    brightness,
    contrast,
) -> np.ndarray:
    """Composite visible channels into an ``(H, W, 3)`` uint8 RGB image.

    Each channel is auto-ranged (1st..99.5th percentile), adjusted by its
    brightness/contrast, tinted by its color, and added together.
    ``brightness`` / ``contrast`` may be scalars or per-channel sequences.
    """
    n_chan, height, width = frame_cyx.shape
    acc = np.zeros((height, width, 3), dtype=np.float32)

    def _per_channel(seq, c, default):
        if seq is None:
            return default
        if np.isscalar(seq):
            return float(seq)
        return float(seq[c]) if c < len(seq) else default

    for c in range(n_chan):
        vis = True if visible is None else (c < len(visible) and bool(visible[c]))
        if not vis:
            continue
        chan = frame_cyx[c].astype(np.float32)
        finite = chan[np.isfinite(chan)]
        if finite.size == 0:
            continue  # nothing displayable in this channel
        lo, hi = np.percentile(finite, (1.0, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        norm = (chan - lo) / (hi - lo)
        # NaN/inf pixels map to black rather than poisoning the whole channel.
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = _apply_brightness_contrast(
            norm, _per_channel(brightness, c, 0.5), _per_channel(contrast, c, 0.5)
        )
        np.clip(norm, 0.0, 1.0, out=norm)

        color = channel_colors[c] if c < len(channel_colors) else (255, 255, 255)
        acc[..., 0] += norm * color[0]
        acc[..., 1] += norm * color[1]
        acc[..., 2] += norm * color[2]

    np.clip(acc, 0, 255, out=acc)
    return acc.astype(np.uint8)


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an ``(H, W, 3)`` uint8 array to a standalone QImage (owns its data)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    qimg = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return qimg.copy()  # detach from the numpy buffer


def build_color_lut(track_ids, color_fn) -> np.ndarray:
    """Build an ``(max_id + 1, 3)`` uint8 LUT mapping track ID -> RGB."""
    max_id = max(track_ids) if track_ids else 0
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for tid in track_ids:
        lut[tid] = color_fn(tid)
    return lut


def outline_overlay(track_labels: np.ndarray, lut: np.ndarray) -> QImage:
    """Build an RGBA overlay where cell boundaries are tinted by track color.

    A pixel is a boundary if any 4-neighbour belongs to a different label. Only
    foreground (label > 0) boundaries are drawn; background stays transparent.
    """
    lab = track_labels
    height, width = lab.shape
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:-1, :] |= lab[:-1, :] != lab[1:, :]
    boundary[1:, :] |= lab[1:, :] != lab[:-1, :]
    boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    boundary[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    boundary &= lab > 0

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    ys, xs = np.nonzero(boundary)
    if ys.size:
        ids = lab[ys, xs]
        ids = np.clip(ids, 0, lut.shape[0] - 1)
        rgba[ys, xs, 0:3] = lut[ids]
        rgba[ys, xs, 3] = 255

    rgba = np.ascontiguousarray(rgba)
    qimg = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
    return qimg.copy()
