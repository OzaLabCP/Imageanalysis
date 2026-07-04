"""Color helpers shared across the app.

Channel colors describe how each microscopy channel is tinted when rendered.
Track colors give every tracked cell a stable, visually distinct color derived
deterministically from its ID (so the same cell looks the same every session).
"""

from __future__ import annotations

import colorsys

# Default look-up colors for the mock dataset's channels.
# (R, G, B), 0-255. Nuclei read as a cool blue, the reporter as green.
DEFAULT_CHANNEL_COLORS: list[tuple[int, int, int]] = [
    (90, 170, 255),   # channel 0 - nuclei
    (120, 230, 130),  # channel 1 - reporter
    (255, 170, 90),   # channel 2 - spare
    (230, 120, 230),  # channel 3 - spare
]

DEFAULT_CHANNEL_NAMES: list[str] = ["Nuclei", "Reporter", "Channel 3", "Channel 4"]

# Distinct, legible tints for plate-map conditions (assigned in order).
CONDITION_PALETTE: list[tuple[int, int, int]] = [
    (0, 122, 255), (52, 199, 89), (255, 149, 0), (175, 82, 222),
    (255, 59, 48), (0, 199, 190), (255, 204, 0), (255, 45, 146),
    (90, 200, 250), (162, 132, 94), (48, 176, 199), (142, 142, 147),
]


def condition_color(index: int) -> tuple[int, int, int]:
    """Stable tint for the Nth distinct plate-map condition."""
    return CONDITION_PALETTE[index % len(CONDITION_PALETTE)]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable, distinct RGB color for a cell track ID.

    Uses golden-ratio hue spacing so consecutive IDs land far apart on the
    color wheel and the whole set stays easy to tell apart.
    """
    hue = (track_id * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def channel_names_for(n_channels: int) -> list[str]:
    names = list(DEFAULT_CHANNEL_NAMES[:n_channels])
    while len(names) < n_channels:
        names.append(f"Channel {len(names) + 1}")
    return names


def channel_colors_for(n_channels: int) -> list[tuple[int, int, int]]:
    colors = list(DEFAULT_CHANNEL_COLORS[:n_channels])
    while len(colors) < n_channels:
        # Cycle through hues for any extra channels.
        idx = len(colors)
        r, g, b = colorsys.hsv_to_rgb((idx * 0.27) % 1.0, 0.7, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors
