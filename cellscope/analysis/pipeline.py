"""End-to-end analysis for one well: detect -> track -> quantify.

Produces a ``WellAnalysis`` that the UI can render directly:
  * ``track_label_images`` (T, Y, X): each pixel holds the *track ID* of the cell
    it belongs to (0 = background). This makes overlays trivial - outline color
    and cell number come straight from the value, and identity is stable in time.
  * ``measurements``: a flat list of per-cell, per-frame rows for the table/CSV.
  * ``counts_per_frame``: cells detected at each timepoint, for the chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cellscope.analysis.quantify import measure_frame
from cellscope.analysis.segmentation import centroids_of, segment_frame
from cellscope.analysis.tracking import track_centroids
from cellscope.data.loader import DatasetLoader


@dataclass
class AnalysisSettings:
    """Everything that controls a run. Only ``sensitivity`` is surfaced by default;
    the rest live behind the Advanced sheet with sensible defaults."""

    sensitivity: float = 0.5      # the single primary control (0..1)
    smoothing: float = 1.5        # Gaussian sigma (px)
    min_size: int = 25            # drop specks smaller than this (px)
    seg_channel: int = 0          # channel used to find cells
    z: int = 0                    # Z slice analysed (M1 is 2D + time)
    max_distance: float = 30.0    # tracking gate (px)

    def copy(self) -> "AnalysisSettings":
        return AnalysisSettings(
            sensitivity=self.sensitivity,
            smoothing=self.smoothing,
            min_size=self.min_size,
            seg_channel=self.seg_channel,
            z=self.z,
            max_distance=self.max_distance,
        )


@dataclass
class CellMeasurement:
    well_id: str
    track_id: int
    frame: int
    centroid_x: float
    centroid_y: float
    area_px: int
    area_um2: float
    equiv_diameter_um: float
    feret_diameter_um: float
    mean_intensity: list[float]
    total_intensity: list[float]


@dataclass
class WellAnalysis:
    well_id: str
    n_time: int
    height: int
    width: int
    channel_names: list[str]
    pixel_size_um: float
    settings: AnalysisSettings
    track_label_images: np.ndarray                 # (T, Y, X) int32, valued by track ID
    tracks: dict[int, np.ndarray]                  # track ID -> (n, 3) of (frame, y, x)
    downsample: int = 1                            # resolution factor the analysis ran at
    measurements: list[CellMeasurement] = field(default_factory=list)
    counts_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def track_ids(self) -> list[int]:
        return sorted(self.tracks.keys())

    def measurements_for_track(self, track_id: int) -> list[CellMeasurement]:
        return [m for m in self.measurements if m.track_id == track_id]


def run_analysis(
    loader: DatasetLoader,
    well_id: str,
    settings: AnalysisSettings,
    progress_cb=None,
    pixel_size_um: float | None = None,
    downsample: int = 1,
) -> WellAnalysis:
    """Run the full pipeline for one well. ``progress_cb(percent)`` is optional.

    ``pixel_size_um`` is the FULL-RESOLUTION microns/pixel (overrides the
    loader's value when given). ``downsample`` runs the analysis at 1/N
    resolution for speed; the effective pixel size is scaled by N so areas and
    diameters still come out in real microns.
    """

    def report(pct: int) -> None:
        if progress_cb is not None:
            progress_cb(int(max(0, min(100, pct))))

    downsample = max(1, int(downsample))
    base_px = loader.pixel_size_um if pixel_size_um is None else float(pixel_size_um)
    px_size = base_px * downsample  # effective microns per (preview) pixel

    report(1)
    arr = loader.get_well(well_id, downsample=downsample)  # (T, Z, C, Y, X)
    n_time, n_z, n_chan, height, width = arr.shape
    z = int(np.clip(settings.z, 0, n_z - 1))
    seg_c = int(np.clip(settings.seg_channel, 0, n_chan - 1))

    # --- detect + collect centroids per frame -----------------------------
    frame_labels: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for t in range(n_time):
        frame = arr[t, z]  # (C, Y, X)
        labels = segment_frame(
            frame[seg_c],
            sensitivity=settings.sensitivity,
            smoothing=settings.smoothing,
            min_size=settings.min_size,
        )
        frame_labels.append(labels)
        centroids.append(centroids_of(labels))
        report(2 + int(58 * (t + 1) / n_time))  # 2..60

    # --- track ------------------------------------------------------------
    assignment, tracks = track_centroids(centroids, max_distance=settings.max_distance)
    report(64)

    # --- build track-ID label images + measurements -----------------------
    track_label_images = np.zeros((n_time, height, width), dtype=np.int32)
    measurements: list[CellMeasurement] = []
    counts_per_frame = np.zeros(n_time, dtype=int)

    for t in range(n_time):
        labels = frame_labels[t]
        a = assignment[t]
        if labels.max() > 0 and a.size > 0:
            # Local label value v (1..k) maps to track ID a[v-1].
            remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
            remap[1 : a.size + 1] = a
            track_labels = remap[labels]
        else:
            track_labels = labels.astype(np.int32)
        track_label_images[t] = track_labels

        frame = arr[t, z]  # (C, Y, X)
        per_cell = measure_frame(track_labels, frame, px_size)
        counts_per_frame[t] = len(per_cell)
        for track_id, m in per_cell.items():
            measurements.append(
                CellMeasurement(
                    well_id=well_id,
                    track_id=int(track_id),
                    frame=t,
                    centroid_x=m["centroid_x"],
                    centroid_y=m["centroid_y"],
                    area_px=m["area_px"],
                    area_um2=m["area_um2"],
                    equiv_diameter_um=m["equiv_diameter_um"],
                    feret_diameter_um=m["feret_diameter_um"],
                    mean_intensity=m["mean_intensity"],
                    total_intensity=m["total_intensity"],
                )
            )
        report(64 + int(34 * (t + 1) / n_time))  # 64..98

    report(100)
    return WellAnalysis(
        well_id=well_id,
        n_time=n_time,
        height=height,
        width=width,
        channel_names=loader.channel_names,
        pixel_size_um=px_size,
        settings=settings.copy(),
        track_label_images=track_label_images,
        tracks=tracks,
        downsample=downsample,
        measurements=measurements,
        counts_per_frame=counts_per_frame,
    )
