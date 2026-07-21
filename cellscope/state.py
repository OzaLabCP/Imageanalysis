"""Shared application state and the signals that keep the four tabs in sync.

Views read from ``AppState`` and react to its signals; they never talk to each
other directly. Heavy work (well synthesis/loading, analysis) is dispatched to
the thread pool so the UI never blocks.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from cellscope.analysis import AnalysisSettings, WellAnalysis, run_analysis
from cellscope.data.loader import DatasetLoader, WellInfo
from cellscope.widgets.worker import run_async


class AppState(QObject):
    experimentLoaded = Signal()
    currentWellChanged = Signal(str)
    wellArrayLoaded = Signal(str)
    wellLoadFailed = Signal(str, str)
    busyChanged = Signal(bool, str)
    indicesChanged = Signal()
    channelModeChanged = Signal()
    displayChanged = Signal()
    selectedTrackChanged = Signal(int)
    selectionChanged = Signal()
    analysisStarted = Signal(str)
    analysisProgress = Signal(str, int)
    analysisFinished = Signal(str)
    analysisFailed = Signal(str, str)
    conditionsChanged = Signal()
    pixelSizeChanged = Signal(float)
    previewChanged = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.loader: DatasetLoader | None = None
        self.wells: list[WellInfo] = []
        self.current_well_id: str = ""
        self.current_array: np.ndarray | None = None  # (T, Z, C, Y, X)

        self.selected_wells: set[str] = set()
        self.t = 0
        self.z = 0
        self.active_channel = -1  # -1 = merge all visible channels
        self.channel_visible: list[bool] = []
        self.brightness: list[float] = []
        self.contrast: list[float] = []
        self.layers = {"image": True, "outlines": True, "tracks": True, "labels": True}
        self.selected_track = -1

        self.settings = AnalysisSettings()
        self.results: dict[str, WellAnalysis] = {}
        self.pixel_size_um: float = 1.0  # microns per FULL-RES pixel (ruler-calibratable)
        self.preview_on: bool = False    # fast (downsampled) preview mode
        self.preview_target: int = 700   # target max dimension in preview mode

        # Plate map: well_id -> condition name (insertion-ordered names give
        # each condition a stable color).
        self.conditions: dict[str, str] = {}
        self.condition_order: list[str] = []

        self._load_token = 0
        self._tasks: list = []
        self._running: set[str] = set()
        self._errors: dict[str, str] = {}

    # --- experiment / wells ----------------------------------------------
    def set_loader(self, loader: DatasetLoader) -> None:
        self.loader = loader
        self.wells = loader.list_wells()
        self.results.clear()
        self._errors.clear()
        self.selected_wells.clear()
        self.conditions.clear()
        self.condition_order.clear()
        self.selected_track = -1
        self.pixel_size_um = float(loader.pixel_size_um)

        n_chan = self.wells[0].n_channels if self.wells else 0
        self.channel_visible = [True] * n_chan
        self.brightness = [0.5] * n_chan
        self.contrast = [0.5] * n_chan
        self.active_channel = -1
        self.t = 0
        self.z = 0

        self.experimentLoaded.emit()
        if self.wells:
            self.set_current_well(self.wells[0].well_id)

    def current_well_info(self) -> WellInfo | None:
        for w in self.wells:
            if w.well_id == self.current_well_id:
                return w
        return None

    def set_current_well(self, well_id: str) -> None:
        if self.loader is None:
            return
        self.current_well_id = well_id
        self.selected_track = -1
        self.current_array = None
        self.currentWellChanged.emit(well_id)
        self.selectedTrackChanged.emit(-1)
        self._load_current_array()

    def current_downsample(self) -> int:
        """Resolution factor for loading/analysis: 1 (full) or N in preview mode."""
        if not self.preview_on:
            return 1
        info = self.current_well_info()
        if info is None:
            return 1
        return max(1, int(max(info.height, info.width) // self.preview_target))

    def _load_current_array(self) -> None:
        if self.loader is None or not self.current_well_id:
            return
        well_id = self.current_well_id
        self.current_array = None
        self._load_token += 1
        token = self._load_token
        downsample = self.current_downsample()
        note = " (fast preview)" if downsample > 1 else ""
        self.busyChanged.emit(True, f"Loading well {well_id}{note}...")

        def done(arr: np.ndarray) -> None:
            if token != self._load_token:
                return  # a newer load superseded this one
            self.current_array = arr
            self.t = min(self.t, arr.shape[0] - 1)
            self.z = min(self.z, arr.shape[1] - 1)
            self.busyChanged.emit(False, "")
            self.wellArrayLoaded.emit(well_id)
            self.indicesChanged.emit()

        def failed(msg: str) -> None:
            if token != self._load_token:
                return
            self.busyChanged.emit(False, "")
            self.wellLoadFailed.emit(well_id, msg)

        run_async(
            self.loader.get_well, well_id, downsample,
            on_done=done, on_failed=failed, registry=self._tasks,
        )

    def set_preview(self, on: bool) -> None:
        on = bool(on)
        if on == self.preview_on:
            return
        self.preview_on = on
        self.previewChanged.emit(on)
        self._load_current_array()  # reload the current well at the new resolution

    # --- dimensions / display --------------------------------------------
    def current_frame(self) -> np.ndarray | None:
        """The ``(C, Y, X)`` slice at the current time/Z, or None if not loaded."""
        if self.current_array is None:
            return None
        t = min(self.t, self.current_array.shape[0] - 1)
        z = min(self.z, self.current_array.shape[1] - 1)
        return self.current_array[t, z]

    def n_time(self) -> int:
        info = self.current_well_info()
        return info.n_time if info else 0

    def n_z(self) -> int:
        info = self.current_well_info()
        return info.n_z if info else 0

    def n_channels(self) -> int:
        info = self.current_well_info()
        return info.n_channels if info else 0

    def set_t(self, t: int) -> None:
        t = max(0, min(t, max(0, self.n_time() - 1)))
        if t != self.t:
            self.t = t
            self.indicesChanged.emit()

    def set_z(self, z: int) -> None:
        z = max(0, min(z, max(0, self.n_z() - 1)))
        if z != self.z:
            self.z = z
            self.indicesChanged.emit()

    def set_active_channel(self, channel: int) -> None:
        self.active_channel = channel
        self.channelModeChanged.emit()

    def set_channel_visible(self, channel: int, visible: bool) -> None:
        if 0 <= channel < len(self.channel_visible):
            self.channel_visible[channel] = visible
            self.displayChanged.emit()

    def set_brightness(self, channel: int, value: float) -> None:
        if 0 <= channel < len(self.brightness):
            self.brightness[channel] = value
            self.displayChanged.emit()

    def set_contrast(self, channel: int, value: float) -> None:
        if 0 <= channel < len(self.contrast):
            self.contrast[channel] = value
            self.displayChanged.emit()

    def set_layer(self, name: str, on: bool) -> None:
        if name in self.layers:
            self.layers[name] = on
            self.displayChanged.emit()

    def effective_visibility(self) -> list[bool]:
        """Per-channel visibility accounting for the active-channel selector."""
        n = len(self.channel_visible)
        if self.active_channel == -1:
            return list(self.channel_visible)
        return [i == self.active_channel for i in range(n)]

    # --- selection --------------------------------------------------------
    def set_selected_track(self, tid: int) -> None:
        if tid != self.selected_track:
            self.selected_track = tid
            self.selectedTrackChanged.emit(tid)

    def toggle_well_selected(self, well_id: str) -> None:
        if well_id in self.selected_wells:
            self.selected_wells.discard(well_id)
        else:
            self.selected_wells.add(well_id)
        self.selectionChanged.emit()

    # --- plate map / conditions ------------------------------------------
    def _prune_condition_order(self) -> None:
        # Drop names no longer used by any well so color indices stay compact
        # and distinct across the conditions actually in use.
        used = set(self.conditions.values())
        self.condition_order = [n for n in self.condition_order if n in used]

    def set_condition(self, well_ids, name: str) -> None:
        name = (name or "").strip()
        for wid in well_ids:
            if name:
                self.conditions[wid] = name
                if name not in self.condition_order:
                    self.condition_order.append(name)
            else:
                self.conditions.pop(wid, None)
        self._prune_condition_order()
        self.conditionsChanged.emit()

    def clear_conditions(self, well_ids) -> None:
        for wid in well_ids:
            self.conditions.pop(wid, None)
        self._prune_condition_order()
        self.conditionsChanged.emit()

    def condition_of(self, well_id: str) -> str:
        return self.conditions.get(well_id, "")

    def distinct_conditions(self) -> list[str]:
        return [n for n in self.condition_order if n in self.conditions.values()]

    def condition_color(self, name: str):
        from cellscope.colors import condition_color
        idx = self.condition_order.index(name) if name in self.condition_order else 0
        return condition_color(idx)

    # --- pixel-size calibration (ruler) ----------------------------------
    @staticmethod
    def _rescale_wa(wa: WellAnalysis, new_px: float) -> None:
        """Rescale one analysis's measurements to ``new_px`` microns/pixel.

        Areas/diameters are exact multiples of the pixel size (area with the
        ratio squared, lengths linearly), so we rescale rather than re-run.
        """
        old = wa.pixel_size_um if wa.pixel_size_um > 0 else 1.0
        if abs(new_px - old) < 1e-12:
            return
        ratio = new_px / old
        ratio2 = ratio * ratio
        for m in wa.measurements:
            m.area_um2 *= ratio2
            m.equiv_diameter_um *= ratio
            m.feret_diameter_um *= ratio
            m.length_major_um *= ratio
            m.length_minor_um *= ratio
            m.perimeter_um *= ratio
            # eccentricity and intensity stats are scale-invariant.
        wa.pixel_size_um = new_px

    def set_pixel_size_um(self, value: float) -> None:
        """Set microns-per-pixel and rescale any existing results in place."""
        value = float(value)
        if value <= 0 or abs(value - self.pixel_size_um) < 1e-9:
            return
        for wa in self.results.values():
            self._rescale_wa(wa, value * wa.downsample)
        self.pixel_size_um = value
        self.pixelSizeChanged.emit(value)

    # --- analysis ---------------------------------------------------------
    def is_running(self, well_id: str) -> bool:
        return well_id in self._running

    def analysis_for(self, well_id: str) -> WellAnalysis | None:
        return self.results.get(well_id)

    def error_for(self, well_id: str) -> str:
        return self._errors.get(well_id, "")

    def start_analysis(self, well_id: str) -> None:
        if self.loader is None or well_id in self._running:
            return
        self._running.add(well_id)
        self._errors.pop(well_id, None)
        self.analysisStarted.emit(well_id)

        def progress(p: int) -> None:
            self.analysisProgress.emit(well_id, p)

        def done(result: WellAnalysis) -> None:
            self._running.discard(well_id)
            self._errors.pop(well_id, None)
            # Reconcile scale: the pixel size may have been calibrated while this
            # ran, so bring the freshly-computed result to the current scale
            # (accounting for the resolution it was computed at).
            self._rescale_wa(result, self.pixel_size_um * result.downsample)
            self.results[well_id] = result
            self.analysisFinished.emit(well_id)

        def failed(msg: str) -> None:
            self._running.discard(well_id)
            self._errors[well_id] = msg
            self.analysisFailed.emit(well_id, msg)

        run_async(
            run_analysis,
            self.loader,
            well_id,
            self.settings.copy(),
            on_done=done,
            on_failed=failed,
            on_progress=progress,
            with_progress=True,
            pixel_size_um=self.pixel_size_um,
            downsample=self.current_downsample(),
            registry=self._tasks,
        )
