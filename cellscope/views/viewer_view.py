"""Viewer tab: napari-style canvas + dimension sliders + layer toggles, with
the Detect Cells primary action and live progress.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellscope import theme
from cellscope.analysis import AnalysisSettings
from cellscope.render import compose_rgb, rgb_to_qimage
from cellscope.widgets.canvas import ImageCanvas
from cellscope.widgets.controls import (
    Card,
    Fab,
    Header,
    IconButton,
    LabeledSlider,
    SegmentedControl,
    ToggleSwitch,
    make_button,
)
from cellscope.widgets.sheet import BottomSheet  # noqa: F401  (type hint clarity)


class ViewerView(QWidget):
    def __init__(self, state, shell, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.shell = shell
        self._centroid_cache: dict[str, list[dict]] = {}
        self._syncing = False

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(120)
        self._play_timer.timeout.connect(self._advance_frame)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = Header("Viewer", "No well open")
        root.addWidget(self._header)

        # --- canvas ---
        self._canvas = ImageCanvas()
        self._canvas.cellClicked.connect(self.state.set_selected_track)
        self._canvas.rulerMeasured.connect(self._on_ruler_measured)
        self._ruler_active = False
        root.addWidget(self._canvas, 1)

        self._loading = QLabel("Loading...", self._canvas)
        self._loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading.setStyleSheet("color: #FFFFFF; font-size: 14px; background: transparent;")
        self._loading.hide()

        # --- controls card ---
        controls = Card(shadow=False)
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(16, 12, 16, 14)
        cl.setSpacing(10)

        # progress (hidden until analysis runs)
        self._progress_row = QWidget()
        pr = QHBoxLayout(self._progress_row)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.setSpacing(10)
        self._progress_label = QLabel("Detecting cells...")
        self._progress_label.setObjectName("Hint")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        pr.addWidget(self._progress_label)
        pr.addWidget(self._progress, 1)
        self._progress_row.hide()
        cl.addWidget(self._progress_row)

        # channel segmented control
        self._channel_seg = SegmentedControl(["Merge"])
        self._channel_seg.currentChanged.connect(self._on_channel_changed)
        cl.addWidget(self._channel_seg)

        # time row: play + slider + label
        time_row = QHBoxLayout()
        time_row.setSpacing(10)
        self._play_btn = IconButton("play", size=40)
        self._play_btn.clicked.connect(self._toggle_play)
        time_row.addWidget(self._play_btn)
        self._time_slider = self._make_dim_slider()
        self._time_slider.valueChanged.connect(lambda v: self.state.set_t(v))
        time_row.addWidget(self._time_slider, 1)
        self._time_label = QLabel("Time 1/1")
        self._time_label.setObjectName("Hint")
        self._time_label.setMinimumWidth(78)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        time_row.addWidget(self._time_label)
        cl.addLayout(time_row)

        # focus (Z) + well rows
        self._focus_slider, self._focus_label, focus_row = self._labeled_dim_row("Focus")
        self._focus_slider.valueChanged.connect(lambda v: self.state.set_z(v))
        cl.addLayout(focus_row)

        self._well_slider, self._well_label, well_row = self._labeled_dim_row("Well")
        self._well_slider.valueChanged.connect(self._on_well_slider)
        cl.addLayout(well_row)

        # Fast preview: load/detect at reduced resolution for big datasets.
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)
        pv_label = QLabel("Fast preview")
        pv_label.setObjectName("Hint")
        preview_row.addWidget(pv_label)
        self._preview_hint = QLabel("")
        self._preview_hint.setObjectName("Hint")
        preview_row.addWidget(self._preview_hint)
        preview_row.addStretch(1)
        self._preview_switch = ToggleSwitch()
        self._preview_switch.setChecked(self.state.preview_on)
        self._preview_switch.toggled.connect(self.state.set_preview)
        preview_row.addWidget(self._preview_switch)
        cl.addLayout(preview_row)

        # layer toggles + display/advanced buttons
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(14)
        self._layer_switches = {}
        for name, label in (("image", "Image"), ("outlines", "Outlines"),
                            ("tracks", "Tracks"), ("labels", "IDs")):
            box = QVBoxLayout()
            box.setSpacing(2)
            box.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            sw = ToggleSwitch()
            sw.setChecked(self.state.layers[name])
            sw.toggled.connect(lambda on, n=name: self.state.set_layer(n, on))
            cap = QLabel(label)
            cap.setObjectName("Hint")
            cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            box.addWidget(sw, 0, Qt.AlignmentFlag.AlignHCenter)
            box.addWidget(cap)
            bottom_row.addLayout(box)
            self._layer_switches[name] = sw
        bottom_row.addStretch(1)
        self._scale_btn = make_button("Scale", "ghost")
        self._scale_btn.clicked.connect(self._toggle_ruler)
        bottom_row.addWidget(self._scale_btn)
        display_btn = make_button("Display", "ghost")
        display_btn.clicked.connect(self._open_display_sheet)
        bottom_row.addWidget(display_btn)
        cl.addLayout(bottom_row)

        root.addWidget(controls)

        # --- floating action button ---
        self._fab = Fab("Detect Cells", self)
        self._fab.clicked.connect(self._open_detect_sheet)
        self._fab.raise_()

        # --- wiring ---
        state.experimentLoaded.connect(self._on_experiment_loaded)
        state.currentWellChanged.connect(self._on_well_changed)
        state.wellArrayLoaded.connect(lambda _wid: self._refresh_canvas())
        state.busyChanged.connect(self._on_busy)
        state.indicesChanged.connect(self._on_indices_changed)
        state.channelModeChanged.connect(self._refresh_canvas)
        state.displayChanged.connect(self._on_display_changed)
        state.selectedTrackChanged.connect(self._on_selected_changed)
        state.analysisStarted.connect(self._on_analysis_started)
        state.analysisProgress.connect(self._on_analysis_progress)
        state.analysisFinished.connect(self._on_analysis_finished)
        state.analysisFailed.connect(self._on_analysis_failed)
        state.pixelSizeChanged.connect(lambda _v: self._update_subtitle())
        state.previewChanged.connect(self._on_preview_changed)

    # --- small builders ---------------------------------------------------
    def _make_dim_slider(self):
        from PySide6.QtWidgets import QSlider
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 0)
        return s

    def _labeled_dim_row(self, title: str):
        from PySide6.QtWidgets import QSlider
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl_title = QLabel(title)
        lbl_title.setObjectName("Hint")
        lbl_title.setMinimumWidth(40)
        row.addWidget(lbl_title)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        row.addWidget(slider, 1)
        value_lbl = QLabel("-")
        value_lbl.setObjectName("Hint")
        value_lbl.setMinimumWidth(78)
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value_lbl)
        return slider, value_lbl, row

    # --- layout -----------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fab.adjustSize()
        margin = 22
        x = self.width() - self._fab.width() - margin
        y = self._canvas.geometry().bottom() - self._fab.height() - margin
        self._fab.move(max(0, x), max(0, y))
        self._loading.setGeometry(0, 0, self._canvas.width(), self._canvas.height())

    # --- experiment / well ------------------------------------------------
    def _on_experiment_loaded(self) -> None:
        names = self.state.loader.channel_names if self.state.loader else []
        self._channel_seg.set_segments(list(names) + ["Merge"])
        self._channel_seg.set_current_index(len(names), emit=False)  # Merge

        n_wells = len(self.state.wells)
        self._syncing = True
        self._well_slider.setRange(0, max(0, n_wells - 1))
        self._well_slider.setEnabled(n_wells > 1)
        self._syncing = False

    def _on_well_changed(self, well_id: str) -> None:
        self._stop_play()
        self._end_ruler()
        info = self.state.current_well_info()
        self._update_subtitle()

        self._syncing = True
        if info:
            self._time_slider.setRange(0, max(0, info.n_time - 1))
            self._time_slider.setEnabled(info.n_time > 1)
            self._play_btn.setEnabled(info.n_time > 1)
            self._focus_slider.setRange(0, max(0, info.n_z - 1))
            self._focus_slider.setEnabled(info.n_z > 1)
            ids = [w.well_id for w in self.state.wells]
            idx = ids.index(well_id) if well_id in ids else 0
            self._well_slider.setValue(idx)
        self._syncing = False
        self._sync_dim_labels()

    def _on_busy(self, busy: bool, message: str) -> None:
        self._loading.setText(message or "Loading...")
        self._loading.setVisible(busy)
        if busy:
            self._loading.raise_()

    # --- dimension syncing ------------------------------------------------
    def _on_indices_changed(self) -> None:
        self._syncing = True
        self._time_slider.setValue(self.state.t)
        self._focus_slider.setValue(self.state.z)
        self._syncing = False
        self._sync_dim_labels()
        self._refresh_canvas()

    def _sync_dim_labels(self) -> None:
        n_t = self.state.n_time()
        n_z = self.state.n_z()
        self._time_label.setText(f"Time {self.state.t + 1}/{max(1, n_t)}")
        self._focus_label.setText(
            f"{self.state.z + 1}/{n_z}" if n_z > 1 else "Single plane"
        )
        self._well_label.setText(f"Well {self.state.current_well_id or '-'}")

    def _on_well_slider(self, value: int) -> None:
        if self._syncing:
            return
        if 0 <= value < len(self.state.wells):
            self.state.set_current_well(self.state.wells[value].well_id)

    # --- play -------------------------------------------------------------
    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._stop_play()
        elif self.state.n_time() > 1:
            self._play_timer.start()
            self._play_btn.set_icon_name("pause")

    def _stop_play(self) -> None:
        self._play_timer.stop()
        self._play_btn.set_icon_name("play")

    def _advance_frame(self) -> None:
        n_t = self.state.n_time()
        if n_t <= 1:
            self._stop_play()
            return
        self.state.set_t((self.state.t + 1) % n_t)

    # --- channels / display -----------------------------------------------
    def _on_channel_changed(self, index: int) -> None:
        n = self.state.n_channels()
        self.state.set_active_channel(-1 if index >= n else index)

    def _on_display_changed(self) -> None:
        for name, sw in self._layer_switches.items():
            if sw.isChecked() != self.state.layers[name]:
                sw.blockSignals(True)
                sw.setChecked(self.state.layers[name])
                sw.blockSignals(False)
        self._refresh_canvas()

    def _on_selected_changed(self, tid: int) -> None:
        self._canvas.set_selected_track(tid)

    # --- canvas refresh ---------------------------------------------------
    def _frame_centroids(self, well_id: str):
        if well_id not in self._centroid_cache:
            wa = self.state.analysis_for(well_id)
            if wa is None:
                return []
            per_frame: list[dict] = [dict() for _ in range(wa.n_time)]
            for m in wa.measurements:
                if 0 <= m.frame < wa.n_time:
                    per_frame[m.frame][m.track_id] = (m.centroid_y, m.centroid_x)
            self._centroid_cache[well_id] = per_frame
        return self._centroid_cache[well_id]

    def _refresh_canvas(self) -> None:
        frame = self.state.current_frame()
        if frame is None:
            self._canvas.set_base_image(None)
            self._canvas.set_label_image(None)
            self._canvas.set_tracks({})
            self._canvas.set_frame_centroids({})
            return
        if not self.state.layers["image"]:
            # Still show a black canvas with overlays if the image layer is off.
            import numpy as np
            blank = np.zeros((frame.shape[1], frame.shape[2], 3), dtype=np.uint8)
            self._canvas.set_base_image(rgb_to_qimage(blank))
        else:
            rgb = compose_rgb(
                frame,
                self.state.loader.channel_colors,
                self.state.effective_visibility(),
                self.state.brightness,
                self.state.contrast,
            )
            self._canvas.set_base_image(rgb_to_qimage(rgb))

        wa = self.state.analysis_for(self.state.current_well_id)
        t = self.state.t
        # Overlays only align when the analysis ran at the resolution now shown.
        overlay_ok = (wa is not None and wa.downsample == self.state.current_downsample()
                      and 0 <= t < wa.track_label_images.shape[0])
        if overlay_ok:
            self._canvas.set_label_image(wa.track_label_images[t])
            self._canvas.set_tracks(wa.tracks)
            per_frame = self._frame_centroids(self.state.current_well_id)
            self._canvas.set_frame_centroids(per_frame[t] if t < len(per_frame) else {})
        else:
            self._canvas.set_label_image(None)
            self._canvas.set_tracks({})
            self._canvas.set_frame_centroids({})

        self._canvas.set_current_t(t)
        self._canvas.set_selected_track(self.state.selected_track)
        self._canvas.set_layers(
            self.state.layers["outlines"],
            self.state.layers["tracks"],
            self.state.layers["labels"],
        )

    # --- analysis lifecycle ----------------------------------------------
    def _on_analysis_started(self, well_id: str) -> None:
        if well_id != self.state.current_well_id:
            return
        self._progress.setValue(0)
        self._progress_label.setText("Detecting cells...")
        self._progress_row.show()
        self._fab.setEnabled(False)
        self._fab.setText("Working...")

    def _on_analysis_progress(self, well_id: str, pct: int) -> None:
        if well_id == self.state.current_well_id:
            self._progress.setValue(pct)

    def _on_analysis_finished(self, well_id: str) -> None:
        self._centroid_cache.pop(well_id, None)
        if well_id == self.state.current_well_id:
            self._progress_row.hide()
            self._fab.setEnabled(True)
            self._fab.setText("Detect Cells")
            self._refresh_canvas()
            wa = self.state.analysis_for(well_id)
            if wa and wa.n_tracks == 0:
                self.shell.toast("No cells found. Try raising Sensitivity or "
                                 "picking another channel in Advanced.")
            elif wa:
                self.shell.toast(f"Found {wa.n_tracks} cells in well {well_id}")

    def _on_analysis_failed(self, well_id: str, message: str) -> None:
        if well_id == self.state.current_well_id:
            self._progress_row.hide()
            self._fab.setEnabled(True)
            self._fab.setText("Detect Cells")
            self.shell.toast(f"Detection failed: {message}")

    # --- detect sheet -----------------------------------------------------
    def _open_detect_sheet(self) -> None:
        if not self.state.current_well_id:
            self.shell.toast("Open a well first")
            return
        content, read_settings = self._build_detect_content()
        self._detect_reader = read_settings
        self.shell.present_sheet("Detect Cells", content)

    def _build_detect_content(self):
        s = self.state.settings
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(20, 4, 20, 22)
        v.setSpacing(16)

        intro = QLabel("Find and outline cells, then follow them across time. "
                       "Use one slider, run, and refine if needed.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        sensitivity = LabeledSlider("Sensitivity", "Fewer cells", "More cells", s.sensitivity)
        v.addWidget(sensitivity)

        adv_btn = make_button("Advanced options", "ghost")
        v.addWidget(adv_btn, 0, Qt.AlignmentFlag.AlignLeft)

        adv = QWidget()
        av = QVBoxLayout(adv)
        av.setContentsMargins(0, 0, 0, 0)
        av.setSpacing(14)

        names = self.state.loader.channel_names if self.state.loader else ["Channel 1"]
        chan_label = QLabel("Detect on channel")
        chan_label.setObjectName("SectionLabel")
        av.addWidget(chan_label)
        chan_seg = SegmentedControl(list(names))
        chan_seg.set_current_index(min(s.seg_channel, len(names) - 1), emit=False)
        av.addWidget(chan_seg)

        smoothing = LabeledSlider("Smoothing", "Sharp", "Smooth", s.smoothing / 4.0)
        av.addWidget(smoothing)
        min_size = LabeledSlider("Ignore specks", "Keep small", "Larger only",
                                 min(1.0, s.min_size / 120.0))
        av.addWidget(min_size)
        distance = LabeledSlider("Tracking distance", "Strict", "Loose",
                                 (s.max_distance - 5.0) / 75.0)
        av.addWidget(distance)
        adv.setVisible(False)
        v.addWidget(adv)

        def toggle_adv() -> None:
            adv.setVisible(not adv.isVisible())
            adv_btn.setText("Hide advanced options" if adv.isVisible() else "Advanced options")
        adv_btn.clicked.connect(toggle_adv)

        run_btn = make_button("Run detection", "primary")
        run_btn.clicked.connect(self._run_detection)
        v.addWidget(run_btn)

        def read_settings() -> AnalysisSettings:
            return AnalysisSettings(
                sensitivity=sensitivity.value(),
                smoothing=smoothing.value() * 4.0,
                min_size=int(min_size.value() * 120),
                seg_channel=chan_seg.current_index(),
                z=self.state.z,
                max_distance=5.0 + distance.value() * 75.0,
            )

        return wrap, read_settings

    def _run_detection(self) -> None:
        if hasattr(self, "_detect_reader"):
            self.state.settings = self._detect_reader()
        self.shell.dismiss_sheet()
        self.state.start_analysis(self.state.current_well_id)

    # --- ruler / scale calibration ---------------------------------------
    def _update_subtitle(self) -> None:
        info = self.state.current_well_info()
        if not info:
            self._header.set_subtitle("No well open")
            self._preview_hint.setText("")
            return
        parts = [f"Well {self.state.current_well_id}",
                 f"{self.state.pixel_size_um:.3g} um/px"]
        ds = self.state.current_downsample()
        if self.state.preview_on and ds > 1:
            parts.append(f"fast preview 1/{ds}")
        self._header.set_subtitle(" · ".join(parts))
        potential = max(1, int(max(info.height, info.width) // self.state.preview_target))
        self._preview_hint.setText(f"1/{potential}" if potential > 1 else "")

    def _on_preview_changed(self, on: bool) -> None:
        if self._preview_switch.isChecked() != on:
            self._preview_switch.blockSignals(True)
            self._preview_switch.setChecked(on)
            self._preview_switch.blockSignals(False)
        self._update_subtitle()

    def _toggle_ruler(self) -> None:
        if not self.state.current_well_id:
            self.shell.toast("Open a well first")
            return
        self._ruler_active = not self._ruler_active
        self._canvas.set_ruler_mode(self._ruler_active)
        if self._ruler_active:
            self._scale_btn.setText("Measuring...")
            self.shell.toast("Drag a line across a known distance, e.g. the scale bar")
        else:
            self._scale_btn.setText("Scale")
            self._canvas.clear_ruler()

    def _end_ruler(self) -> None:
        self._ruler_active = False
        self._scale_btn.setText("Scale")
        self._canvas.set_ruler_mode(False)
        self._canvas.clear_ruler()

    def _on_ruler_measured(self, px_len: float) -> None:
        if px_len < 2:
            self.shell.toast("Draw a longer line")
            return
        self.shell.present_sheet("Set scale", self._build_scale_content(px_len))

    def _build_scale_content(self, px_len: float):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(20, 4, 20, 22)
        v.setSpacing(14)

        intro = QLabel(
            "Enter the real length of the line you just drew. CellScope converts "
            "pixels to microns for every area and diameter measurement."
        )
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        measured = QLabel(f"Measured line: {px_len:.0f} pixels")
        measured.setObjectName("CardTitle")
        v.addWidget(measured)

        row = QHBoxLayout()
        field = QLineEdit()
        field.setValidator(QDoubleValidator(0.0001, 1e7, 4))
        field.setPlaceholderText("Length")
        row.addWidget(field, 1)
        unit = QLabel("microns")
        unit.setObjectName("Hint")
        row.addWidget(unit)
        v.addLayout(row)

        current = QLabel(f"Current scale: {self.state.pixel_size_um:.4g} um/pixel")
        current.setObjectName("Hint")
        v.addWidget(current)

        apply_btn = make_button("Set scale", "primary")
        apply_btn.clicked.connect(lambda: self._apply_scale(px_len, field))
        field.returnPressed.connect(lambda: self._apply_scale(px_len, field))
        v.addWidget(apply_btn)
        return wrap

    def _apply_scale(self, px_len: float, field: QLineEdit) -> None:
        text = field.text().strip().replace(",", ".")
        try:
            microns = float(text)
        except ValueError:
            self.shell.toast("Enter the length in microns")
            return
        if microns <= 0 or px_len <= 0:
            self.shell.toast("Enter a length greater than zero")
            return
        px_size = microns / px_len
        self.state.set_pixel_size_um(px_size)
        self.shell.dismiss_sheet()
        self._end_ruler()
        self.shell.toast(f"Scale set: 1 pixel = {px_size:.4g} um")

    # --- display sheet ----------------------------------------------------
    def _open_display_sheet(self) -> None:
        if not self.state.current_well_id:
            self.shell.toast("Open a well first")
            return
        content = self._build_display_content()
        self.shell.present_sheet("Display", content)

    def _build_display_content(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(20, 4, 20, 22)
        v.setSpacing(18)

        names = self.state.loader.channel_names if self.state.loader else []
        colors = self.state.loader.channel_colors if self.state.loader else []
        for c, name in enumerate(names):
            block = QVBoxLayout()
            block.setSpacing(8)
            head = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            col = colors[c] if c < len(colors) else (200, 200, 200)
            swatch.setStyleSheet(
                f"background-color: rgb{tuple(col)}; border-radius: 8px;"
            )
            title = QLabel(name)
            title.setObjectName("CardTitle")
            head.addWidget(swatch)
            head.addWidget(title)
            head.addStretch(1)
            vis = ToggleSwitch()
            vis.setChecked(self.state.channel_visible[c] if c < len(self.state.channel_visible) else True)
            vis.toggled.connect(lambda on, ch=c: self.state.set_channel_visible(ch, on))
            head.addWidget(vis)
            block.addLayout(head)

            bright = LabeledSlider("Brightness", "", "", self.state.brightness[c])
            bright.valueChanged.connect(lambda val, ch=c: self.state.set_brightness(ch, val))
            block.addWidget(bright)
            contrast = LabeledSlider("Contrast", "", "", self.state.contrast[c])
            contrast.valueChanged.connect(lambda val, ch=c: self.state.set_contrast(ch, val))
            block.addWidget(contrast)
            v.addLayout(block)

        return wrap

    # --- visibility hook --------------------------------------------------
    def on_shown(self) -> None:
        self._refresh_canvas()
