# CellScope

**Follow and measure individual cells over time, across many wells of a plate, without writing code or tuning parameters.**

CellScope is a desktop app that reimagines a scientific image viewer as a clean,
phone-style tool. Load microscopy images, detect cells, track each one across a
time course, and quantify them across multiple wells, all through big buttons and
sliders instead of code.

It runs **100% on your computer**. No internet, no accounts, no uploads. Your
images never leave your disk.

> The app opens **your own TIFF and JPG images** (see *Opening your own images*
> below), and also ships with a built-in **demo plate** of synthetic cells so you
> can try everything immediately. Heavier algorithms (Cellpose, btrack) and more
> formats (ND2, CZI) plug in later without changing how the app looks or feels.

---

## Pipeline at a glance

Point CellScope at a folder of images and **one command** carries it all the way
to figures — segment, track, measure, quality-check, and plot. Measurements stay
in real microns at every step, and the run warns loudly whenever the input is off
rather than emitting plausible-looking wrong numbers.

```bash
cellscope-batch "path/to/acquisition" -o results --engine cellpose --format parquet --analyze
```

1. **Intake** — auto-detects the folder layout (Cephla/Squid, OME-TIFF, or a plain
   folder of frames) and the list of positions to analyze. No config.
2. **Metadata** — reads channels, timepoints, and the **tube-lens-corrected pixel
   size** from the acquisition, so everything downstream is in real microns.
3. **Load** — pulls each position's `(Time, Z, Channel, Y, X)` stack; optional
   **downsampling (default 4×)** for speed, with the pixel size scaled to match.
4. **Segment** — turns each frame into labeled cells (Otsu threshold by default, or
   the **Cellpose** GPU model with `--engine cellpose`).
5. **Track** — links cells across consecutive frames into stable per-cell IDs; a
   `segment` column marks gaps so a track is never joined across a missing frame.
6. **Quantify** — per cell, per frame: area, equivalent/Feret diameter, axis
   lengths, perimeter, eccentricity, centroid, and per-channel intensity
   (mean/total/max/min/std). A blank channel is recorded as **NaN**, never `0.0`.
7. **Export + combine** — one row per cell per timepoint, keyed uniquely by
   `(Dataset, Well, fov, Timepoint, Label)`, merged into `all_measurements.parquet`.
8. **QC** — scans the table for silent-corruption modes (key collisions, blank
   channels, saturation, timepoint coverage) and writes `qc.json` + a summary.
9. **Provenance** — writes `run_metadata.json`: git commit, engine, GPU, pixel
   size, downsample factor, and positions, so any run is reproducible and diffable.
10. **Analyze** (`--analyze`) — gates cells, sets a data-driven responder threshold,
    groups by condition (from the parquet or `--platemap`), and renders the report:
    **fog plots** (per-cell intensity over time), a **well-level superplot with
    proper statistics** (tested across wells, not cells — no pseudo-replication),
    responder-fraction, distributions, percentile bands, per-cell ramp-rate
    trajectories, CSV + JSON summaries, and an `index.html`.

**What lands in `results/`:**

```
results/
├─ H2-0.parquet, H2-1.parquet, …   # per-position measurements
├─ all_measurements.parquet        # master table: one row per cell per timepoint
├─ qc.json                         # data-integrity report
├─ run_metadata.json               # exact provenance (code, GPU, pixel size, downsample)
└─ report/
   ├─ fog_over_time.png             # the fog plots
   ├─ responder_fraction.png, distributions_over_time.png, percentile_bands.png, …
   ├─ group_timepoint_summary.csv, responder_characteristics.csv
   └─ index.html                    # open this
```

The headless batch above needs `pip install "cellscope[cellpose,analysis]"` (see
[Running headless](#running-headless-on-a-compute-server-batch) and
[Cellpose](#better-masks-with-cellpose-local-gpu)). Prefer clicking to typing? The
same detect → track → quantify engine runs behind the **desktop app** below.

---

## Install and run (for non-developers)

You need **Python 3.10 or newer** installed. Then open a terminal (Command Prompt
or PowerShell on Windows, Terminal on macOS/Linux) and run these three lines:

```bash
cd path/to/cellscope          # the folder that contains pyproject.toml
pip install -e .              # installs CellScope and its dependencies
cellscope                     # opens the app window
```

That's it. The CellScope window opens with a demo plate already loaded.

If `cellscope` isn't found after install, you can always launch it with:

```bash
python -m cellscope
```

### Tips
- On Windows, if `pip` or `python` aren't recognized, reinstall Python from
  [python.org](https://www.python.org/downloads/) and check **"Add Python to PATH"**.
- The first time you open a well, it is generated and cached, so it may take a
  moment; after that it is instant.

---

## What you can do

The app has four tabs along the bottom, like a phone:

1. **Wells** — a gallery of every well on the plate. Tap one to open it. Tap the
   circle in a card's corner to add wells to a comparison. **Open** loads a
   folder of your own TIFF/JPG images; **Demo** reloads the synthetic plate.
2. **Viewer** — the image canvas.
   - Scrub **Time**, **Focus**, and **Well** with sliders; press **play** to
     animate through the time course.
   - Switch channels or **Merge** them; open **Display** to adjust brightness,
     contrast, and which channels show.
   - Tap **Detect Cells** (the big button). One **Sensitivity** slider is all you
     need; everything else hides under *Advanced options*.
   - Toggle the **Outlines / Tracks / IDs** overlays. **Tap a cell** to follow it.
   - Zoom with the scroll wheel or pinch; drag to pan; double-click to reset.
   - Flip **Fast preview** on for big datasets: positions load and detect at
     reduced resolution (much faster, less memory) while areas/diameters stay in
     real microns. Turn it off and re-run Detect Cells for a full-resolution pass.
   - Tap **Scale** to calibrate real units: drag a line across a known distance
     (e.g. the scale bar burned into the image) and type its length in microns.
     Every area and diameter then reports in microns, and existing results
     rescale instantly. The current scale shows under the Viewer title.
3. **Cells** — a list of every tracked cell. Tap one to see its lifespan, size,
   and an intensity-over-time chart, then jump back to the Viewer to watch it.
4. **Results** — a **cells-over-time chart**, a sortable **measurements table**,
   and **Export CSV** (per-cell measurements) and **Export PNG** (the chart) to a
   folder you choose.

---

## Opening your own images

Tap **Open** on the Wells tab and pick a folder of `.tif/.tiff`, `.png`, or
`.jpg/.jpeg` images (or launch straight into one: `cellscope "C:\path\to\folder"`).
CellScope figures out the structure automatically, no setup:

- **One multi-dimensional TIFF per well** (OME-TIFF or an ImageJ/Fiji
  hyperstack): the file's own Time/Z/Channel axes are used. Each such file
  becomes one well, named from the filename (or a well token in it).
- **A folder of single images named with tokens** it recognizes, grouped into
  wells and stacked along the right axes:
  - **well**: `A1`…`H12`, or `well_<name>`
  - **time**: `t3`, `time3`, `frame3`
  - **channel**: `c0`, `ch0`, `channel0`, or a fluorophore name (`DAPI`, `GFP`,
    `RFP`, `Cy5`, `BF`, …)
  - **focus (Z)**: `z2`
  - Tokens may be separated or run together: `A1_t005_c1.tif`, `A2_GFP_z0.tif`,
    or `3B4T0.png` (team 3, well B4, time 0).
- **A plain folder of frames** with no recognizable tokens: treated as **one
  well's time series**, ordered by filename.
- **A Cephla / Squid time-lapse acquisition** (numbered timepoint folders with
  `<region>_<fov>_<z>_<channel>.tiff` files and an `acquisition parameters.json`)
  is auto-detected: each **(well, field of view)** becomes one position, channels
  and colors come from the acquisition, and the **pixel size is read from the
  metadata** so measurements are already in microns without ruler calibration.
  A **single flat folder** of those `<region>_<fov>_<z>_<channel>.tiff` images
  (e.g. one downloaded timepoint, with the metadata alongside or a level up) is
  recognized too. The pixel size honors the **tube-lens correction** — an
  objective's magnification is specified for its design tube lens, so a 20x used
  on a 50 mm tube lens instead of its 180 mm design images at 20·50/180 = 5.56x.

Color (RGB) images become three channels (Red/Green/Blue) unless a channel token
says otherwise. If a folder has no images, CellScope tells you and keeps your
current data. *(Coming next: ND2 and CZI via `aicsimageio`.)*

## How the analysis works

Real, simple algorithms, each behind a clean module boundary so they can be
upgraded later:

- **Detect** (`analysis/segmentation.py`): Gaussian smoothing, an automatic
  (Otsu) threshold nudged by the Sensitivity slider, then connected-component
  labeling. *(Upgrade path: Cellpose / StarDist.)*
- **Track** (`analysis/tracking.py`): nearest-neighbour centroid linking between
  frames with a distance gate (`--max-distance`), giving each cell a track ID.
  **Tracking contract — read before `groupby("Label")`:** IDs are stable only
  across *consecutive* frames and are **per-position, not global** (each position
  restarts at 1, which is why the parquet needs `fov` to key uniquely). A cell
  present at frame N ends there; because a missing timepoint is laid down as an
  empty frame, a **gap restarts the label space** (frame N ends at ID K, frame
  N+2 resumes at K+1 for an unrelated cell). Group tracks by
  `(Well, fov, segment, Label)` — never `(Well, Label)` across a gap — where the
  exported `segment` column marks each contiguous, gap-free run of timepoints.
  *(Upgrade path: btrack with cell division and gap-closing.)*
- **Quantify** (`analysis/quantify.py`): per cell, per frame: area (pixels and
  microns squared), size as **equivalent and Feret (max-caliper) diameter in
  microns**, centroid, and **mean / total intensity per channel**. Exported as a
  tidy long-format CSV (one row per cell per timepoint) ready for pandas/seaborn.

Analysis runs on a background thread, so the window never freezes; a progress bar
shows how it's going.

---

## Running headless on a compute server (batch)

For large acquisitions (e.g. a 96-position Cephla time-lapse), analyze without the
GUI on a many-core machine. `cellscope-batch` reads a folder, analyzes every
position (or a subset) **in parallel across CPU cores**, and writes one per-cell
CSV per position plus an optional combined table:

```bash
cellscope-batch "/data/2026.06.26-gm-ppk2..." -o results -j 16 --combine
cellscope-batch "/data/exp" --list                      # preview positions
cellscope-batch "/data/exp" -o out --downsample 2 --resume
python -m cellscope.batch "/data/exp" -o out -j 8        # module form
```

Options: `-j/--jobs` (workers), `--positions "B2-*,B3-0"` (glob subset),
`--downsample N` (**default 4**; analyze at 1/N resolution — measurements stay in
real microns because the pixel size is scaled), `--pixel-size`,
`--sensitivity`/`--min-size`/`--seg-channel`/`--max-distance`,
`--resume` (skip finished positions), `--combine`, `--list`,
`--format csv|parquet`, `--dataset <name>`.

> **Downsampling defaults to 4×.** For this lab's acquisitions a 4× reduction is
> validated to leave cells well-resolved (~1.3 µm/px) while cutting analysis time
> substantially, so it's the default. The factor is printed in the run header
> (`downsample 1/4`) and recorded in `run_metadata.json`, so it's never silent.
> Pass **`--downsample 1`** for full resolution / the finest morphometrics, or
> `--downsample 2` for a middle ground.

Every run also writes a **`run_metadata.json`** provenance sidecar into the output
folder recording exactly how the results were produced (engine, settings, pixel
size, resolved GPU device, CellScope version **and git commit**, positions).
Comparing two of these proves whether two runs were analyzed identically - and the
git SHA distinguishes two builds that report the same `0.1.0` version but differ in
output schema.

### Parquet output (`--format parquet`)

For pipelines built around a fixed "regionprops-style" table, `--format parquet`
(needs `pip install "cellscope[parquet]"`) writes one row per cell per timepoint.
The first **18 columns are the fixed reference schema** (so a notebook built on it
still reads by name): `Label`, `Diameter (Equivalent) (um)`,
`Diameter (Feret) (um)`, `Length Major (um)`, `Length Minor (um)`,
`Perimeter (um)`, per-channel `Intensity Mean/Max/STD/Min (<channel>)`,
`Eccentricity`, `Dataset`, `Timepoint` (0-based), and `Well` (region).

Three columns are **appended**: **`fov`**, **`condition`**, and **`segment`**.
CellScope segments each field of view separately, so `Label` restarts at 1 per
FOV; without `fov` the per-cell key `(Well, Timepoint, Label)` silently *collides*
across the FOVs pooled into a well (two different cells map to one key). The unique
per-cell key is therefore **`(Dataset, Well, fov, Timepoint, Label)`**. `condition`
carries the platemap group through so downstream code needn't re-join. `segment`
marks each contiguous, gap-free run of timepoints in a position — group *tracks*
by **`(Well, fov, segment, Label)`** so a track ID is never joined across a
missing frame (see the tracking contract above). `--combine` writes one
`all_measurements.parquet` across positions.

> **Storage dtype / saturation.** Intensity columns are written as float64, but
> the *values* inherit the source images' dtype. Squid/Cephla exports are often
> float16, whose largest finite value is **65504** (not uint16's 65535), so a
> naive `>= 65535` saturation test never fires. The QC pass flags cells at or
> above **65504**, catching the cap for both dtypes.

### Built-in QC (`qc.json`)

Because the pipeline's worst failure mode is emitting *plausible-looking* wrong
data with no warning, every `--combine`/`--analyze` run writes a **`qc.json`** and
prints a one-line summary. It flags, at run time, the silent corruption modes:

- **non-unique per-cell keys** (e.g. a missing `fov` column collapsing distinct
  cells onto one key),
- rows with a **missing / blank channel** (NaN intensity - an all-zero plane is
  recorded as absent, *not* as `0.0`, so it can't poison a mean),
- cells **at or above sensor saturation**, and
- positions **missing timepoints** or gaps in the timepoint sequence.

The analysis report surfaces the same findings as a banner at the top of
`report/index.html`, so a data-integrity problem is visible before you read a
single figure. A clean run prints `QC: no issues found.`

### Automated analysis report (`--analyze` / `cellscope-analyze`)

Turn any run's parquet into a comprehensive **subpopulation report** in one step
(needs `pip install "cellscope[analysis]"`). It is aimed at *"is there a
subpopulation that behaves differently - and better - in some conditions?"*, so it
works at the population/distribution level (a responder subpopulation shows up as
a second, high mode - not a shift in the mean).

Run it automatically as part of a batch run, or standalone on an existing parquet:

```bash
cellscope-batch "/data/exp" -o results --engine cellpose --format parquet --analyze \
    --platemap plate.csv                       # report written to results/report/
cellscope-analyze results/all_measurements.parquet -o results/report --platemap plate.csv
```

Grouping by **condition** happens automatically when the parquet carries a
`condition` column (the batch runner threads a plate map through, so a run
acquired with one groups by condition with no extra step); `--platemap`
(`well,condition` CSV, e.g. `H2,Drug A`) overrides it, and without either the
report groups by well. The report (`report/index.html`) contains:

- **Subpopulation statistics** (`subpopulation_stats.json`, `subpopulation_superplot.png`) -
  the inferential answer. Cells are pooled into a per-well value and every test
  runs **across wells, not cells**, so significance isn't faked by counting
  thousands of correlated cells (pseudo-replication). Reports responder fraction,
  median intensity, and **per-cell ramp rate** (trajectory slope) per condition,
  with Mann-Whitney U / Kruskal-Wallis + BH-corrected pairwise tests, Cliff's delta
  effect sizes, and bootstrap CIs. A **superplot** shows each well as a big dot
  (the unit tested) over the faint cells. When only one well exists per condition
  it drops to the FOV level and says so loudly (technical, not biological,
  replicates); it also reports **bimodality** per group so a unimodal distribution
  (where the responder gate is meaningless) is visible.
- **Fog over time** - per group, one dot per cell, x = time, with the responder gate drawn.
- **Distributions over time** - per-timepoint violins; two modes reveal a subpopulation.
- **Responder fraction over time** - % of cells above a data-driven (Otsu-on-log)
  gate, per group over time.
- **Percentile bands** - median vs top decile (p90); a rising tail flags responders.
- **Responder characterization** - how the high-green cells differ in size / red / shape.
- **CSV summaries** (`group_timepoint_summary.csv`, `responder_characteristics.csv`)
  ready for Prism/R/Excel.

Per-cell **trajectories** (the ramp-rate metric) use the globally-unique cell id
`(Dataset, Well, fov, segment, Label)` the schema now provides, built only within
a gap-free `segment` so a track is never fused across a missing frame.

The output CSV includes a **`region`** column (well without the field-of-view
suffix) so you can pool per well directly in pandas/seaborn. Analyzing positions
is embarrassingly parallel, so wall time is roughly *(per-position time x
positions / workers)*. Segmentation is CPU-bound today; a GPU segmenter
(Cellpose/StarDist) is the next step and would drop in behind
`analysis/segmentation.py`.

> **Tip for a compute box:** stage the raw data onto the machine's local/scratch
> disk once, then run from there. Don't analyze straight off a cloud-synced
> folder (OneDrive on-demand reads are slow and can fail mid-run).

---

## Better masks with Cellpose (local GPU)

Segmentation is **pluggable**. The default `threshold` engine (Gaussian + Otsu)
needs no GPU and runs anywhere, but for dense or varied objects (e.g. polydisperse
droplets) the deep-learning **Cellpose** engine gives far better masks. Run it on
a machine with an **NVIDIA GPU**, with the data on that machine's local disk — so
nothing has to be transferred to a remote GPU.

**Install (on the GPU machine, once).** Order matters: install CellScope +
Cellpose first, then (re)install the CUDA build of PyTorch **last**, so Cellpose
can't pull a CPU-only torch over the top of it.
```bash
# 1) Python 3.10-3.12 (3.13/3.14 may not have wheels yet); a fresh venv is cleanest.
pip install "cellscope[cellpose,analysis]" packaging

# 2) A CUDA build of PyTorch MATCHING YOUR GPU, installed last. Pick the index-url:
#      RTX 50-series (Blackwell, e.g. 5080/5090)  -> cu128   (cu124 will NOT work)
#      RTX 30/40-series and most others           -> cu124
#    See https://pytorch.org for the exact current line.
pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps

# 3) Verify torch actually sees the GPU (must print: True  12.8):
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
If step 3 prints `False`, the CUDA/torch build doesn't match your driver — fix
that before running, or CellScope will fall back to CPU (and say so). The batch
header prints the resolved device so you can confirm it picked the GPU.

**Run it — headless batch:**
```bash
cellscope-batch "/data/exp" -o results --engine cellpose --combine
cellscope-batch "/data/exp" --engine cellpose --cellpose-diameter 40   # if you know the cell size
```
With `--engine cellpose` the runner uses **one worker** (the GPU parallelizes
internally by batching a position's frames; multiple workers would load a model
per process and exhaust GPU memory); the `threshold` engine scales across CPU
cores instead. Measurements come out identically in real microns either way.

The batch header now prints the **resolved compute device** (e.g. `CUDA GPU:
NVIDIA RTX ...`), and warns loudly if a GPU was requested but isn't visible to
PyTorch — Cellpose otherwise falls back to CPU silently, which is much slower.

**In the app:** the Detect sheet's *Advanced options* shows a **Segmentation engine**
choice (Threshold / Cellpose) whenever Cellpose is installed.

*The current Cellpose default is the size-agnostic Cellpose-SAM model (v4). No GPU
on a plain laptop is needed for the threshold engine or to browse results — only for
running Cellpose itself.*

---

## Project layout

```
cellscope/
├─ pyproject.toml            # metadata + the `cellscope` command
├─ requirements.txt
├─ README.md
├─ tests/                    # headless checks (no display needed)
└─ cellscope/
   ├─ __main__.py            # enables `python -m cellscope`
   ├─ app.py                 # window, bottom tab bar, overlays
   ├─ state.py               # shared state + signals tying the tabs together
   ├─ theme.py               # light/dark palette + QSS (retheme here)
   ├─ render.py              # numpy -> Qt image compositing
   ├─ colors.py              # channel + track colors
   ├─ config.py              # tiny local JSON settings
   ├─ data/
   │  ├─ loader.py           # DatasetLoader interface (real readers plug in here)
   │  └─ mock.py             # synthetic plate generator
   ├─ analysis/
   │  ├─ segmentation.py
   │  ├─ tracking.py
   │  ├─ quantify.py
   │  └─ pipeline.py         # detect -> track -> quantify
   ├─ views/                 # the four tabs
   │  ├─ wells_view.py
   │  ├─ viewer_view.py
   │  ├─ cells_view.py
   │  └─ results_view.py
   └─ widgets/               # reusable phone-style controls + the canvas
```

The **loader** and **analysis** boundaries are deliberately clean: dropping in a
real OME-TIFF reader or a Cellpose engine does not touch the views.

---

## Privacy

CellScope makes **no network calls**. There is no server, no cloud, no database,
and no telemetry. The only thing it writes outside your image folders is a small
preferences file at `~/.cellscope/config.json` and whatever CSV/PNG files you
choose to export.

---

## Development

```bash
pip install -e .
python tests/test_analysis.py     # analysis checks (no display)
python tests/test_smoke.py        # full app smoke test (offscreen)
```
