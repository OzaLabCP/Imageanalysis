"""Headless, parallel batch analysis - run CellScope's pipeline with no GUI.

Point it at an acquisition folder; it analyzes every position (or a subset) in
parallel across CPU cores and writes one per-cell CSV per position (plus an
optional combined CSV). Designed for a compute box like Thunder: no display, no
Qt, resumable, and it scales across cores.

    cellscope-batch "/data/2026.06.26-gm-ppk2..." -o results -j 16 --combine
    python -m cellscope.batch "/data/exp" -o out --downsample 2 --resume

Analyzing 96 positions is embarrassingly parallel, so wall time is roughly
(single-position time * positions / workers). The segmentation is CPU-bound
today; a GPU segmenter (Cellpose/StarDist) is the next step to use Thunder's
GPU - it would drop into analysis/segmentation.py behind the same interface.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from cellscope.analysis import (
    AnalysisSettings,
    available_engines,
    cellpose_available,
    resolve_device,
    run_analysis,
)
from cellscope.data import CephlaLoader, FolderLoader, MockLoader
from cellscope.export import (
    combine_parquet,
    measurements_header,
    split_region_fov,
    write_measurements_csv,
    write_measurements_parquet,
)
from cellscope.provenance import (
    RUN_METADATA_NAME,
    build_run_metadata,
    write_run_metadata,
)

MOCK_SENTINEL = "__mock__"


def build_loader(folder: str):
    """Construct the right loader for a folder (Cephla-aware), or the demo plate."""
    if folder == MOCK_SENTINEL:
        return MockLoader()
    path = Path(folder)
    if CephlaLoader.looks_like(path):
        return CephlaLoader(str(path))
    return FolderLoader(str(path))


def _safe_name(well_id: str) -> str:
    return well_id.replace("/", "_").replace("\\", "_").replace(":", "_")


# --- worker (one loader per process, reused across that process's positions) ---
_WORKER: dict = {}


def _init_worker(folder, settings_kw, pixel_size, downsample, out_dir, resume,
                 fmt, dataset, profile=False, condition_map=None) -> None:
    _WORKER["loader"] = build_loader(folder)
    _WORKER["settings"] = AnalysisSettings(**settings_kw)
    _WORKER["pixel_size"] = pixel_size
    _WORKER["downsample"] = downsample
    _WORKER["out_dir"] = Path(out_dir)
    _WORKER["resume"] = resume
    _WORKER["format"] = fmt
    _WORKER["dataset"] = dataset
    _WORKER["profile"] = profile
    _WORKER["condition_map"] = condition_map or {}


def _analyze_one(well_id: str, array=None, load_secs: float = 0.0):
    ext = "parquet" if _WORKER["format"] == "parquet" else "csv"
    out = _WORKER["out_dir"] / f"{_safe_name(well_id)}.{ext}"
    if _WORKER["resume"] and out.exists() and out.stat().st_size > 0:
        return (well_id, -1, "skipped")
    try:
        stamps: dict = {}
        cb = None
        if _WORKER.get("profile"):
            def cb(pct, _stamps=stamps):
                now = time.time()
                _stamps.setdefault("start", now)
                if pct >= 60 and "seg" not in _stamps:
                    _stamps["seg"] = now
                if pct >= 100:
                    _stamps["end"] = now
        wa = run_analysis(
            _WORKER["loader"], well_id, _WORKER["settings"], progress_cb=cb,
            pixel_size_um=_WORKER["pixel_size"], downsample=_WORKER["downsample"],
            array=array,
        )
        region, _fov = split_region_fov(well_id)
        condition = _WORKER["condition_map"].get(region, "")
        if _WORKER["format"] == "parquet":
            write_measurements_parquet(str(out), [(well_id, condition, wa)],
                                       dataset=_WORKER["dataset"])
        else:
            write_measurements_csv(str(out), [(well_id, condition, wa)])
        if _WORKER.get("profile") and "end" in stamps:
            seg = stamps.get("seg", stamps["end"]) - stamps["start"]
            quant = stamps["end"] - stamps.get("seg", stamps["end"])
            print(f"    profile {well_id}: load {load_secs:.1f}s, segment {seg:.1f}s, "
                  f"quantify {quant:.1f}s", flush=True)
        return (well_id, wa.n_tracks, "ok")
    except Exception as exc:  # noqa: BLE001 - one bad position must not kill the run
        return (well_id, 0, f"error: {exc}")


def _select(wells, patterns: str | None):
    if not patterns:
        return wells
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    return [w for w in wells if any(fnmatch.fnmatch(w, p) for p in pats)]


def _combine(out_dir: Path, wells, channel_names) -> Path:
    combined = out_dir / "all_measurements.csv"
    header = ",".join(measurements_header(channel_names))
    with open(combined, "w", newline="", encoding="utf-8") as dst:
        dst.write(header + "\n")
        for well_id in wells:
            part = out_dir / f"{_safe_name(well_id)}.csv"
            if not part.exists():
                continue
            with open(part, encoding="utf-8") as src:
                next(src, None)  # skip the per-file header
                for line in src:
                    dst.write(line)
    return combined


def _load_platemap(path) -> dict:
    """Read a well->condition CSV (columns: well/region + condition/treatment)."""
    if not path:
        return {}
    import csv as _csv
    mapping: dict = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            lower = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            wcol = lower.get("well") or lower.get("region")
            ccol = lower.get("condition") or lower.get("treatment") or lower.get("group")
            if wcol and ccol:
                for row in reader:
                    mapping[str(row[wcol]).strip()] = str(row[ccol]).strip()
    except OSError:
        pass
    return mapping


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cellscope-batch",
        description="Headless parallel cell analysis -> per-cell CSVs.",
    )
    ap.add_argument("folder", help="Acquisition folder (or '__mock__' for the demo plate)")
    ap.add_argument("-o", "--out", default="cellscope_out", help="Output directory")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="Parallel workers (0 = auto). Each worker holds ~1 position in RAM.")
    ap.add_argument("--positions", default=None,
                    help="Comma-separated ids or glob patterns, e.g. 'B2-*,B3-0'")
    ap.add_argument("--downsample", type=int, default=4,
                    help="Analyze at 1/N resolution for speed; measurements stay in real "
                         "microns (the pixel size is scaled). DEFAULT 4, validated for this "
                         "lab's acquisitions (cells stay well-resolved at ~1.3 um/px). "
                         "Pass --downsample 1 for full resolution / the finest morphometrics.")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="Microns per full-res pixel (default: from the acquisition metadata)")
    ap.add_argument("--engine", default="threshold", choices=["threshold", "cellpose"],
                    help="Segmentation engine (cellpose needs a GPU install)")
    ap.add_argument("--cellpose-model", default="",
                    help="Cellpose model name ('' = default cpsam)")
    ap.add_argument("--cellpose-diameter", type=float, default=0.0,
                    help="Expected cell diameter in px (0 = auto)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Run Cellpose on CPU (slow) instead of GPU")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv",
                    help="Output: tidy CSV (default) or fixed-schema parquet (needs pyarrow)")
    ap.add_argument("--dataset", default="",
                    help="Value for the parquet 'Dataset' column (default: the source folder)")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="Single-worker only: load this many positions ahead on a "
                         "background thread so disk I/O overlaps compute (0 = off)")
    ap.add_argument("--profile", action="store_true",
                    help="Print per-position load / segment / quantify seconds")
    ap.add_argument("--analyze", action="store_true",
                    help="After the run, auto-write a subpopulation analysis report "
                         "(parquet only; needs pandas+matplotlib). Implies --combine.")
    ap.add_argument("--platemap", default=None,
                    help="CSV mapping Well->condition, used by --analyze")
    ap.add_argument("--xlsx", action="store_true",
                    help="with --analyze, also write an interactive Excel workbook "
                         "(report.xlsx: live formulas + native charts; needs openpyxl)")
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=1.5)
    ap.add_argument("--min-size", type=int, default=25)
    ap.add_argument("--seg-channel", type=int, default=0)
    ap.add_argument("--max-distance", type=float, default=30.0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip positions whose output file already exists")
    ap.add_argument("--combine", action="store_true",
                    help="Also write one combined all_measurements file")
    ap.add_argument("--list", action="store_true", help="List positions and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.analyze:
        args.combine = True  # the report reads the combined parquet

    loader = build_loader(args.folder)
    wells = _select([w.well_id for w in loader.list_wells()], args.positions)
    channel_names = loader.channel_names

    if args.list:
        for w in wells:
            print(w)
        print(f"# {len(wells)} positions; channels={channel_names}; "
              f"pixel={loader.pixel_size_um:.4g} um/px")
        return 0

    if not wells:
        print("No matching positions.", file=sys.stderr)
        return 1

    if args.engine == "cellpose" and not cellpose_available():
        print("Cellpose engine requested but 'cellpose' is not installed. On the "
              "GPU machine: pip install 'cellscope[cellpose]' plus a CUDA build of "
              "torch. Available engines: " + ", ".join(available_engines()),
              file=sys.stderr)
        return 1

    if args.format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("Parquet output needs pyarrow. Install it with: pip install pyarrow "
                  "(or: pip install 'cellscope[parquet]').", file=sys.stderr)
            return 1

    # Resolve the real compute device up front so the header reflects reality:
    # Cellpose silently falls back to CPU when the GPU is not visible to torch.
    gpu_info = resolve_device(not args.no_gpu) if args.engine == "cellpose" else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cellpose holds a GPU model per process, so it MUST run one worker (the GPU
    # parallelizes internally by batching frames); N workers would load N models
    # onto one card and exhaust GPU memory. The threshold engine scales on CPU.
    if args.engine == "cellpose":
        if args.jobs > 1:
            print(f"Note: --engine cellpose runs 1 worker (requested {args.jobs}); "
                  "multiple workers load a model per process and exhaust GPU memory.",
                  file=sys.stderr)
        jobs = 1
    elif args.jobs > 0:
        jobs = args.jobs
    else:
        jobs = min(mp.cpu_count() or 1, 8)

    effective_pixel = (args.pixel_size if args.pixel_size is not None
                       else loader.pixel_size_um)
    dataset = args.dataset or (str(args.folder) if args.folder != MOCK_SENTINEL else "demo")
    settings_kw = dict(
        sensitivity=args.sensitivity, smoothing=args.smoothing,
        min_size=args.min_size, seg_channel=args.seg_channel,
        max_distance=args.max_distance,
        engine=args.engine, cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter, cellpose_gpu=not args.no_gpu,
    )
    condition_map = _load_platemap(args.platemap)
    initargs = (args.folder, settings_kw, args.pixel_size, args.downsample,
                str(out_dir), args.resume, args.format, dataset, args.profile,
                condition_map)

    device_note = f", {gpu_info['detail']}" if gpu_info else ""
    print(f"CellScope batch: {len(wells)} positions, engine={args.engine}{device_note}, "
          f"{jobs} worker(s), downsample 1/{args.downsample}, format={args.format}, "
          f"out={out_dir}", flush=True)
    if gpu_info and gpu_info["fell_back"]:
        print("WARNING: GPU requested but not available - Cellpose will run on CPU "
              "(much slower). Fix the CUDA/torch install, or pass --no-gpu to silence.",
              file=sys.stderr, flush=True)
    t0 = time.time()
    results = []

    def report(i, r):
        tag = f"{r[1]} cells" if r[1] >= 0 else "skipped"
        print(f"[{i}/{len(wells)}] {r[0]}: {r[2]} ({tag}) "
              f"[{time.time() - t0:.0f}s]", flush=True)

    if jobs == 1:
        _init_worker(*initargs)
        prefetch = max(0, int(args.prefetch))
        if prefetch > 0 and len(wells) > 1:
            # Load the next position(s) on a background thread so disk I/O
            # overlaps the GPU compute of the current one (the GPU is the serial
            # bottleneck; reads shouldn't idle it). The queue bounds how far
            # ahead - and thus the memory - the prefetcher runs.
            ext = "parquet" if args.format == "parquet" else "csv"
            ds = max(1, int(args.downsample))
            ready: "queue.Queue" = queue.Queue(maxsize=prefetch)

            def _producer():
                for wid in wells:
                    arr, load_secs = None, 0.0
                    out = out_dir / f"{_safe_name(wid)}.{ext}"
                    skip = args.resume and out.exists() and out.stat().st_size > 0
                    if not skip:
                        try:
                            t_load = time.time()
                            arr = _WORKER["loader"].get_well(wid, downsample=ds)
                            load_secs = time.time() - t_load
                        except Exception:
                            arr = None  # the error resurfaces in _analyze_one
                    ready.put((wid, arr, load_secs))

            threading.Thread(target=_producer, daemon=True).start()
            for i in range(1, len(wells) + 1):
                wid, arr, load_secs = ready.get()
                r = _analyze_one(wid, array=arr, load_secs=load_secs)
                results.append(r)
                report(i, r)
        else:
            for i, wid in enumerate(wells, 1):
                r = _analyze_one(wid)
                results.append(r)
                report(i, r)
    else:
        with mp.Pool(jobs, initializer=_init_worker, initargs=initargs) as pool:
            for i, r in enumerate(pool.imap_unordered(_analyze_one, wells), 1):
                results.append(r)
                report(i, r)

    ok = sum(1 for r in results if r[2] == "ok")
    skipped = sum(1 for r in results if r[2] == "skipped")
    failed = [r for r in results if r[2].startswith("error")]
    total_cells = sum(r[1] for r in results if r[1] > 0)
    # A position that analyzed cleanly but found zero cells is either a real
    # acquisition problem or a segmentation failure - either way it must be
    # reported, not left as an empty file for a reader to notice.
    empty = [r[0] for r in results if r[2] == "ok" and r[1] == 0]
    print(f"\nDone: {ok} analyzed, {skipped} skipped, {len(failed)} failed, "
          f"{len(empty)} empty (0 cells), {total_cells} cells total, "
          f"in {time.time() - t0:.0f}s", flush=True)
    for r in failed:
        print(f"  FAILED {r[0]}: {r[2]}", file=sys.stderr)
    if empty:
        shown = ", ".join(empty[:20]) + (" ..." if len(empty) > 20 else "")
        print(f"  EMPTY (0 cells in every frame): {shown}", file=sys.stderr, flush=True)

    # Provenance sidecar: exactly how these results were produced. Comparing two
    # of these files proves whether two runs were analyzed identically.
    meta = build_run_metadata(
        source=args.folder, engine=args.engine, settings=settings_kw,
        pixel_size_um=effective_pixel, downsample=args.downsample,
        channel_names=channel_names, gpu=gpu_info, positions=wells,
        output_format=args.format,
        created_utc=datetime.now(timezone.utc).isoformat(),
        extra={"dataset": dataset, "empty_positions": empty},
    )
    write_run_metadata(str(out_dir / RUN_METADATA_NAME), meta)

    combined_path = None
    if args.combine:
        if args.format == "parquet":
            parts = [str(out_dir / f"{_safe_name(w)}.parquet") for w in wells]
            combined_path = out_dir / "all_measurements.parquet"
            rows = combine_parquet(str(combined_path), parts)
            print(f"Combined {rows} rows -> {combined_path}", flush=True)
        else:
            combined = _combine(out_dir, wells, channel_names)
            print(f"Combined -> {combined}", flush=True)

    # QC pass: the pipeline's worst failure is emitting plausible-looking wrong
    # data silently, so a run flags its own corruption modes (non-unique per-cell
    # keys, missing/blank channels, saturation, missing timepoints) before the
    # numbers reach analysis. Never fails the run - it only reports.
    if combined_path is not None and args.format == "parquet":
        try:
            from cellscope.qc import format_issues, qc_report
            rep = qc_report(str(combined_path), str(out_dir / "qc.json"))
            print(format_issues(rep), flush=True)
        except Exception as exc:  # noqa: BLE001 - QC must never fail the run
            print(f"QC check skipped: {exc}", file=sys.stderr)

    if args.analyze and args.format == "parquet" and combined_path is not None:
        try:
            from cellscope.analyze import run as analyze_run
            report_dir = out_dir / "report"
            info = analyze_run(str(combined_path), str(report_dir),
                               platemap=args.platemap, xlsx=args.xlsx)
            verdict = ("subpopulation detected" if info.get("subpopulation_detected")
                       else "no subpopulation detected")
            print(f"Analysis report -> {report_dir / 'index.html'}  "
                  f"({info['cells_gated']:,} cells, {len(info['groups'])} groups, "
                  f"{len(info['timepoints'])} timepoints, {verdict})", flush=True)
            if info.get("xlsx"):
                print(f"Excel workbook -> {info['xlsx']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a report failure must not fail the run
            print(f"Analysis report skipped: {exc}", file=sys.stderr)
    elif args.analyze and args.format != "parquet":
        print("--analyze needs --format parquet; skipping report.", file=sys.stderr)

    return 0 if not failed else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
