"""Automated downsampling - mirror an acquisition folder at reduced resolution.

Run this ONCE where the full-resolution data lives. It walks the input folder,
block-mean downsamples every image by an integer factor (antialiased, so mean
intensity is preserved), and writes a same-structure copy elsewhere. Non-image
files (metadata) are copied through, and the effective pixel size is scaled and
recorded so downstream analysis stays in real microns.

Why: the latest Cellpose wants cells < ~100 px across, so full 3000x3000 frames
are usually more resolution than it needs. Downsampling 2-3x makes Cellpose
faster AND shrinks the dataset ~4-9x, so only a small copy has to move to a GPU
box. You read the full data once locally; only the reduced copy travels.

    cellscope-downsample "/data/2026.06.26-gm-ppk2..." "/data/gm-ppk2_ds2" --factor 2 -j 16
    cellscope-downsample "/data/exp" "/data/exp_small" --max-dim 1000 --resume
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# --- downsampling ------------------------------------------------------------
def _block_mean2(a: np.ndarray, n: int) -> np.ndarray:
    """Average non-overlapping n x n blocks of a 2-D array (crops the remainder)."""
    h, w = a.shape
    h2, w2 = (h // n) * n, (w // n) * n
    if h2 == 0 or w2 == 0:
        return a
    b = a[:h2, :w2].reshape(h2 // n, n, w2 // n, n).mean(axis=(1, 3))
    if np.issubdtype(a.dtype, np.integer):
        return np.rint(b).astype(a.dtype)
    return b.astype(a.dtype)


def downsample_array(a: np.ndarray, n: int) -> np.ndarray:
    """Downsample the spatial axes of an image by factor ``n`` (block mean)."""
    n = int(n)
    if n <= 1:
        return a
    if a.ndim == 2:
        return _block_mean2(a, n)
    if a.ndim == 3 and a.shape[-1] in (2, 3, 4):          # (Y, X, C) e.g. RGB(A)
        return np.stack([_block_mean2(a[..., c], n) for c in range(a.shape[-1])], axis=-1)
    if a.ndim == 3:                                        # (pages, Y, X) stack
        return np.stack([_block_mean2(a[p], n) for p in range(a.shape[0])], axis=0)
    return a  # unknown layout: leave untouched


def _read(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    with Image.open(str(path)) as im:
        return np.asarray(im)


def _write(path: Path, arr: np.ndarray) -> None:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(str(path), arr, compression="zlib")  # compress -> smaller
    else:
        from PIL import Image
        Image.fromarray(arr).save(str(path))


# --- worker ------------------------------------------------------------------
_CFG: dict = {}


def _init(factor: int, resume: bool) -> None:
    _CFG["factor"] = factor
    _CFG["resume"] = resume


def _process(pair):
    src, dst = Path(pair[0]), Path(pair[1])
    if _CFG["resume"] and dst.exists() and dst.stat().st_size > 0:
        return (str(src), dst.stat().st_size, "skip")
    try:
        out = downsample_array(_read(src), _CFG["factor"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        _write(dst, out)
        return (str(src), dst.stat().st_size, "ok")
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
        return (str(src), 0, f"error: {exc}")


# --- metadata ----------------------------------------------------------------
def _copy_metadata(in_dir: Path, out_dir: Path, factor: int) -> None:
    """Copy non-image files through, scaling the Cephla pixel size."""
    for src in in_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() in IMAGE_EXTS:
            continue
        dst = out_dir / src.relative_to(in_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.name == "acquisition parameters.json":
            _write_scaled_params(src, dst, factor)
        else:
            shutil.copy2(src, dst)


def _write_scaled_params(src: Path, dst: Path, factor: int) -> None:
    from cellscope.data.cephla_loader import _pixel_size_from_params
    try:
        params = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(src, dst)
        return
    # Use the same tube-lens-corrected pixel size the loader would, so the
    # reduced copy records the true effective micron scale (not sensor/mag).
    obj = params.get("objective") if isinstance(params.get("objective"), dict) else {}
    has_info = bool(params.get("pixel_size_um")) or bool(
        params.get("sensor_pixel_size_um")
        and (obj.get("magnification") or params.get("magnification"))
    )
    if has_info:
        params["pixel_size_um"] = _pixel_size_from_params(params) * factor
    params["cellscope_downsample_factor"] = factor
    dst.write_text(json.dumps(params, indent=2), encoding="utf-8")


def _factor_from_maxdim(files, max_dim: int) -> int:
    for f in files:
        try:
            a = _read(Path(f[0]))
        except Exception:
            continue
        hw = a.shape[:2] if a.ndim == 2 else (a.shape[-2], a.shape[-1]) \
            if a.ndim == 3 and a.shape[-1] not in (2, 3, 4) else a.shape[:2]
        big = max(hw)
        return max(1, -(-big // max_dim))  # ceil
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cellscope-downsample",
        description="Mirror an acquisition folder at reduced resolution.",
    )
    ap.add_argument("input", help="Full-resolution acquisition folder")
    ap.add_argument("output", help="Where to write the reduced-resolution copy")
    ap.add_argument("--factor", type=int, default=2, help="Integer downsample factor")
    ap.add_argument("--max-dim", type=int, default=0,
                    help="Instead of --factor, pick a factor so the largest side <= this")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="Parallel workers (0=auto)")
    ap.add_argument("--resume", action="store_true", help="Skip files already written")
    args = ap.parse_args(argv)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"Not a folder: {in_dir}", file=sys.stderr)
        return 1

    images = sorted(p for p in in_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1
    pairs = [(str(p), str(out_dir / p.relative_to(in_dir))) for p in images]

    factor = args.factor
    if args.max_dim > 0:
        factor = _factor_from_maxdim(pairs, args.max_dim)
    factor = max(1, factor)

    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_metadata(in_dir, out_dir, factor)

    jobs = args.jobs if args.jobs > 0 else min(mp.cpu_count() or 1, 8)
    in_bytes = sum(Path(s).stat().st_size for s, _ in pairs)
    print(f"Downsampling {len(pairs)} images by 1/{factor} with {jobs} worker(s)\n"
          f"  {in_dir}  ->  {out_dir}", flush=True)

    t0 = time.time()
    out_bytes = 0
    done = failed = skipped = 0
    initargs = (factor, args.resume)
    if jobs == 1:
        _init(*initargs)
        results = (_process(p) for p in pairs)
        results = list(results)
    else:
        with mp.Pool(jobs, initializer=_init, initargs=initargs) as pool:
            results = list(pool.imap_unordered(_process, pairs, chunksize=8))

    for i, (src, size, status) in enumerate(results, 1):
        if status == "ok":
            done += 1
            out_bytes += size
        elif status == "skip":
            skipped += 1
            out_bytes += size
        else:
            failed += 1
            print(f"  FAILED {src}: {status}", file=sys.stderr)
        if i % 200 == 0 or i == len(results):
            print(f"  [{i}/{len(results)}] {time.time()-t0:.0f}s", flush=True)

    ratio = (in_bytes / out_bytes) if out_bytes else 0.0
    print(f"\nDone: {done} written, {skipped} skipped, {failed} failed in "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"Size: {in_bytes/1e9:.1f} GB -> {out_bytes/1e9:.1f} GB "
          f"({ratio:.1f}x smaller). Effective pixel size scaled by {factor}x.",
          flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
