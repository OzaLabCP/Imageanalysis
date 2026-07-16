#!/usr/bin/env python3
"""Compare CellScope per-cell parquet outputs across processing variants.

Reads two or more CellScope `all_measurements.parquet` files (e.g. full-res vs
2x vs 4x downsampled) and reports, relative to the FIRST (reference) file:
  * cell counts, total and per Well;
  * for every numeric measurement: reference mean/median, each variant's mean,
    its percent change vs reference, and a Kolmogorov-Smirnov distribution-shift
    statistic (0 = identical distribution, 1 = fully separated);
  * a boundary verdict flagging where a variant drifts past a tolerance.

Usage (Windows-friendly - positional paths + labels):
    python compare_downsample.py ^
        "C:\\...\\7_ds1\\all_measurements.parquet" ^
        "C:\\...\\7_ds2\\all_measurements.parquet" ^
        "C:\\...\\7_ds4\\all_measurements.parquet" ^
        --labels 1x 2x 4x --plots "C:\\...\\7_compare"

Requires pandas + scipy (already in the CellScope venv). --plots is optional
(needs matplotlib) and writes one overlaid-histogram PNG per metric.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ID_COLS = {"Label", "Dataset", "Timepoint", "Well"}


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare CellScope parquet outputs.")
    ap.add_argument("inputs", nargs="+", help="parquet files; the first is the reference")
    ap.add_argument("--labels", nargs="*", default=None, help="short label per input")
    ap.add_argument("--tol", type=float, default=5.0,
                    help="percent-change / count tolerance to flag as DRIFT (default 5)")
    ap.add_argument("--plots", default=None, help="directory for overlaid-histogram PNGs")
    args = ap.parse_args(argv)

    labels, dfs = [], []
    for i, path in enumerate(args.inputs):
        lab = args.labels[i] if args.labels and i < len(args.labels) else f"set{i+1}"
        labels.append(lab)
        dfs.append(pd.read_parquet(path))
    ref_label, ref = labels[0], dfs[0]
    cols = _metric_cols(ref)

    print("=" * 78)
    print("CELL COUNTS (total, then per Well)")
    print("=" * 78)
    counts = pd.DataFrame(index=["TOTAL"])
    for lab, df in zip(labels, dfs):
        counts.loc["TOTAL", lab] = len(df)
        if "Well" in df.columns:
            for w, n in df.groupby("Well").size().items():
                counts.loc[str(w), lab] = n
    print(counts.fillna(0).astype(int).to_string())
    print()
    for lab, df in zip(labels[1:], dfs[1:]):
        d = (len(df) - len(ref)) / max(1, len(ref)) * 100
        print(f"  {lab}: {len(df):,} cells ({d:+.1f}% vs {ref_label})")

    from scipy.stats import ks_2samp
    print("\n" + "=" * 78)
    print(f"METRIC DISTRIBUTION SHIFT vs {ref_label}")
    print("=" * 78)
    rows = []
    for c in cols:
        r = ref[c].dropna().to_numpy()
        rmean = float(np.mean(r)) if r.size else np.nan
        row = {"metric": c, f"{ref_label} mean": rmean}
        for lab, df in zip(labels[1:], dfs[1:]):
            o = df[c].dropna().to_numpy() if c in df.columns else np.array([])
            m = float(np.mean(o)) if o.size else np.nan
            row[f"{lab} mean"] = m
            row[f"{lab} Δ%"] = (m - rmean) / rmean * 100 if rmean else np.nan
            row[f"{lab} KS"] = float(ks_2samp(r, o).statistic) if r.size and o.size else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("metric")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print(table.round(3).to_string())

    print("\n" + "=" * 78)
    print(f"BOUNDARY VERDICT (tolerance ±{args.tol:.0f}%)")
    print("=" * 78)
    for lab, df in zip(labels[1:], dfs[1:]):
        dcount = (len(df) - len(ref)) / max(1, len(ref)) * 100
        drifted = [c for c in cols
                   if not np.isnan(table.loc[c, f"{lab} Δ%"])
                   and abs(table.loc[c, f"{lab} Δ%"]) > args.tol]
        ks_vals = [table.loc[c, f"{lab} KS"] for c in cols
                   if not np.isnan(table.loc[c, f"{lab} KS"])]
        maxks = max(ks_vals) if ks_vals else float("nan")
        status = "OK   " if not drifted and abs(dcount) <= args.tol else "DRIFT"
        print(f"[{status}] {lab}: cell count {dcount:+.1f}%, max KS {maxks:.3f}, "
              f"{len(drifted)}/{len(cols)} metrics past ±{args.tol:.0f}%")
        if drifted:
            print("        " + ", ".join(
                f"{c} ({table.loc[c, f'{lab} Δ%']:+.0f}%)" for c in drifted))

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(args.plots, exist_ok=True)
            for c in cols:
                plt.figure(figsize=(6, 4))
                for lab, df in zip(labels, dfs):
                    v = df[c].dropna().to_numpy()
                    if v.size:
                        plt.hist(v, bins=60, histtype="step", density=True, label=lab)
                plt.title(c)
                plt.xlabel(c)
                plt.ylabel("density")
                plt.legend()
                plt.tight_layout()
                safe = "".join(ch if ch.isalnum() else "_" for ch in c)
                plt.savefig(os.path.join(args.plots, f"hist_{safe}.png"), dpi=110)
                plt.close()
            print(f"\nSaved overlaid histograms to {args.plots}")
        except ImportError:
            print("\n(matplotlib not installed - skipped --plots; pip install matplotlib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
