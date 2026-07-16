#!/usr/bin/env python3
"""Fog plot: one semi-transparent dot per cell, y = an intensity, x = time.

If the parquet has a single timepoint (nothing to trend over), the x-axis falls
back to Well so the per-cell distribution is still shown. Overlays each group's
median and interquartile range so the summary reads through the fog.

Usage:
    python fog_plot.py all_measurements.parquet [-o out.png]
        [--channel "Intensity Mean (488 nm)"] [--dark]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="fog_plot.png")
    ap.add_argument("--channel", default="Intensity Mean (488 nm)")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    y = df[args.channel].to_numpy(dtype=float)

    # Time if it varies; otherwise fall back to Well.
    if "Timepoint" in df and df["Timepoint"].nunique() > 1:
        key, xlabel = "Timepoint", "Timepoint"
    else:
        key, xlabel = "Well", "Well"
    groups = list(dict.fromkeys(df[key].tolist()))
    try:
        groups = sorted(groups)
    except TypeError:
        pass
    pos = {g: i for i, g in enumerate(groups)}
    xg = df[key].map(pos).to_numpy(dtype=float)

    # Theme
    if args.dark:
        bg, ink, ink2, grid, dot = "#14161a", "#f0f2f5", "#aab1bd", "#2a2e35", "#3ddc84"
    else:
        bg, ink, ink2, grid, dot = "#ffffff", "#1a1a1a", "#666666", "#eaeaea", "#1a9850"

    rng = np.random.default_rng(0)
    jitter = (rng.random(len(df)) - 0.5) * 0.7
    x = xg + jitter

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    ax.scatter(x, y, s=3, c=dot, alpha=0.03, linewidths=0, rasterized=True)

    # Median + IQR overlay per group (reads through the fog).
    for g in groups:
        sub = df.loc[df[key] == g, args.channel].to_numpy(dtype=float)
        med = np.median(sub); q1, q3 = np.percentile(sub, [25, 75])
        i = pos[g]
        ax.plot([i - 0.42, i + 0.42], [med, med], color=ink, lw=2.2, zorder=5,
                solid_capstyle="round")
        ax.plot([i, i], [q1, q3], color=ink, lw=1.4, alpha=0.9, zorder=4,
                solid_capstyle="round")
        ax.annotate(f"{med:,.0f}", (i, med), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9, color=ink, weight="bold")
        ax.annotate(f"n={len(sub):,}", (i, ax.get_ylim()[0] if False else q1),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=8, color=ink2)

    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel, fontsize=11, color=ink2)
    ax.set_ylabel(args.channel + "  (log)", fontsize=11, color=ink2)
    ch = "green (488 nm)" if "488" in args.channel else args.channel
    sub2 = "each dot = one cell" + ("" if key == "Timepoint"
           else "   ·   single timepoint (t=0), so grouped by well")
    ax.set_title(f"Per-cell {ch} intensity", fontsize=14, color=ink, weight="bold", loc="left", pad=18)
    ax.annotate(sub2, (0, 1.015), xycoords="axes fraction", fontsize=10, color=ink2)

    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=bg, dpi=150)
    print("wrote", args.out, "| x-axis:", xlabel, "| cells:", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
