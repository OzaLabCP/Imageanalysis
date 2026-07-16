#!/usr/bin/env python3
"""Fog plot(s): one semi-transparent dot per cell, y = an intensity, x = time.

Modes:
  * single panel  - x = Timepoint (if the data has more than one) else Well.
  * --facet-by Well - a small-multiple grid, one panel per well, x = Timepoint,
    with the per-timepoint median drawn as a trend line through the fog.

If there is only one timepoint the x-axis (or facet time course) collapses; the
tool says so rather than drawing a misleading single column.

Usage:
    python fog_plot.py all_measurements.parquet -o out.png
        [--channel "Intensity Mean (488 nm)"] [--facet-by Well] [--dark] [--linear]
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _theme(dark):
    if dark:
        return dict(bg="#14161a", ink="#f0f2f5", ink2="#aab1bd", grid="#2a2e35",
                    dot="#3ddc84", trend="#f0f2f5")
    return dict(bg="#ffffff", ink="#1a1a1a", ink2="#666666", grid="#eaeaea",
                dot="#1a9850", trend="#1a1a1a")


def _fog(ax, x, y, t, dot, ink, ink2, groups, log):
    rng = np.random.default_rng(0)
    xj = np.array([groups.index(v) for v in x], float) + (rng.random(len(x)) - .5) * .7
    ax.scatter(xj, y, s=3, c=dot, alpha=0.04, linewidths=0, rasterized=True)
    # median (+IQR) per group, and a trend line joining medians
    meds = []
    for i, g in enumerate(groups):
        s = y[np.asarray(x) == g]
        if not len(s):
            meds.append(np.nan); continue
        med = np.median(s); q1, q3 = np.percentile(s, [25, 75]); meds.append(med)
        ax.plot([i - .40, i + .40], [med, med], color=ink, lw=2.0, zorder=6,
                solid_capstyle="round")
        ax.plot([i, i], [q1, q3], color=ink, lw=1.2, alpha=.85, zorder=5)
    xs = [i for i, m in enumerate(meds) if not np.isnan(m)]
    ax.plot(xs, [meds[i] for i in xs], color=t, lw=1.6, alpha=.55, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if log:
        ax.set_yscale("log")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="fog_plot.png")
    ap.add_argument("--channel", default="Intensity Mean (488 nm)")
    ap.add_argument("--facet-by", default=None)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--linear", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    th = _theme(args.dark)
    log = not args.linear
    n_time = df["Timepoint"].nunique() if "Timepoint" in df else 1
    ch = "green (488 nm)" if "488" in args.channel else args.channel

    if args.facet_by and args.facet_by in df:
        facets = sorted(df[args.facet_by].unique())
        ncol = min(3, len(facets)); nrow = math.ceil(len(facets) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                                 dpi=150, sharey=True, squeeze=False)
        fig.patch.set_facecolor(th["bg"])
        tvals = sorted(df["Timepoint"].unique())
        ylim = (df[args.channel].quantile(.002), df[args.channel].quantile(.999))
        for k, fac in enumerate(facets):
            ax = axes[k // ncol][k % ncol]
            ax.set_facecolor(th["bg"])
            sub = df[df[args.facet_by] == fac]
            _fog(ax, sub["Timepoint"].tolist(), sub[args.channel].to_numpy(float),
                 None, th["dot"], th["ink"], th["ink2"], tvals, log)
            ax.set_ylim(ylim)
            ax.set_title(f"{args.facet_by} {fac}   (n={len(sub):,})", color=th["ink"],
                         fontsize=11, weight="bold", loc="left")
            ax.grid(axis="y", color=th["grid"], lw=.7); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color(th["grid"])
            ax.tick_params(colors=th["ink2"], labelsize=9)
        for k in range(len(facets), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.supxlabel("Timepoint", color=th["ink2"])
        fig.supylabel(args.channel + ("  (log)" if log else ""), color=th["ink2"])
        fig.suptitle(f"Per-cell {ch} over time, by {args.facet_by}   ·   each dot = one cell",
                     color=th["ink"], fontsize=14, weight="bold", x=.01, ha="left")
        if n_time <= 1:
            fig.text(.5, .5, "SINGLE TIMEPOINT — no time course in this data",
                     ha="center", va="center", fontsize=18, color="#c0392b", alpha=.5, rotation=12)
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    else:
        key = "Timepoint" if n_time > 1 else "Well"
        groups = sorted(df[key].unique())
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
        fig.patch.set_facecolor(th["bg"]); ax.set_facecolor(th["bg"])
        _fog(ax, df[key].tolist(), df[args.channel].to_numpy(float), None,
             th["dot"], th["ink"], th["ink2"], groups, log)
        ax.set_xlabel(key, color=th["ink2"]); ax.set_ylabel(args.channel + ("  (log)" if log else ""), color=th["ink2"])
        ax.set_title(f"Per-cell {ch} intensity", color=th["ink"], fontsize=14, weight="bold", loc="left")
        ax.grid(axis="y", color=th["grid"], lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(th["grid"])
        ax.tick_params(colors=th["ink2"])
        fig.tight_layout()

    fig.savefig(args.out, facecolor=th["bg"], dpi=150)
    print(f"wrote {args.out} | timepoints={n_time} | facet={args.facet_by} | cells={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
