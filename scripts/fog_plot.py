#!/usr/bin/env python3
"""Fog plot(s): one semi-transparent dot per cell, y = an intensity, x = time.

Each dot is one cell at one timepoint. Two layouts:

  * default single panel - x = Timepoint, pooling every position. Because a
    partial download leaves late timepoints under-covered, the tool computes how
    many positions reached each timepoint, WARNS when that coverage is uneven,
    and annotates each column with its position count - so a timepoint acquired
    in 43 of 125 positions can't masquerade as a complete one.
  * --facet-by <col> - a small-multiple grid, one panel per group (e.g. Well, or
    a condition from the plate map), x = Timepoint, with the per-timepoint median
    drawn as a trend line through the fog.

Grouping by condition: pass ``--facet-by condition`` (runs from the current
exporter carry a ``condition`` column), or ``--platemap plate.csv`` (columns
well,condition) to join one on the fly. Column matching is case-insensitive, so
``--facet-by Condition`` also works. ``--control NAME[,NAME2]`` draws those
groups in dashed grey so a control doesn't read as just another treatment arm.

A log y-axis (the default) cannot show cells with intensity <= 0; rather than
dropping them silently, the tool counts them, WARNS, and states the hidden count
in each panel title so the visible n is honest. ``--linear`` plots every cell.

Usage:
    python fog_plot.py all_measurements.parquet -o out.png --facet-by Well
        [--channel "Intensity Mean (488 nm)"] [--platemap plate.csv]
        [--control Fluorescein] [--dark] [--linear]
"""
from __future__ import annotations
import argparse
import math
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _theme(dark):
    if dark:
        return dict(bg="#14161a", ink="#f0f2f5", ink2="#aab1bd", grid="#2a2e35",
                    dot="#3ddc84", trend="#f0f2f5", ctrl="#8b929c")
    return dict(bg="#ffffff", ink="#1a1a1a", ink2="#666666", grid="#eaeaea",
                dot="#1a9850", trend="#1a1a1a", ctrl="#9aa0a6")


def _resolve_column(df, name):
    """Case-insensitive column lookup (so --facet-by Condition finds condition)."""
    if name in df.columns:
        return name
    low = {str(c).lower(): c for c in df.columns}
    return low.get(str(name).lower())


def _position_cols(df):
    """Columns that identify a distinct imaged position (well + field of view)."""
    cols = [c for c in ("Well", "fov") if c in df.columns]
    return cols or ([df.columns[0]] if len(df.columns) else [])


def _coverage(df):
    """Positions reaching each timepoint. Returns ({timepoint: n_positions}, total).

    A partial acquisition leaves late timepoints present in only some positions;
    pooling them into one Timepoint column hides that. This surfaces it.
    """
    if "Timepoint" not in df.columns or not len(df):
        return {}, 0
    pcols = _position_cols(df)
    total = int(df.drop_duplicates(pcols).shape[0])
    per = {int(t): int(g.drop_duplicates(pcols).shape[0])
           for t, g in df.groupby("Timepoint")}
    return per, total


def _count_nonpositive(y):
    """Cells invisible on a log axis: intensity <= 0 (or non-finite)."""
    y = np.asarray(y, dtype=float)
    return int((~np.isfinite(y)).sum() + (y <= 0).sum())


def _apply_platemap(df, platemap):
    """Add/overwrite a ``condition`` column from a well->condition CSV."""
    pm = pd.read_csv(platemap)
    cols = {str(c).lower().strip(): c for c in pm.columns}
    wcol = cols.get("well") or cols.get("region")
    ccol = cols.get("condition") or cols.get("treatment") or cols.get("group")
    if wcol and ccol and "Well" in df.columns:
        mapping = dict(zip(pm[wcol].astype(str), pm[ccol].astype(str)))
        df["condition"] = df["Well"].astype(str).map(mapping).fillna(df["Well"].astype(str))
    return df


def _facet_title(ax, label, n_note, ink, base_fs=11):
    """A two-line, wrapped, size-adapting title so long condition names don't clip."""
    label = str(label)
    fs = base_fs if len(label) <= 18 else (10 if len(label) <= 28 else 9)
    wrapped = "\n".join(textwrap.wrap(label, 24)) or label
    ax.set_title(f"{wrapped}\n{n_note}", color=ink, fontsize=fs, weight="bold", loc="left")


def _fog(ax, x, y, groups, dot, ink, trend, log, dashed=False):
    """Draw the fog + per-group median bar/IQR + a trend line joining medians."""
    rng = np.random.default_rng(0)
    xj = np.array([groups.index(v) for v in x], float) + (rng.random(len(x)) - .5) * .7
    ax.scatter(xj, y, s=3, c=dot, alpha=0.04, linewidths=0, rasterized=True)
    meds = []
    for i, g in enumerate(groups):
        s = np.asarray(y)[np.asarray(x) == g]
        s = s[np.isfinite(s)]
        if log:
            s = s[s > 0]
        if not len(s):
            meds.append(np.nan); continue
        med = float(np.median(s)); q1, q3 = np.percentile(s, [25, 75]); meds.append(med)
        ax.plot([i - .40, i + .40], [med, med], color=ink, lw=2.0, zorder=6,
                solid_capstyle="round")
        ax.plot([i, i], [q1, q3], color=ink, lw=1.2, alpha=.85, zorder=5)
    xs = [i for i, m in enumerate(meds) if not np.isnan(m)]
    # Trend line: use the theme's trend colour explicitly. Passing color=None here
    # falls through to matplotlib's property cycle (C0 blue) - the original bug.
    ax.plot(xs, [meds[i] for i in xs], color=trend, lw=1.6, alpha=.7, zorder=4,
            ls=("--" if dashed else "-"))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if log:
        ax.set_yscale("log")


def _style_ax(ax, th):
    ax.grid(axis="y", color=th["grid"], lw=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="fog_plot.png")
    ap.add_argument("--channel", default="Intensity Mean (488 nm)")
    ap.add_argument("--facet-by", default=None,
                    help="column to facet by (e.g. Well or condition); case-insensitive")
    ap.add_argument("--platemap", default=None,
                    help="CSV (columns well,condition) to add a condition column")
    ap.add_argument("--control", default="",
                    help="comma-separated facet value(s) to draw in dashed grey as controls")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--linear", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    if args.platemap:
        df = _apply_platemap(df, args.platemap)

    channel = _resolve_column(df, args.channel)
    if channel is None:
        intensity = [c for c in df.columns if str(c).startswith("Intensity")]
        print(f"ERROR: --channel {args.channel!r}: no such column. Intensity columns "
              f"available: {intensity}", file=sys.stderr)
        return 2

    th = _theme(args.dark)
    log = not args.linear
    n_time = df["Timepoint"].nunique() if "Timepoint" in df else 1
    ch = "green (488 nm)" if "488" in str(args.channel) else str(channel)
    controls = {c.strip() for c in args.control.split(",") if c.strip()}

    # Coverage + log-zero honesty, computed once and reported to stderr.
    per, total = _coverage(df)
    if per and len(set(per.values())) > 1:
        counts = ", ".join(f"TP{t}: {n}/{total}" for t, n in sorted(per.items()))
        print(f"WARNING: uneven position coverage across timepoints ({counts} positions). "
              "A timepoint reached by fewer positions is a biased subset, not a like-for-"
              "like column - read late/partial timepoints with care.", file=sys.stderr)
    n_hidden = _count_nonpositive(df[channel].to_numpy(float)) if log else 0
    if n_hidden:
        print(f"WARNING: {n_hidden:,} of {len(df):,} cells have {channel} <= 0 and cannot "
              "appear on the log y-axis (use --linear to include them). Panel n counts "
              "exclude them so the visible total is honest.", file=sys.stderr)

    facet_col = None
    if args.facet_by:
        facet_col = _resolve_column(df, args.facet_by)
        if facet_col is None:
            print(f"ERROR: --facet-by {args.facet_by!r}: no such column. Available: "
                  f"{list(df.columns)}. For condition faceting, pass --platemap or use a "
                  "run whose parquet carries a 'condition' column.", file=sys.stderr)
            return 2

    def _panel_n(sub):
        y = sub[channel].to_numpy(float)
        hidden = _count_nonpositive(y) if log else 0
        note = f"n={len(y) - hidden:,}"
        if hidden:
            note += f"  (+{hidden:,} ≤0 off-log)"
        return note

    if facet_col:
        facets = sorted(df[facet_col].astype(str).unique())
        ncol = min(3, len(facets)); nrow = math.ceil(len(facets) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.8 * nrow),
                                 dpi=150, sharey=True, squeeze=False)
        fig.patch.set_facecolor(th["bg"])
        tvals = sorted(df["Timepoint"].unique())
        pos = df[channel].to_numpy(float)
        pos = pos[np.isfinite(pos) & (pos > 0)] if log else pos[np.isfinite(pos)]
        ylim = (np.quantile(pos, .002), np.quantile(pos, .999)) if len(pos) else None
        for k, fac in enumerate(facets):
            ax = axes[k // ncol][k % ncol]
            ax.set_facecolor(th["bg"])
            sub = df[df[facet_col].astype(str) == fac]
            is_ctrl = fac in controls
            dot = th["ctrl"] if is_ctrl else th["dot"]
            ink = th["ctrl"] if is_ctrl else th["ink"]
            trend = th["ctrl"] if is_ctrl else th["trend"]
            _fog(ax, sub["Timepoint"].tolist(), sub[channel].to_numpy(float),
                 tvals, dot, ink, trend, log, dashed=is_ctrl)
            if ylim:
                ax.set_ylim(ylim)
            label = f"{fac}  (control)" if is_ctrl else fac
            _facet_title(ax, label, _panel_n(sub), th["ink"])
            _style_ax(ax, th)
        for k in range(len(facets), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.supxlabel("Timepoint", color=th["ink2"])
        fig.supylabel(str(channel) + ("  (log)" if log else ""), color=th["ink2"])
        fig.suptitle(f"Per-cell {ch} over time, by {facet_col}   ·   each dot = one cell",
                     color=th["ink"], fontsize=14, weight="bold", x=.01, ha="left")
        if n_time <= 1:
            fig.text(.5, .5, "SINGLE TIMEPOINT - no time course in this data",
                     ha="center", va="center", fontsize=18, color="#c0392b", alpha=.5, rotation=12)
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.94), h_pad=2.2)
    else:
        key = "Timepoint" if n_time > 1 else "Well"
        groups = sorted(df[key].unique())
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
        fig.patch.set_facecolor(th["bg"]); ax.set_facecolor(th["bg"])
        _fog(ax, df[key].tolist(), df[channel].to_numpy(float), groups,
             th["dot"], th["ink"], th["trend"], log)
        # Annotate each timepoint column with how many positions reached it, so an
        # under-covered partial timepoint is visible rather than a silent bias.
        if key == "Timepoint" and per and total:
            ax.set_xticklabels([f"{g}\n{per.get(int(g), 0)}/{total} pos" for g in groups],
                               fontsize=9)
            ax.set_xlabel("Timepoint  (positions reaching each)", color=th["ink2"])
        else:
            ax.set_xlabel(key, color=th["ink2"])
        ax.set_ylabel(str(channel) + ("  (log)" if log else ""), color=th["ink2"])
        ax.set_title(f"Per-cell {ch} intensity", color=th["ink"], fontsize=14,
                     weight="bold", loc="left")
        if log and n_hidden:
            ax.text(0.99, 0.01, f"{n_hidden:,} cells ≤0 not shown (log axis)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#c0392b")
        _style_ax(ax, th); ax.tick_params(labelsize=10)
        fig.tight_layout()

    fig.savefig(args.out, facecolor=th["bg"], dpi=150)
    cov = ""
    if per and len(set(per.values())) > 1:
        cov = " | UNEVEN position coverage (see warning)"
    print(f"wrote {args.out} | timepoints={n_time} | facet={facet_col} | "
          f"cells={len(df)} | hidden_on_log={n_hidden}{cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
