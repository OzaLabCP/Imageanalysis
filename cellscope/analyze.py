"""Automated per-dataset analysis report from a CellScope measurements parquet.

Turns an ``all_measurements.parquet`` (one row per cell per timepoint) into a
self-contained report aimed at one question: **is there a subpopulation that
behaves differently - and better - in some conditions?** Population-level, so it
does not depend on single-cell tracking (the parquet pools FOV, so per-cell
trajectories are not reconstructable).

It writes, into an output folder:
  * ``fog_over_time.png``            - per-group panels, one dot per cell, x=time
  * ``distributions_over_time.png``  - per-group violins per timepoint (bimodality)
  * ``responder_fraction.png``       - %% cells above a data-driven responder gate,
                                       per group over time  (the headline figure)
  * ``percentile_bands.png``         - median vs top-decile (p90) over time, per group
  * ``responder_characterization.png`` - what responders are (size / red / shape)
  * ``group_timepoint_summary.csv``  - n, medians, %%responders, percentiles
  * ``responder_characteristics.csv``- responder vs non-responder metrics
  * ``index.html``                   - the figures + a short written summary

Run standalone (``cellscope-analyze parquet -o report``) or automatically after a
batch run (``cellscope-batch ... --analyze``). Needs pandas + matplotlib.
"""

from __future__ import annotations

import argparse
import html
import os

import numpy as np

GREEN_DEFAULT = "Intensity Mean (488 nm)"
RED_DEFAULT = "Intensity Mean (638 nm)"

# Okabe-Ito colourblind-safe categorical palette (fixed order, never cycled).
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
            "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
_INK, _INK2, _GRID, _GREEN = "#1a1a1a", "#666666", "#eaeaea", "#1a9850"


def _colors(groups) -> dict:
    return {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}


def _resolve_channel(df, channel):
    if channel in df.columns:
        return channel
    cands = [c for c in df.columns if c.startswith("Intensity Mean (")]
    green = [c for c in cands if "488" in c or "green" in c.lower()]
    pick = green or cands
    if not pick:
        raise ValueError(f"No 'Intensity Mean (...)' column found (wanted {channel!r})")
    return pick[0]


def _load(parquet: str, platemap: str | None, channel: str):
    import pandas as pd
    df = pd.read_parquet(parquet)
    channel = _resolve_channel(df, channel)
    df["Well"] = df["Well"].astype(str)
    if platemap and os.path.exists(platemap):
        pm = pd.read_csv(platemap)
        cols = {c.lower().strip(): c for c in pm.columns}
        wcol = cols.get("well") or cols.get("region")
        ccol = cols.get("condition") or cols.get("treatment") or cols.get("group")
        if wcol and ccol:
            mapping = dict(zip(pm[wcol].astype(str), pm[ccol].astype(str)))
            df["group"] = df["Well"].map(mapping).fillna(df["Well"])
        else:
            df["group"] = df["Well"]
    else:
        df["group"] = df["Well"]
    return df, channel


def _gate(df, channel, min_diam, max_ecc):
    keep = df[channel] > 0
    if "Diameter (Equivalent) (um)" in df.columns:
        keep &= df["Diameter (Equivalent) (um)"] >= min_diam
    if "Eccentricity" in df.columns:
        keep &= df["Eccentricity"] <= max_ecc
    return df[keep].copy()


def _responder_threshold(green) -> float:
    """Data-driven responder gate: Otsu split on log10(intensity). Returns the
    threshold in the SAME (linear) units as ``green``."""
    from cellscope.analysis.segmentation import otsu_threshold
    vals = np.asarray(green, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return float("inf")
    thr_log = otsu_threshold(np.log10(vals))
    return float(10.0 ** thr_log)


def _style(ax):
    ax.grid(axis="y", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=9)


def _fig_fog_over_time(df, channel, groups, tvals, thr, out):
    import math
    import matplotlib.pyplot as plt
    ncol = min(3, len(groups))
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.6 * nrow),
                             dpi=140, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    tpos = {t: i for i, t in enumerate(tvals)}
    ylim = (df[channel].quantile(0.002), df[channel].quantile(0.999))
    for k, g in enumerate(groups):
        ax = axes[k // ncol][k % ncol]
        sub = df[df["group"] == g]
        x = sub["Timepoint"].map(tpos).to_numpy(float) + (rng.random(len(sub)) - 0.5) * 0.7
        ax.scatter(x, sub[channel], s=3, c=_GREEN, alpha=0.05, linewidths=0, rasterized=True)
        med = sub.groupby("Timepoint")[channel].median()
        ax.plot([tpos[t] for t in med.index], med.values, color=_INK, lw=1.8, zorder=5)
        ax.axhline(thr, color="#d1495b", lw=1.1, ls="--", zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(ylim)
        ax.set_xticks(range(len(tvals)))
        ax.set_xticklabels(tvals)
        ax.set_title(f"{g}   (n={len(sub):,})", color=_INK, fontsize=11, weight="bold", loc="left")
        _style(ax)
    for k in range(len(groups), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supxlabel("Timepoint", color=_INK2)
    fig.supylabel(f"{channel} (log)", color=_INK2)
    fig.suptitle("Per-cell green over time (dashed = responder gate)",
                 color=_INK, fontsize=14, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _fig_distributions(df, channel, groups, tvals, thr, out):
    import math
    import matplotlib.pyplot as plt
    ncol = min(3, len(groups))
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.6 * nrow),
                             dpi=140, sharey=True, squeeze=False)
    for k, g in enumerate(groups):
        ax = axes[k // ncol][k % ncol]
        sub = df[df["group"] == g]
        data = [np.log10(sub[sub["Timepoint"] == t][channel].clip(lower=1)) for t in tvals]
        data = [d.to_numpy() for d in data]
        positions = list(range(len(tvals)))
        good = [(p, d) for p, d in zip(positions, data) if d.size > 1]
        if good:
            vp = ax.violinplot([d for _, d in good], positions=[p for p, _ in good],
                               widths=0.8, showmedians=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(_GREEN)
                b.set_alpha(0.5)
                b.set_edgecolor(_INK)
            if "cmedians" in vp:
                vp["cmedians"].set_color(_INK)
        ax.axhline(np.log10(thr), color="#d1495b", lw=1.1, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(tvals)
        ax.set_title(f"{g}", color=_INK, fontsize=11, weight="bold", loc="left")
        _style(ax)
    for k in range(len(groups), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supxlabel("Timepoint", color=_INK2)
    fig.supylabel(f"log10 {channel}", color=_INK2)
    fig.suptitle("Green distribution per timepoint - two modes = a subpopulation",
                 color=_INK, fontsize=14, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _fig_responder_fraction(frac, groups, tvals, out):
    import matplotlib.pyplot as plt
    colors = _colors(groups)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for g in groups:
        s = frac[frac["group"] == g].sort_values("Timepoint")
        ax.plot(s["Timepoint"], s["pct_responders"], "-o", color=colors[g],
                lw=2, ms=5, label=str(g))
    ax.set_xlabel("Timepoint", color=_INK2)
    ax.set_ylabel("% cells above responder gate", color=_INK2)
    ax.set_title("Responder subpopulation fraction over time, by group",
                 color=_INK, fontsize=14, weight="bold", loc="left")
    ax.set_xticks(tvals)
    ax.legend(frameon=False, fontsize=10, title="group")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _fig_percentile_bands(df, channel, groups, tvals, out):
    import matplotlib.pyplot as plt
    colors = _colors(groups)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for g in groups:
        sub = df[df["group"] == g]
        med = sub.groupby("Timepoint")[channel].median()
        p90 = sub.groupby("Timepoint")[channel].quantile(0.90)
        ax.plot(med.index, med.values, "-", color=colors[g], lw=2, label=f"{g} median")
        ax.plot(p90.index, p90.values, "--", color=colors[g], lw=1.6, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Timepoint", color=_INK2)
    ax.set_ylabel(f"{channel} (log)", color=_INK2)
    ax.set_title("Median (solid) vs top decile p90 (dashed): a rising tail = responders",
                 color=_INK, fontsize=13, weight="bold", loc="left")
    ax.set_xticks(tvals)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _fig_responder_characterization(df, channel, thr, out):
    import matplotlib.pyplot as plt
    df = df.copy()
    df["responder"] = np.where(df[channel] > thr, "responder", "non-responder")
    metrics = [m for m in ("Diameter (Equivalent) (um)", RED_DEFAULT, "Eccentricity")
               if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2), dpi=140,
                             squeeze=False)
    order = ["non-responder", "responder"]
    for ax, m in zip(axes[0], metrics):
        data = [df[df["responder"] == r][m].dropna().to_numpy() for r in order]
        data = [d for d in data if d.size > 1]
        if data:
            vp = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=False)
            for b, c in zip(vp["bodies"], ("#999999", _GREEN)):
                b.set_facecolor(c)
                b.set_alpha(0.6)
            if "cmedians" in vp:
                vp["cmedians"].set_color(_INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=9)
        ax.set_title(m, color=_INK, fontsize=10, weight="bold")
        _style(ax)
    fig.suptitle("What are the responders? (green-gated vs the rest)",
                 color=_INK, fontsize=13, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def run(parquet, outdir, platemap=None, channel=GREEN_DEFAULT,
        min_diameter=6.0, max_eccentricity=0.95):
    """Generate the full report into ``outdir``. Returns a summary dict."""
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)

    # QC first: the report is only as trustworthy as its input, so surface silent
    # corruption modes (non-unique per-cell keys, missing/blank channels,
    # saturation, missing timepoints) at the top of the report rather than
    # letting them quietly skew every figure below.
    qc = None
    try:
        from cellscope.qc import qc_report
        qc = qc_report(parquet, os.path.join(outdir, "qc.json"))
    except Exception:  # noqa: BLE001 - QC must never block the report
        qc = None

    raw, channel = _load(parquet, platemap, channel)
    df = _gate(raw, channel, min_diameter, max_eccentricity)
    groups = sorted(df["group"].unique())
    tvals = sorted(df["Timepoint"].unique())
    thr = _responder_threshold(df[channel])

    # --- summary tables ---------------------------------------------------
    df["responder"] = df[channel] > thr
    rows = []
    for (g, t), sub in df.groupby(["group", "Timepoint"]):
        rows.append({
            "group": g, "Timepoint": t, "n": len(sub),
            "median_green": float(sub[channel].median()),
            "mean_green": float(sub[channel].mean()),
            "p90_green": float(sub[channel].quantile(0.90)),
            "pct_responders": 100.0 * float(sub["responder"].mean()),
        })
    summary = pd.DataFrame(rows).sort_values(["group", "Timepoint"])
    summary.to_csv(os.path.join(outdir, "group_timepoint_summary.csv"), index=False)

    charac = (df.assign(responder=np.where(df["responder"], "responder", "non-responder"))
                .groupby("responder")
                .agg(n=(channel, "size"),
                     median_green=(channel, "median"),
                     median_diameter=("Diameter (Equivalent) (um)", "median")
                     if "Diameter (Equivalent) (um)" in df.columns else (channel, "median"),
                     median_red=(RED_DEFAULT, "median")
                     if RED_DEFAULT in df.columns else (channel, "median"),
                     median_ecc=("Eccentricity", "median")
                     if "Eccentricity" in df.columns else (channel, "median")))
    charac.to_csv(os.path.join(outdir, "responder_characteristics.csv"))

    # --- figures ----------------------------------------------------------
    figs = []
    if len(tvals) > 1:
        _fig_fog_over_time(df, channel, groups, tvals, thr,
                           os.path.join(outdir, "fog_over_time.png"))
        figs.append(("fog_over_time.png", "Per-cell green over time (dashed = responder gate)"))
        _fig_distributions(df, channel, groups, tvals, thr,
                           os.path.join(outdir, "distributions_over_time.png"))
        figs.append(("distributions_over_time.png",
                     "Green distribution per timepoint - two modes reveal a subpopulation"))
        _fig_responder_fraction(summary, groups, tvals,
                                os.path.join(outdir, "responder_fraction.png"))
        figs.append(("responder_fraction.png",
                     "Responder fraction over time, by group (the headline)"))
        _fig_percentile_bands(df, channel, groups, tvals,
                              os.path.join(outdir, "percentile_bands.png"))
        figs.append(("percentile_bands.png", "Median vs top-decile over time"))
    else:
        # single timepoint: fall back to a by-group fog + distributions
        _fig_fog_over_time(df, channel, groups, tvals, thr,
                           os.path.join(outdir, "fog_over_time.png"))
        figs.append(("fog_over_time.png", "Per-cell green by group (single timepoint)"))
    _fig_responder_characterization(df, channel, thr,
                                    os.path.join(outdir, "responder_characterization.png"))
    figs.append(("responder_characterization.png", "What the responders are"))

    # --- html report ------------------------------------------------------
    n_resp = int(df["responder"].sum())
    grouped = "condition" if (platemap and os.path.exists(platemap)) else "well"
    _write_html(outdir, parquet, channel, thr, len(raw), len(df), n_resp,
                grouped, groups, tvals, summary, figs, qc)
    return {"cells_total": len(raw), "cells_gated": len(df), "responders": n_resp,
            "threshold": thr, "groups": groups, "timepoints": tvals,
            "qc_ok": (qc.get("ok") if qc else None),
            "qc_issues": (qc.get("issues") if qc else [])}


def _write_html(outdir, parquet, channel, thr, n_raw, n_gated, n_resp,
                grouped, groups, tvals, summary, figs, qc=None):
    import pandas as pd  # noqa: F401
    esc = html.escape
    parts = [
        "<h1>CellScope subpopulation report</h1>",
        f"<p><b>Source:</b> {esc(os.path.basename(parquet))} &middot; "
        f"grouped by <b>{grouped}</b> ({esc(', '.join(map(str, groups)))}) &middot; "
        f"{len(tvals)} timepoint(s)</p>",
        f"<p><b>Cells:</b> {n_raw:,} measured &rarr; {n_gated:,} after gating &middot; "
        f"<b>Responder gate</b> (Otsu on log green): {esc(channel)} &gt; {thr:,.0f} "
        f"&middot; {n_resp:,} responders ({100.0*n_resp/max(1,n_gated):.1f}%)</p>",
        "<p style='color:#666'>A subpopulation shows up as a second (high) mode in the "
        "distribution and as a rising responder fraction / top decile in some groups but "
        "not others. Population-level (no single-cell tracking).</p>",
    ]
    # QC banner: green if clean, amber with the specific issues if not. Silent
    # data corruption is the failure mode this whole exercise exists to catch, so
    # it belongs at the top of the report, not buried in a sidecar file.
    if qc is not None:
        if qc.get("ok"):
            parts.append("<div style='background:#e8f5e9;border:1px solid #a5d6a7;"
                         "border-radius:6px;padding:8px 12px;margin:10px 0;color:#1b5e20'>"
                         "&#10003; QC: no data-integrity issues found.</div>")
        else:
            items = "".join(f"<li>{esc(str(m))}</li>" for m in qc.get("issues", []))
            parts.append("<div style='background:#fff3e0;border:1px solid #ffcc80;"
                         "border-radius:6px;padding:8px 12px;margin:10px 0;color:#8a5000'>"
                         f"<b>&#9888; QC: {len(qc.get('issues', []))} data-integrity "
                         f"issue(s)</b> (see qc.json)<ul>{items}</ul></div>")
    for fname, caption in figs:
        parts.append(f"<h3>{esc(caption)}</h3><img src='{esc(fname)}' style='max-width:100%'>")
    parts.append("<h3>Per-group, per-timepoint summary</h3>")
    parts.append(summary.round(1).to_html(index=False, border=0))
    parts.append("<p style='color:#888'>Tables: group_timepoint_summary.csv, "
                 "responder_characteristics.csv</p>")
    doc = ("<!doctype html><meta charset='utf-8'><title>CellScope report</title>"
           "<style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;"
           "margin:24px auto;padding:0 16px;color:#1a1a1a}img{margin:6px 0}"
           "table{border-collapse:collapse;font-size:13px}td,th{padding:3px 8px;"
           "border-bottom:1px solid #eee;text-align:right}</style>" + "".join(parts))
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cellscope-analyze",
        description="Subpopulation analysis report from a CellScope measurements parquet.")
    ap.add_argument("parquet", help="all_measurements.parquet")
    ap.add_argument("-o", "--out", default="cellscope_report", help="output folder")
    ap.add_argument("--platemap", default=None,
                    help="CSV mapping Well->condition (columns: well, condition)")
    ap.add_argument("--channel", default=GREEN_DEFAULT, help="intensity column to analyze")
    ap.add_argument("--min-diameter", type=float, default=6.0,
                    help="drop cells smaller than this (um) as debris")
    ap.add_argument("--max-eccentricity", type=float, default=0.95,
                    help="drop very elongated (likely merged) objects")
    args = ap.parse_args(argv)
    try:
        import pandas  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError:
        print("cellscope-analyze needs pandas + matplotlib: "
              "pip install 'cellscope[analysis]'", flush=True)
        return 1
    info = run(args.parquet, args.out, platemap=args.platemap, channel=args.channel,
               min_diameter=args.min_diameter, max_eccentricity=args.max_eccentricity)
    print(f"Report -> {os.path.join(args.out, 'index.html')}  "
          f"({info['cells_gated']:,} cells, {len(info['groups'])} groups, "
          f"{len(info['timepoints'])} timepoints, {info['responders']:,} responders)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
