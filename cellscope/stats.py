"""Proper subpopulation statistics for a CellScope measurements table.

The exploratory report (``analyze.py``) shows *what* the distributions look like.
This module answers the inferential question - **does a subpopulation behave
differently, and better, in some conditions than others?** - without the two
mistakes that make cell-biology stats lie:

1. **Pseudo-replication.** 20,000 cells in one well are **not** 20,000
   independent samples; the well (a biological replicate) is. Cells are pooled
   into a per-well summary and every test runs across **wells**, not cells, so a
   difference isn't declared significant just because thousands of correlated
   cells were counted. When only one well exists per condition the fields of view
   are the unit instead, and that is flagged loudly as *technical* replication.

2. **Assuming the subpopulation exists.** The responder gate assumes the
   intensity distribution is bimodal. Sarle's bimodality coefficient is reported
   per group so a unimodal distribution (where the gate is meaningless) is
   visible, not hidden.

Tests: Mann-Whitney U (two groups) or Kruskal-Wallis + Benjamini-Hochberg-
corrected pairwise Mann-Whitney (>2 groups) - all non-parametric, since per-well
fractions and intensities are not normal and the number of wells is small.
Effect size is Cliff's delta; the median difference carries a bootstrap 95% CI.
Trajectories (slope of log10 intensity over time per tracked cell) quantify
*"better over time"* directly, aggregated to the well before testing.

numpy + scipy + pandas only.
"""

from __future__ import annotations

import numpy as np

# Below this many replicate units per group, a p-value is not trustworthy; report
# descriptive numbers and say so rather than implying significance.
_MIN_UNITS = 3
_BC_BIMODAL = 0.555  # Sarle's coefficient for the uniform distribution; > => bimodal-ish


def bimodality_coefficient(x) -> float:
    """Sarle's bimodality coefficient in (0, 1]. > ~0.555 suggests two modes.

    BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3))), with the
    sample-corrected (excess) kurtosis. A heuristic, not a formal test - it flags
    when the "responder subpopulation = second mode" assumption is shaky."""
    from scipy.stats import kurtosis, skew
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 or np.allclose(x, x[0]):
        return float("nan")
    g = float(skew(x, bias=False))
    k = float(kurtosis(x, fisher=True, bias=False))  # excess kurtosis
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g * g + 1.0) / denom) if denom > 0 else float("nan")


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size in [-1, 1]: P(a>b) - P(a<b). 0 = no difference."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


def benjamini_hochberg(pvals) -> list:
    """Benjamini-Hochberg FDR-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return [float(v) for v in out]


def _bootstrap_median_diff_ci(a, b, n_boot=2000, seed=0):
    """95% CI for median(a) - median(b) by resampling units with replacement."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        da = rng.choice(a, a.size, replace=True)
        db = rng.choice(b, b.size, replace=True)
        diffs[i] = np.median(da) - np.median(db)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [float(lo), float(hi)]


def _comparison_timepoint(df, unit_cols):
    """Latest timepoint reached by at least half the units (avoids comparing at a
    barely-acquired final frame). Returns (timepoint, units_present, units_total)."""
    tvals = sorted(int(t) for t in df["Timepoint"].unique())
    total = int(df.drop_duplicates(unit_cols).shape[0])
    chosen, present = tvals[0], 0
    for t in tvals:
        n = int(df[df["Timepoint"] == t].drop_duplicates(unit_cols).shape[0])
        if n >= 0.5 * total:
            chosen, present = t, n
    return chosen, present, total


def _replication_unit(df, group_col):
    """Decide the unit of replication and return (unit_cols, kind, caveat).

    Wells are biological replicates; fields of view within a well are technical.
    Use wells when every group has >=2; otherwise fall back to (Well, fov) and
    flag that the comparison is at the technical-replicate level."""
    if "Well" not in df.columns:
        return [group_col], "group", "no 'Well' column; each group treated as one unit"
    wells_per_group = df.groupby(group_col)["Well"].nunique()
    if wells_per_group.min() >= 2:
        return ["Well"], "well", None
    unit = ["Well", "fov"] if "fov" in df.columns else ["Well"]
    caveat = ("only one well per condition, so fields of view are the replication "
              "unit - these are TECHNICAL replicates, not biological; treat any "
              "p-value as within-sample variation, not a between-subjects effect")
    return unit, ("fov" if "fov" in df.columns else "well"), caveat


def _per_unit_values(df, channel, threshold, group_col, unit_cols, timepoint):
    """One row per replication unit at ``timepoint``: %% responders + median level.

    Returns {group: {"pct_responders": [...], "median_level": [...]}} where each
    list has one number per unit (well/FOV) - the values the tests run on."""
    sub = df[df["Timepoint"] == timepoint]
    out: dict = {}
    keys = [group_col] + unit_cols
    for gkey, u in sub.groupby(keys):
        g = gkey[0]
        vals = u[channel].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out.setdefault(g, {"pct_responders": [], "median_level": []})
        out[g]["pct_responders"].append(100.0 * float(np.mean(vals > threshold)))
        out[g]["median_level"].append(float(np.median(vals)))
    return out


def cell_trajectories(df, channel):
    """Per tracked cell, the slope of log10(intensity) over time (the ramp rate).

    A cell is identified globally by (Dataset, Well, fov, segment, Label) - the
    keys the fixed schema provides - so trajectories are only built within a
    gap-free ``segment`` and never fuse two cells across a missing frame. Cells
    with < 3 timepoints are skipped. Returns a DataFrame with the id columns plus
    ``slope`` (log10 units per timepoint) and ``n_points``.
    """
    import pandas as pd
    from scipy.stats import linregress
    id_cols = [c for c in ("Dataset", "Well", "fov", "segment", "Label") if c in df.columns]
    if "Label" not in id_cols or "Timepoint" not in df.columns:
        return pd.DataFrame(columns=id_cols + ["slope", "n_points"])
    d = df[np.isfinite(df[channel]) & (df[channel] > 0)].copy()
    d["_logc"] = np.log10(d[channel].to_numpy(float))
    rows = []
    for key, cell in d.groupby(id_cols):
        t = cell["Timepoint"].to_numpy(float)
        if np.unique(t).size < 3:
            continue
        y = cell["_logc"].to_numpy(float)
        slope = float(linregress(t, y).slope)
        rec = dict(zip(id_cols, key if isinstance(key, tuple) else (key,)))
        rec["slope"] = slope
        rec["n_points"] = int(np.unique(t).size)
        rows.append(rec)
    return pd.DataFrame(rows, columns=id_cols + ["slope", "n_points"])


def _compare(values_by_group, higher_is_better=True):
    """Non-parametric comparison across groups of per-unit values.

    <2 groups or too few units -> descriptive only. 2 groups -> Mann-Whitney U.
    >2 -> Kruskal-Wallis omnibus + BH-corrected pairwise Mann-Whitney. Each
    pairwise entry carries Cliff's delta and a bootstrap CI on the median gap."""
    from scipy.stats import kruskal, mannwhitneyu
    groups = [g for g in values_by_group if len(values_by_group[g]) > 0]
    n_units = {g: len(values_by_group[g]) for g in groups}
    res: dict = {"n_units": n_units, "medians": {
        g: float(np.median(values_by_group[g])) for g in groups}, "warnings": []}
    small = [g for g in groups if n_units[g] < _MIN_UNITS]
    if small:
        res["warnings"].append(
            f"{', '.join(f'{g} (n={n_units[g]})' for g in small)} has < {_MIN_UNITS} "
            "replicate units; p-values are unreliable - read the medians/CIs, not stars")
    if len(groups) < 2 or all(n_units[g] < 2 for g in groups):
        res["test"] = "descriptive-only (need >=2 groups with >=2 units each)"
        res["pairwise"] = []
        return res

    def pair(a, b):
        va, vb = values_by_group[a], values_by_group[b]
        try:
            p = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        return {"a": a, "b": b, "p_value": p,
                "cliffs_delta": cliffs_delta(va, vb),
                "median_diff": float(np.median(va) - np.median(vb)),
                "median_diff_ci95": _bootstrap_median_diff_ci(va, vb)}

    pairs = [pair(groups[i], groups[j])
             for i in range(len(groups)) for j in range(i + 1, len(groups))]
    for pr, padj in zip(pairs, benjamini_hochberg([p["p_value"] for p in pairs])):
        pr["p_adj_bh"] = padj
    if len(groups) == 2:
        res["test"] = "Mann-Whitney U (two-sided), across replicate units"
        res["p_value"] = pairs[0]["p_value"]
    else:
        try:
            res["test"] = "Kruskal-Wallis omnibus + BH-corrected pairwise Mann-Whitney"
            res["omnibus_p"] = float(kruskal(*[values_by_group[g] for g in groups]).pvalue)
        except ValueError:
            res["omnibus_p"] = float("nan")
    res["higher_is_better"] = higher_is_better
    res["pairwise"] = pairs
    return res


def subpopulation_stats(df, channel, threshold, group_col="group"):
    """Full inferential summary. ``df`` is the gated table; ``threshold`` the
    responder gate (linear units). Returns a JSON-serializable dict."""
    import pandas as pd  # noqa: F401
    unit_cols, unit_kind, unit_caveat = _replication_unit(df, group_col)
    tp, units_present, units_total = _comparison_timepoint(df, unit_cols)
    per_unit = _per_unit_values(df, channel, threshold, group_col, unit_cols, tp)

    resp = {g: v["pct_responders"] for g, v in per_unit.items()}
    level = {g: v["median_level"] for g, v in per_unit.items()}

    rep = {
        "channel": channel,
        "responder_threshold": float(threshold),
        "group_column": group_col,
        "replication_unit": unit_kind,
        "replication_unit_columns": unit_cols,
        "replication_caveat": unit_caveat,
        "comparison_timepoint": int(tp),
        "units_at_comparison_timepoint": int(units_present),
        "units_total": int(units_total),
        "responder_pct_by_condition": _compare(resp, higher_is_better=True),
        "median_intensity_by_condition": _compare(level, higher_is_better=True),
        "bimodality": {},
        "notes": [],
    }
    if unit_caveat:
        rep["notes"].append(unit_caveat)

    # Bimodality per group at the comparison timepoint: does the subpopulation
    # actually exist as a second mode, or is the gate splitting one blob?
    endpoint = df[df["Timepoint"] == tp]
    for g, u in endpoint.groupby(group_col):
        vals = u[channel].to_numpy(float)
        vals = np.log10(vals[np.isfinite(vals) & (vals > 0)])
        bc = bimodality_coefficient(vals)
        rep["bimodality"][str(g)] = {
            "bimodality_coefficient": (None if bc != bc else round(bc, 3)),
            "looks_bimodal": (None if bc != bc else bool(bc > _BC_BIMODAL))}
    if rep["bimodality"] and all(
            v["looks_bimodal"] is False for v in rep["bimodality"].values()):
        rep["notes"].append(
            "no group looks bimodal (Sarle's coefficient < 0.555 everywhere), so the "
            "responder gate is splitting a single distribution - interpret 'responders' "
            "as a top-fraction cut, not a distinct subpopulation")

    # Trajectories: ramp rate (slope of log10 intensity over time), tested at the
    # well level too - the direct measure of "behaves better over time".
    traj = cell_trajectories(df, channel)
    if len(traj):
        gmap = df.drop_duplicates(unit_cols + [group_col]).set_index(unit_cols)[group_col]
        tkey = [c for c in unit_cols if c in traj.columns]
        if tkey:
            per_unit_slope = (traj.groupby(tkey)["slope"].median().reset_index())
            per_unit_slope["_g"] = per_unit_slope.set_index(tkey).index.map(gmap)
            slope_by_group = {g: sdf["slope"].to_numpy(float).tolist()
                              for g, sdf in per_unit_slope.groupby("_g")}
            rep["trajectory_slope_by_condition"] = _compare(slope_by_group, higher_is_better=True)
            rep["trajectory_slope_by_condition"]["units"] = "log10(intensity) per timepoint"
            rep["n_cells_with_trajectory"] = int(len(traj))
    else:
        rep["notes"].append(
            "no per-cell trajectories (need >=3 timepoints per tracked cell within a "
            "gap-free segment); trajectory test skipped")
    return rep


def format_stats(rep: dict) -> str:
    """Plain-text summary of the inferential result."""
    lines = [f"Subpopulation statistics ({rep['channel']}):",
             f"  replication unit: {rep['replication_unit']} "
             f"({rep['units_at_comparison_timepoint']}/{rep['units_total']} at "
             f"timepoint {rep['comparison_timepoint']})"]
    if rep.get("replication_caveat"):
        lines.append(f"  ! {rep['replication_caveat']}")

    def render(title, block):
        if not block:
            return
        lines.append(f"  {title}: {block.get('test', 'n/a')}")
        for g, m in block.get("medians", {}).items():
            lines.append(f"      {g}: median={m:.3g} (n={block['n_units'].get(g, 0)} units)")
        for pr in block.get("pairwise", []):
            p = pr.get("p_adj_bh", pr.get("p_value"))
            lines.append(f"      {pr['a']} vs {pr['b']}: p={_p(p)}, "
                         f"Cliff's d={pr['cliffs_delta']:+.2f}")
        for w in block.get("warnings", []):
            lines.append(f"      ! {w}")

    render("responder %", rep.get("responder_pct_by_condition"))
    render("median intensity", rep.get("median_intensity_by_condition"))
    render("ramp rate (trajectory slope)", rep.get("trajectory_slope_by_condition"))
    for n in rep.get("notes", []):
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def _p(v):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.3g}"
