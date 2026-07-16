"""Runtime QC on a combined CellScope parquet.

The pipeline's worst failure mode is emitting plausible-looking but wrong data
with no warning. This surfaces the silent modes so a run flags them itself:

  * per-cell keys that are not unique (e.g. a missing ``fov`` column collapsing
    cells from different fields of view onto one Label),
  * rows with a missing / blank channel (NaN intensity),
  * cells at or above sensor saturation,
  * positions missing timepoints, and gaps in the timepoint sequence,
  * a per-timepoint coverage matrix (positions reaching each timepoint), and a
    flag when a timepoint is present in under half the positions - which makes a
    partial acquisition read as a biological crash rather than absent data.

Writes ``qc.json`` and returns the report; ``format_issues`` renders warnings.
"""

from __future__ import annotations

import json

_SATURATION = 65504.0  # float16 cap; below uint16's 65535, so it fires for both


def qc_report(parquet: str, out_json: str | None = None) -> dict:
    import pandas as pd

    df = pd.read_parquet(parquet)
    n = len(df)
    rep: dict = {"rows": int(n), "issues": []}
    issues = rep["issues"]

    # --- per-cell key uniqueness -----------------------------------------
    if {"Well", "Timepoint", "Label"}.issubset(df.columns):
        key = [c for c in ("Dataset", "Well", "fov", "Timepoint", "Label") if c in df.columns]
        dup = int(df.duplicated(subset=key).sum())
        rep["key_columns"] = key
        rep["duplicate_key_rows"] = dup
        if "fov" not in df.columns:
            issues.append("no 'fov' column: Label restarts per field of view, so the "
                          "per-cell key collides across the FOVs pooled into a well")
        if dup:
            issues.append(f"{dup:,} rows ({100 * dup / max(1, n):.1f}%) share a per-cell "
                          f"key {tuple(key)} - cells are not uniquely identified")

    # --- missing / blank channels (NaN intensity) ------------------------
    mean_cols = [c for c in df.columns if c.startswith("Intensity Mean (")]
    nan_by_chan = {c: int(df[c].isna().sum()) for c in mean_cols}
    rep["nan_intensity_rows"] = nan_by_chan
    for c, k in nan_by_chan.items():
        if k:
            issues.append(f"{k:,} rows have no {c} (missing/blank channel image)")

    # --- saturation ------------------------------------------------------
    max_cols = [c for c in df.columns if c.startswith("Intensity Max (")]
    sat = {c: int((df[c] >= _SATURATION).sum()) for c in max_cols}
    rep["saturated_rows"] = sat
    for c, k in sat.items():
        if k:
            issues.append(f"{k:,} cells at/above sensor saturation in {c}")

    # --- timepoint coverage ----------------------------------------------
    if {"Well", "Timepoint"}.issubset(df.columns):
        tps = sorted(int(t) for t in df["Timepoint"].unique())
        rep["timepoints"] = tps
        if tps and (tps[-1] - tps[0] + 1) != len(tps):
            issues.append(f"timepoint sequence has gaps: {tps}")
        grp = ["Well"] + (["fov"] if "fov" in df.columns else [])
        per_pos = df.groupby(grp)["Timepoint"].nunique()
        n_pos = int(len(per_pos))
        rep["positions"] = n_pos
        # Coverage matrix: how many positions reached each timepoint. A timepoint
        # present in only a fraction of positions (e.g. a partial download) makes
        # any count-over-time look like a biological crash when it is just absent
        # positions - so surface it as data, and flag the under-covered ones.
        cover = {int(t): int(g.drop_duplicates(grp).shape[0])
                 for t, g in df.groupby("Timepoint")}
        rep["positions_per_timepoint"] = cover
        short = int((per_pos < len(tps)).sum())
        if short:
            issues.append(f"{short} of {n_pos} positions are missing timepoints "
                          f"(partial coverage vs {len(tps)} total)")
        # A timepoint reached by well under half the positions is almost always an
        # acquisition artifact masquerading as data.
        under = {t: c for t, c in cover.items() if n_pos and c < 0.5 * n_pos}
        if under:
            worst = ", ".join(f"TP{t}: {c}/{n_pos}" for t, c in sorted(under.items()))
            issues.append(f"timepoint(s) reached by <50% of positions ({worst}) - "
                          f"a count-over-time will read as a crash, not biology")

    rep["ok"] = not issues
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return rep


def format_issues(rep: dict) -> str:
    """One-line-per-issue summary (empty string if the report is clean)."""
    if rep.get("ok"):
        return "QC: no issues found."
    lines = [f"QC: {len(rep['issues'])} issue(s) found -"]
    lines += [f"  - {msg}" for msg in rep["issues"]]
    return "\n".join(lines)
