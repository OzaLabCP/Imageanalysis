"""GUI-free export of per-cell measurements.

Shared by the desktop Results tab and the headless batch runner, so both write
exactly the same tidy long-format table (one row per cell per timepoint). Has no
Qt dependency, so it is safe to import in headless / cluster contexts.

Two output shapes are offered:

* ``write_measurements_csv`` - CellScope's native tidy CSV (well/region/fov,
  condition, per-channel mean+total intensity, ...), 1-based ``time``.
* ``write_measurements_parquet`` - a fixed "regionprops-style" parquet schema
  (``Label``, diameters, axis lengths, perimeter, per-channel
  Mean/Max/STD/Min intensity, ``Eccentricity``, ``Dataset``, 0-based
  ``Timepoint``, ``Well``) for pipelines built around that column layout.
"""

from __future__ import annotations

import csv


def split_region_fov(well_id: str) -> tuple[str, str]:
    """Split a well id into (region, fov).

    Cephla positions are ``"<region>-<fov>"`` (e.g. ``B2-0`` -> ``("B2", "0")``);
    plain wells have no fov (``"A1"`` -> ``("A1", "")``). The region column lets
    downstream analysis pool per well without string surgery.
    """
    if "-" in well_id:
        region, fov = well_id.rsplit("-", 1)
        return region, fov
    return well_id, ""


def measurements_header(channel_names) -> list[str]:
    return (
        ["well", "region", "fov", "condition", "cell_id", "time",
         "centroid_x", "centroid_y", "area_px", "area_um2",
         "equiv_diameter_um", "feret_diameter_um"]
        + [f"mean_{n}" for n in channel_names]
        + [f"total_{n}" for n in channel_names]
    )


def measurement_rows(well_id: str, condition: str, wa):
    region, fov = split_region_fov(well_id)
    for m in wa.measurements:
        yield (
            [well_id, region, fov, condition, m.track_id, m.frame + 1,
             round(m.centroid_x, 2), round(m.centroid_y, 2),
             m.area_px, round(m.area_um2, 3),
             round(m.equiv_diameter_um, 3), round(m.feret_diameter_um, 3)]
            + [round(v, 3) for v in m.mean_intensity]
            + [round(v, 3) for v in m.total_intensity]
        )


def write_measurements_csv(path: str, items) -> int:
    """Write a CSV for one or more analyzed wells.

    ``items`` is a list of ``(well_id, condition, WellAnalysis)`` that share a
    channel set. Returns the number of measurement rows written.
    """
    items = [it for it in items if it[2] is not None]
    if not items:
        return 0
    header = measurements_header(items[0][2].channel_names)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for well_id, condition, wa in items:
            for row in measurement_rows(well_id, condition, wa):
                writer.writerow(row)
                n += 1
    return n


# --- fixed "regionprops-style" parquet schema ---------------------------------
# One row per cell per timepoint. Column names, order, and dtypes are fixed so
# downstream analysis built around this layout reads CellScope output unchanged.
# The per-channel intensity block is grouped by statistic, then channel:
#   Intensity Mean (<ch0>), Intensity Mean (<ch1>), Intensity Max (<ch0>), ...
_PARQUET_STATS = (
    ("Mean", "mean_intensity"),
    ("Max", "max_intensity"),
    ("STD", "std_intensity"),
    ("Min", "min_intensity"),
)
_PARQUET_MORPHO = [
    ("Diameter (Equivalent) (um)", "equiv_diameter_um"),
    ("Diameter (Feret) (um)", "feret_diameter_um"),
    ("Length Major (um)", "length_major_um"),
    ("Length Minor (um)", "length_minor_um"),
    ("Perimeter (um)", "perimeter_um"),
]


# The first 18 columns are the fixed "regionprops-style" reference layout (so a
# notebook built on that schema still reads by name). ``fov`` and ``condition``
# are appended because CellScope segments each field of view separately, so
# ``Label`` restarts at 1 per FOV - without ``fov`` the per-cell key
# (Well, Timepoint, Label) collides across the FOVs pooled into a well. ``Well``
# holds the region; the unique per-cell key is (Dataset, Well, fov, Timepoint, Label).
_PARQUET_KEY_TAIL = ["Eccentricity", "Dataset", "Timepoint", "Well", "fov", "condition"]


def parquet_columns(channel_names) -> list[str]:
    """Ordered column names of the parquet schema for a given channel set."""
    cols = ["Label"] + [name for name, _ in _PARQUET_MORPHO]
    for stat, _attr in _PARQUET_STATS:
        cols += [f"Intensity {stat} ({n})" for n in channel_names]
    cols += _PARQUET_KEY_TAIL
    return cols


def _parquet_column_data(items, channel_names, dataset: str) -> "dict[str, list]":
    """Build column-oriented data for the parquet schema from analyzed wells.

    ``items`` is a list of ``(well_id, condition, WellAnalysis)`` (same shape the
    CSV writer takes, so the two schemas can't drift on identity columns).
    """
    cols: dict[str, list] = {c: [] for c in parquet_columns(channel_names)}
    n_chan = len(channel_names)
    for well_id, condition, wa in items:
        region, fov = split_region_fov(well_id)
        for m in wa.measurements:
            cols["Label"].append(int(m.track_id))
            for name, attr in _PARQUET_MORPHO:
                cols[name].append(float(getattr(m, attr)))
            for stat, attr in _PARQUET_STATS:
                values = getattr(m, attr)
                for ci in range(n_chan):
                    v = float(values[ci]) if ci < len(values) else float("nan")
                    cols[f"Intensity {stat} ({channel_names[ci]})"].append(v)
            cols["Eccentricity"].append(float(m.eccentricity))
            cols["Dataset"].append(dataset)
            cols["Timepoint"].append(int(m.frame))  # 0-based, matches the schema
            cols["Well"].append(region)
            cols["fov"].append(fov)
            cols["condition"].append(condition or "")
    return cols


def write_measurements_parquet(path: str, items, dataset: str = "") -> int:
    """Write the fixed-schema parquet for one or more analyzed wells.

    ``items`` is a list of ``(well_id, condition, WellAnalysis)`` sharing a
    channel set. ``dataset`` fills the ``Dataset`` column (e.g. the acquisition
    folder). Returns the number of measurement rows written. Requires ``pyarrow``.
    """
    pa, pq = _require_pyarrow()
    items = [it for it in items if it[2] is not None]
    if not items:
        return 0
    channel_names = items[0][2].channel_names
    data = _parquet_column_data(items, channel_names, dataset)

    int_cols = {"Label", "Timepoint"}
    str_cols = {"Dataset", "Well", "fov", "condition"}
    arrays, names = [], []
    for name in parquet_columns(channel_names):
        values = data[name]
        if name in int_cols:
            arrays.append(pa.array(values, type=pa.int64()))
        elif name in str_cols:
            arrays.append(pa.array(values, type=pa.string()))
        else:
            arrays.append(pa.array(values, type=pa.float64()))
        names.append(name)
    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(table, path)
    return table.num_rows


def combine_parquet(out_path: str, part_paths) -> int:
    """Concatenate per-position parquet files into one. Returns total rows."""
    pa, pq = _require_pyarrow()
    tables = [pq.read_table(p) for p in part_paths if _exists_nonempty(p)]
    if not tables:
        return 0
    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, out_path)
    return combined.num_rows


def _exists_nonempty(path) -> bool:
    import os
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise RuntimeError(
            "Parquet output needs pyarrow. Install it with:\n"
            "    pip install pyarrow      (or: pip install 'cellscope[parquet]')"
        ) from exc
    return pa, pq
