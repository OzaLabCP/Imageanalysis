"""GUI-free CSV export of per-cell measurements.

Shared by the desktop Results tab and the headless batch runner, so both write
exactly the same tidy long-format table (one row per cell per timepoint). Has no
Qt dependency, so it is safe to import in headless / cluster contexts.
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
