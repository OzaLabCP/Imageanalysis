"""Run-level provenance for reproducible analysis.

Records exactly HOW a set of results was produced - the analysis settings,
segmentation engine, resolved GPU device, pixel size, software version, source,
and channel set - as a small JSON sidecar written next to the results. Two runs
can be proven identical (or diffed) by comparing these files, which is the part
of reproducibility that the measurements themselves cannot capture.

No Qt / heavy dependencies, so it is safe to import in headless / cluster runs.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass

from cellscope import __version__

RUN_METADATA_NAME = "run_metadata.json"


def _git_revision() -> str | None:
    """Best-effort git commit (short SHA, with ``+dirty``) of the installed code.

    ``__version__`` alone can't tell two builds apart - both branches report
    0.1.0 despite different output schemas - so record the commit when the code
    is a checkout (e.g. an editable install)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        if not sha:
            return None
        dirty = subprocess.call(
            ["git", "-C", root, "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def build_run_metadata(
    *,
    source: str,
    engine: str,
    settings,
    pixel_size_um,
    downsample: int,
    channel_names,
    gpu: dict | None,
    positions,
    output_format: str = "csv",
    created_utc: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the provenance record for a run.

    ``settings`` may be an ``AnalysisSettings`` dataclass (or any dataclass);
    it is serialized field-by-field. ``gpu`` is the engine's resolved-device
    dict (see ``engines.resolve_device``). ``positions`` may be a count or a
    list of position ids. ``created_utc`` is caller-supplied so the record is
    deterministic in tests; callers pass an ISO-8601 UTC timestamp.
    """
    settings_dict = asdict(settings) if is_dataclass(settings) else dict(settings or {})
    if isinstance(positions, int):
        n_positions, position_ids = positions, None
    else:
        position_ids = list(positions)
        n_positions = len(position_ids)

    meta = {
        "cellscope_version": __version__,
        "cellscope_git": _git_revision(),
        "created_utc": created_utc,
        "source": str(source),
        "output_format": output_format,
        "engine": engine,
        "gpu": gpu,
        "pixel_size_um": (float(pixel_size_um) if pixel_size_um is not None else None),
        "downsample": int(downsample),
        "channel_names": list(channel_names),
        "n_positions": n_positions,
        "settings": settings_dict,
    }
    if position_ids is not None:
        meta["positions"] = position_ids
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(path: str, meta: dict) -> None:
    """Write a provenance record to ``path`` as pretty JSON (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")
