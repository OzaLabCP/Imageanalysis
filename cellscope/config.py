"""Tiny local-only settings store.

Persists a handful of preferences as JSON under the user's home directory.
No network, no database. Missing or corrupt files degrade gracefully to defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".cellscope"
CONFIG_PATH = CONFIG_DIR / "config.json"

_DEFAULTS: dict = {
    "theme": "system",        # "system" | "light" | "dark"
    "first_run_done": False,
    "last_export_dir": "",
}


def load_config() -> dict:
    data = dict(_DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
    except Exception:
        # Corrupt config should never block launch.
        pass
    return data


def save_config(data: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def get(key: str, default=None):
    return load_config().get(key, default)


def set_value(key: str, value) -> None:
    data = load_config()
    data[key] = value
    save_config(data)
