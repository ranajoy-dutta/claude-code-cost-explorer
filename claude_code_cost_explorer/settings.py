import os
import json

CLAUDE_DIR = os.path.expanduser(os.environ.get("CLAUDE_DIR", "~/.claude"))
_SETTINGS_PATH = os.environ.get(
    "CCX_SETTINGS_PATH", os.path.join(CLAUDE_DIR, "ccx_settings.json")
)
_DEFAULT_THRESHOLDS = {"low": 1.0, "medium": 5.0, "high": 15.0}

_settings_cache: dict | None = None
_settings_cache_mtime: int | None = None


def _load_settings() -> dict:
    global _settings_cache, _settings_cache_mtime
    try:
        mtime = os.stat(_SETTINGS_PATH).st_mtime_ns
    except OSError:
        mtime = None
    if _settings_cache is not None and mtime == _settings_cache_mtime:
        return _settings_cache
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        t = data.get("cost_thresholds", {})
        result = {
            "low": float(t["low"]),
            "medium": float(t["medium"]),
            "high": float(t["high"]),
        }
    except Exception:
        result = dict(_DEFAULT_THRESHOLDS)
        mtime = None
    _settings_cache = result
    _settings_cache_mtime = mtime
    return result


def _save_settings(low: float, medium: float, high: float) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    data = {}
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        pass
    data["cost_thresholds"] = {"low": low, "medium": medium, "high": high}
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
