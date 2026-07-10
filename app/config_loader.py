"""Load config/defaults.yaml (optional PyYAML)."""
from __future__ import annotations

from typing import Any

from app import paths

_DEFAULTS: dict[str, Any] | None = None


def load_defaults() -> dict[str, Any]:
    global _DEFAULTS
    if _DEFAULTS is not None:
        return _DEFAULTS
    cfg_path = paths.CONFIG_DIR / "defaults.yaml"
    if not cfg_path.exists():
        _DEFAULTS = {}
        return _DEFAULTS
    try:
        import yaml
    except ImportError:
        _DEFAULTS = {}
        return _DEFAULTS
    with open(cfg_path, encoding="utf-8") as f:
        _DEFAULTS = yaml.safe_load(f) or {}
    return _DEFAULTS


def default_assignments() -> str:
    """Canonical terminal-assignment preset name (data/presets/{name}.json)."""
    cfg = load_defaults()
    name = cfg.get("synthesize", {}).get("assignments") or cfg.get("assignments")
    if not name:
        return "s1_assignments"
    return str(name)


def resolve_assignments(name: str | None = None) -> str:
    return name if name else default_assignments()


def preprocess_defaults() -> dict[str, Any]:
    return load_defaults().get("preprocess", {})
