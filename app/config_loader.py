"""Load config/defaults.yaml (electrode pads and motion tuning)."""

from __future__ import annotations

from typing import Any

from app import paths

_DEFAULTS: dict[str, Any] | None = None


def load_defaults() -> dict[str, Any]:
    global _DEFAULTS
    if _DEFAULTS is not None:
        return _DEFAULTS
    cfg_path = paths.defaults_yaml()
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


def reset_defaults_cache() -> None:
    global _DEFAULTS
    _DEFAULTS = None


def _section(name: str) -> dict[str, Any]:
    raw = load_defaults().get(name) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config/defaults.yaml {name}: must be a mapping")
    return raw


def _number(section: dict[str, Any], key: str, default: float | int, cast, *, low, high=None):
    value = cast(section.get(key, default))
    if value < low or (high is not None and value > high):
        limit = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise ValueError(f"{key} must be {limit}, got {value}")
    return value


def layout_config() -> dict[str, float]:
    """10–20 size slider from config/defaults.yaml."""
    section = _section("layout")
    scale_min = _number(section, "scale_min", 0.5, float, low=0.01)
    scale_max = _number(section, "scale_max", 1.5, float, low=scale_min)
    scale = _number(section, "scale", 0.8, float, low=scale_min, high=scale_max)
    return {"scale": scale, "scale_min": scale_min, "scale_max": scale_max}


def mesh_config() -> dict[str, float | int]:
    """Poisson, island removal, and ICP limits from config/defaults.yaml."""
    section = _section("mesh")
    return {
        "scale_to_mm": _number(section, "scale_to_mm", 1000.0, float, low=1e-9),
        "poisson_depth": _number(section, "poisson_depth", 12, int, low=1),
        "poisson_density_quantile": _number(
            section, "poisson_density_quantile", 0.02, float, low=0.0, high=0.45
        ),
        "outlier_neighbors": _number(section, "outlier_neighbors", 40, int, low=1),
        "outlier_std_ratio": _number(section, "outlier_std_ratio", 1.0, float, low=1e-6),
        "island_min_size_ratio": _number(section, "island_min_size_ratio", 0.02, float, low=0.0),
        "island_min_cells": _number(section, "island_min_cells", 200, int, low=1),
        "icp_samples": _number(section, "icp_samples", 50000, int, low=1000),
        "icp_max_correspondence_mm": _number(
            section, "icp_max_correspondence_mm", 15.0, float, low=1e-6
        ),
        "icp_max_iteration": _number(section, "icp_max_iteration", 80, int, low=1),
        "icp_fitness_min": _number(section, "icp_fitness_min", 0.3, float, low=0.0, high=1.0),
        "icp_mean_dist_max_mm": _number(section, "icp_mean_dist_max_mm", 3.0, float, low=1e-6),
    }
