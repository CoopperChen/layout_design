"""Pre-export validation gates for smooth JSON → bundle / .mat."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import paths


class ExportValidationError(ValueError):
    """Raised when smooth JSON fails export readiness checks."""


def _load_collision_metrics(smooth_data: dict, smooth_path: Path) -> dict | None:
    if "collision_metrics" in smooth_data:
        return smooth_data["collision_metrics"]

    source = smooth_data.get("source_applied")
    if not source:
        return None

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = paths.REPO_ROOT / source
    if not source_path.is_file():
        return None

    layout = json.loads(source_path.read_text(encoding="utf-8"))
    return layout.get("collision_metrics")


def validate_smooth_for_export(
    smooth_data: dict,
    *,
    smooth_path: Path | None = None,
    require_collision_free: bool = True,
) -> None:
    """
    Fail fast before expensive mesh export.

    Checks:
    - final_paths present and non-empty
    - each interconnect has >= 2 points with finite coordinates
    - collision_metrics.layout_collision_free when available
    """
    final_paths = smooth_data.get("final_paths")
    if not final_paths:
        raise ExportValidationError("smooth JSON has no final_paths")

    for entry in final_paths:
        name = entry.get("electrode", "?")
        path_3d = entry.get("path_3d")
        if not path_3d or len(path_3d) < 2:
            raise ExportValidationError(
                f"interconnect {name!r} needs at least 2 path_3d points"
            )
        arr = np.asarray(path_3d, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ExportValidationError(
                f"interconnect {name!r} has non-finite path_3d coordinates"
            )

    if not require_collision_free:
        return

    cm = _load_collision_metrics(smooth_data, smooth_path or Path("."))
    if cm is None:
        raise ExportValidationError(
            "collision_metrics not found in smooth JSON or source_applied layout; "
            "re-run smooth from a synthesized layout or pass --skip-validation"
        )
    if not cm.get("layout_collision_free", False):
        crossings = cm.get("crossing_count", "?")
        violations = cm.get("electrode_violations", "?")
        raise ExportValidationError(
            f"layout is not collision-free "
            f"(crossing_count={crossings}, electrode_violations={violations}); "
            f"polish or re-synthesize before export"
        )


def validate_smooth_file(
    smooth_json: str | Path,
    *,
    require_collision_free: bool = True,
) -> dict:
    smooth_path = paths.resolve_json_path(smooth_json, role="Smooth JSON")
    data = json.loads(smooth_path.read_text(encoding="utf-8"))
    validate_smooth_for_export(
        data,
        smooth_path=smooth_path,
        require_collision_free=require_collision_free,
    )
    return data
