"""Pre-export validation gates."""

from pathlib import Path

import pytest

from app.postprocess.validate_export import ExportValidationError, validate_smooth_for_export
from tests.fixtures.bundle_factory import write_synthetic_smooth


def test_validate_accepts_collision_free_smooth(tmp_path: Path):
    smooth = tmp_path / "smooth.json"
    write_synthetic_smooth(smooth)
    data = __import__("json").loads(smooth.read_text(encoding="utf-8"))
    validate_smooth_for_export(data, smooth_path=smooth)


def test_validate_rejects_short_path(tmp_path: Path):
    smooth = tmp_path / "smooth.json"
    write_synthetic_smooth(smooth)
    data = __import__("json").loads(smooth.read_text(encoding="utf-8"))
    data["final_paths"][0]["path_3d"] = [[1, 2, 3]]
    with pytest.raises(ExportValidationError, match="at least 2"):
        validate_smooth_for_export(data, smooth_path=smooth)


def test_validate_rejects_collision(tmp_path: Path):
    smooth = tmp_path / "smooth.json"
    write_synthetic_smooth(smooth)
    data = __import__("json").loads(smooth.read_text(encoding="utf-8"))
    data["collision_metrics"]["layout_collision_free"] = False
    data["collision_metrics"]["crossing_count"] = 2
    with pytest.raises(ExportValidationError, match="not collision-free"):
        validate_smooth_for_export(data, smooth_path=smooth)
