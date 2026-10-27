"""Physical landmark YAML."""

from pathlib import Path

import numpy as np
import pytest

from app.subject import (
    load_physical_landmarks,
    pm_is_measured,
    save_physical_landmarks,
    validate_landmark_triangle,
)


def test_save_and_load_roundtrip(tmp_path: Path):
    pm = np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [10.0, 70.0, 5.0]])
    path = tmp_path / "pm.yaml"
    save_physical_landmarks(path, pm, subject_id=3)
    loaded = load_physical_landmarks(path)
    np.testing.assert_allclose(loaded, pm)
    assert pm_is_measured(path)


def test_all_zero_triangle_rejected():
    with pytest.raises(ValueError, match="zeros"):
        validate_landmark_triangle(np.zeros((3, 3)))


def test_unmeasured_file_is_not_measured(tmp_path: Path):
    path = tmp_path / "pm.yaml"
    save_physical_landmarks(path, np.zeros((3, 3)), subject_id=1)
    assert pm_is_measured(path) is False
