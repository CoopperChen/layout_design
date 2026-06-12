"""G-code parser tests."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.simulator.parser import parse_gcode_file, parse_gcode_text

SAMPLE = """G94 G1 X-58.5 Y32.38 Z27 B77.21 C33.4 F1000
X-67.74 Y30.85 Z-125.28 B77.77 C31.59 F818
X-67.74 Y30.85 Z-125.28 B77.77 C31.59 F818 M10
X-73.17 Y31.85 Z-120.22 B76.24 C31.31 F646 M11
"""


def test_parse_gcode_text_axes_and_markers():
    rows = parse_gcode_text(SAMPLE)
    assert rows.shape == (4, 7)
    assert rows[0, 0] == -58.5
    assert rows[0, 3] == 77.21
    assert rows[2, 6] == 10.0
    assert rows[3, 6] == 11.0


def test_parse_inherits_modal_axes():
    text = "G1 X1 Y2 Z3 B4 C5 F100\nX10\n"
    rows = parse_gcode_text(text)
    assert rows.shape == (2, 7)
    assert rows[1, 0] == 10.0
    assert rows[1, 1] == 2.0
    assert rows[1, 3] == 4.0


def test_parse_strips_comments():
    text = "G1 X1 (comment) Y2 ; tail\n"
    rows = parse_gcode_text(text)
    assert rows[0, 0] == 1.0
    assert rows[0, 1] == 2.0


def test_parse_subject_4_gcode_header():
    gcode_path = (
        paths.REPO_ROOT
        / "data/output/gcode/subject_4_post/allinterconnects.txt"
    )
    if not gcode_path.is_file():
        pytest.skip("subject_4 gcode not present")
    rows = parse_gcode_file(gcode_path)
    assert rows.shape[0] > 100
    assert np.all(np.isfinite(rows[:, :6]))
