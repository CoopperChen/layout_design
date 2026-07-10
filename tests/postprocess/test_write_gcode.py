"""G-code text formatting."""

from __future__ import annotations

import numpy as np

from app.postprocess.gcode.io.write_gcode import format_gcode_lines


def test_format_gcode_rounds_offset_legs_to_two_decimals():
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 5.0, 1500.0, 0.0],
            [-276.8075520755918, -180.65290730895032, -220.5, 106.84, -39.17, 750.0, 0.0],
            [-270.41635709598893, -203.25388395945447, 18.0, 80.94, -48.13, 1500.0, 10.0],
        ]
    )
    lines = format_gcode_lines(rows)
    assert lines[1] == "X-276.81 Y-180.65 Z-220.5 B106.84 C-39.17 F750"
    assert lines[2] == "X-270.42 Y-203.25 Z18 B80.94 C-48.13 F1500 M10"


def test_format_gcode_strips_trailing_zeros():
    rows = np.array([[10.0, -20.0, 18.0, 84.8, 5.0, 1500.0, 0.0]])
    lines = format_gcode_lines(rows)
    assert lines[0] == "G94 G1 X10 Y-20 Z18 B84.8 C5 F1500"
