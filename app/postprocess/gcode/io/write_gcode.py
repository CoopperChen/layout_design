"""G-code formatting and file output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def format_gcode_lines(gcode_matrix: np.ndarray) -> list[str]:
    """
    Format internal gcode matrix [X, Y, Z, B, C, F, marker] to G-code text lines.

    marker: 0=none, 10=M10 jet on, 11=M11 jet off
    """
    lines: list[str] = []
    for i in range(gcode_matrix.shape[0]):
        row = gcode_matrix[i]
        x, y, z, b, c, f = row[0], row[1], row[2], row[3], row[4], row[5]
        marker = int(row[6]) if row.shape[0] > 6 else 0

        def _fmt(v: float) -> str:
            if np.isnan(v):
                return "0"
            if v == int(v):
                return str(int(v))
            return str(v)

        if i == 0:
            line = f"G94 G1 X{_fmt(x)} Y{_fmt(y)} Z{_fmt(z)} B{_fmt(b)} C{_fmt(c)} F{_fmt(f)}"
        else:
            line = f"X{_fmt(x)} Y{_fmt(y)} Z{_fmt(z)} B{_fmt(b)} C{_fmt(c)} F{_fmt(f)}"
            if marker in (10, 11):
                line += f" M{marker}"
        lines.append(line)
    return lines


def write_gcode_file(path: Path | str, gcode_matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = format_gcode_lines(gcode_matrix)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
