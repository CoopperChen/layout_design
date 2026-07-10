"""G-code formatting and file output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GCODE_DECIMALS = 2


def _fmt_gcode_value(v: float) -> str:
    if np.isnan(v):
        return "0"
    rounded = round(float(v), _GCODE_DECIMALS)
    if rounded == int(rounded):
        return str(int(rounded))
    text = f"{rounded:.{_GCODE_DECIMALS}f}"
    return text.rstrip("0").rstrip(".")


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

        if i == 0:
            line = (
                f"G94 G1 X{_fmt_gcode_value(x)} Y{_fmt_gcode_value(y)} Z{_fmt_gcode_value(z)} "
                f"B{_fmt_gcode_value(b)} C{_fmt_gcode_value(c)} F{_fmt_gcode_value(f)}"
            )
        else:
            line = (
                f"X{_fmt_gcode_value(x)} Y{_fmt_gcode_value(y)} Z{_fmt_gcode_value(z)} "
                f"B{_fmt_gcode_value(b)} C{_fmt_gcode_value(c)} F{_fmt_gcode_value(f)}"
            )
            if marker in (10, 11):
                line += f" M{marker}"
        lines.append(line)
    return lines


def write_gcode_file(path: Path | str, gcode_matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = format_gcode_lines(gcode_matrix)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
