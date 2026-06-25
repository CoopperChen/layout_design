"""Parse 5-axis G-code text into internal row matrix."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_AXIS_RE = re.compile(r"([XYZBCF])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
_M_RE = re.compile(r"\bM(\d+)\b", re.IGNORECASE)
_PAREN_COMMENT_RE = re.compile(r"\([^)]*\)")
_SEMICOLON_COMMENT_RE = re.compile(r";.*$")


def _strip_comments(line: str) -> str:
    line = _PAREN_COMMENT_RE.sub("", line)
    return _SEMICOLON_COMMENT_RE.sub("", line).strip()


def _parse_line(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in _AXIS_RE.finditer(line):
        out[match.group(1).upper()] = float(match.group(2))
    for match in _M_RE.finditer(line):
        out[f"M{match.group(1)}"] = float(match.group(1))
    return out


def parse_gcode_text(text: str) -> np.ndarray:
    """
    Parse G-code lines into Nx7 array [X, Y, Z, B, C, F, marker].

    marker: 0=none, 10=M10 jet on, 11=M11 jet off.
    Partial lines inherit missing XYZBCF from the previous motion row.
    """
    rows: list[list[float]] = []
    last: dict[str, float] = {"X": 0.0, "Y": 0.0, "Z": 0.0, "B": 0.0, "C": 0.0, "F": 0.0}

    for raw in text.splitlines():
        line = _strip_comments(raw)
        if not line:
            continue
        if line.upper().startswith("G") and "X" not in line.upper():
            continue

        vals = _parse_line(line)
        if not any(k in vals for k in ("X", "Y", "Z", "B", "C")):
            continue

        marker = 0.0
        if "M10" in vals:
            marker = 10.0
        elif "M11" in vals:
            marker = 11.0

        row_vals = {key: vals.get(key, last[key]) for key in ("X", "Y", "Z", "B", "C", "F")}
        last = row_vals
        rows.append(
            [
                row_vals["X"],
                row_vals["Y"],
                row_vals["Z"],
                row_vals["B"],
                row_vals["C"],
                row_vals["F"],
                marker,
            ]
        )

    if not rows:
        raise ValueError("No motion lines found in G-code text")
    return np.asarray(rows, dtype=float)


def parse_gcode_file(path: Path | str) -> np.ndarray:
    path = Path(path)
    return parse_gcode_text(path.read_text(encoding="utf-8"))
