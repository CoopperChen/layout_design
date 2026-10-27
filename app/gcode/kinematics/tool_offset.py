"""Tool length and B/C axis offset compensation."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig


def apply_tool_offset(
    positions: np.ndarray,
    normals: np.ndarray,
    c_angles: np.ndarray,
    machine: MachineConfig,
    *,
    gap_mm: float | None = None,
) -> np.ndarray:
    g = positions.copy()
    gap = machine.gap_size_mm if gap_mm is None else float(gap_mm)
    t = machine.d_mm + gap
    a = machine.a_mm
    npts = g.shape[0]

    for i in range(npts):
        x_offset = normals[i, 0] * t - abs(np.cos(np.deg2rad(c_angles[i]))) * a
        g[i, 0] += x_offset

        y_offset = normals[i, 1] * t + np.sin(np.deg2rad(c_angles[i])) * a
        g[i, 1] += y_offset

        cross_p = np.cross(normals[i], [0, 0, 1])
        if np.linalg.norm(cross_p) < 0.01:
            total_offset = normals[i] * t
        else:
            total_offset = normals[i] * t - cross_p / np.linalg.norm(cross_p) * a
        g[i, 2] += total_offset[2]

    return np.round(g, 2)
