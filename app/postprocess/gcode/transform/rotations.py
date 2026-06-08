"""Rotation matrices matching MATLAB rotx/roty/rotz conventions."""

from __future__ import annotations

import numpy as np


def rotx(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def roty(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotz(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def apply_rotation(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Apply 3x3 rotation to Nx3 points (row vectors, MATLAB style: p @ R.T)."""
    if points.ndim == 1:
        return points @ rot.T
    return points @ rot.T
