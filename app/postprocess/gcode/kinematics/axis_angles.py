"""B/C axis angle computation from surface normals."""

from __future__ import annotations

import numpy as np


def find_angle(comp_v: np.ndarray, proj_v: np.ndarray) -> float:
    comp_v = np.asarray(comp_v, dtype=float)
    proj_v = np.asarray(proj_v, dtype=float)
    denom = np.linalg.norm(comp_v) * np.linalg.norm(proj_v)
    if denom == 0:
        return float("nan")
    return float(np.rad2deg(np.arccos(np.dot(proj_v, comp_v) / denom)))


def find_caxis_angle(norm_v: np.ndarray, i: int) -> float:
    c_vc = np.array([1.0, 0.0, 0.0])
    p_vc = np.array([norm_v[i, 0], norm_v[i, 1], 0.0])
    sign_angle = np.cross(p_vc, c_vc)
    if sign_angle[2] == 0:
        return float("nan")
    return float(np.sign(sign_angle[2]) * (-90 + find_angle(c_vc, p_vc)))


def find_baxis_angle(norm_v: np.ndarray, i: int) -> float:
    c_vb = norm_v[i, :]
    p_vb = np.array([norm_v[i, 0], norm_v[i, 1], 0.0])
    if c_vb[2] >= 0:
        return float(np.sign(norm_v[i, 1]) * (find_angle(-c_vb, p_vb) - 90))
    return float(np.sign(norm_v[i, 1]) * (270 - find_angle(-c_vb, p_vb)))


def _fill_nan_angles(angles: np.ndarray) -> np.ndarray:
    """Replace indefinite angles with neighbor average (MATLAB fallback)."""
    out = angles.copy()
    npts = len(out)
    for i in range(npts):
        if not np.isnan(out[i]):
            continue
        neighbors = []
        if i > 0 and not np.isnan(out[i - 1]):
            neighbors.append(out[i - 1])
        if i < npts - 1 and not np.isnan(out[i + 1]):
            neighbors.append(out[i + 1])
        out[i] = float(np.mean(neighbors)) if neighbors else 0.0
    return out


def compute_axis_angles(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    npts = normals.shape[0]
    c_angles = np.zeros(npts, dtype=float)
    b_angles = np.zeros(npts, dtype=float)
    for i in range(npts):
        c_angles[i] = find_caxis_angle(normals, i)
        b_angles[i] = find_baxis_angle(normals, i)
    c_angles = _fill_nan_angles(c_angles)
    b_angles = _fill_nan_angles(b_angles)
    return b_angles, c_angles
