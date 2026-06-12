"""Outward-oriented surface normals for head meshes and traces."""

from __future__ import annotations

import numpy as np


def head_center_from_points(points: np.ndarray) -> np.ndarray:
    """Approximate head interior reference from vertex/path cloud."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return np.mean(pts, axis=0)


def orient_normals_outward(
    points: np.ndarray,
    normals: np.ndarray,
    head_center: np.ndarray,
) -> np.ndarray:
    """
    Flip normals so they point away from ``head_center``.

    For each point, the normal should have a positive dot product with
    ``point - head_center`` (radial outward on a convex head-like surface).
    """
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if nrm.ndim == 1:
        nrm = nrm.reshape(1, 3)
    center = np.asarray(head_center, dtype=float).reshape(3)
    out = nrm.copy()
    radial = pts - center
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm = np.maximum(radial_norm, 1e-12)
    radial_unit = radial / radial_norm
    inward = np.sum(out * radial_unit, axis=1) < 0.0
    out[inward] *= -1.0
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    return out / lengths


def orient_trace_xyzn(trace: np.ndarray, head_center: np.ndarray) -> np.ndarray:
    """Orient columns 3:6 of an Nx6 trace to point outward from ``head_center``."""
    out = np.asarray(trace, dtype=float).copy()
    out[:, 3:6] = orient_normals_outward(out[:, :3], out[:, 3:6], head_center)
    return out
