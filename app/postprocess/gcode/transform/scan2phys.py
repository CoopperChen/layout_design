"""Scan-to-physical landmark registration."""

from __future__ import annotations

import numpy as np

from app.postprocess.print_config import validate_landmark_triangle

from .plane_frames import plane_coordinates_rotation_matrices


def _safe_unit(v: np.ndarray, *, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"Cannot normalize {name} (norm={n})")
    return v / n


def scan2phys(data: np.ndarray, ps: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """
    Transform points/normals from scan frame to physical frame.

    Matches MATLAB scan2phys.m: fun_scan2meas = @(p) ((p-scan_sys.p0)*scan_sys.T)*meas_sys.Ttilde+meas_sys.p0
    """
    data = np.asarray(data, dtype=float)
    ps = validate_landmark_triangle(ps, label="digital landmarks (ps)")
    pm = validate_landmark_triangle(pm, label="physical_landmarks_mm (pm)")
    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    normal_scan = _safe_unit(
        np.cross(ps[1] - ps[0], ps[2] - ps[0]), name="scan landmark normal"
    )
    vs = _safe_unit(ps[1] - ps[0], name="scan landmark edge p0→p1")

    normal_meas = _safe_unit(
        np.cross(pm[1] - pm[0], pm[2] - pm[0]), name="physical landmark normal"
    )
    vm = _safe_unit(pm[1] - pm[0], name="physical landmark edge p0→p1")

    _, scan_ttilde = plane_coordinates_rotation_matrices(normal_scan, vs)
    meas_t, meas_ttilde = plane_coordinates_rotation_matrices(normal_meas, vm)

    scan_t = np.linalg.inv(scan_ttilde)
    p0_scan = ps[0]
    p0_meas = pm[0]

    rotated = ((data - p0_scan) @ scan_t) @ meas_ttilde + p0_meas
    return rotated[0] if single else rotated


def scan2phys_direction(data: np.ndarray, ps: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """Rotate direction vectors (e.g. surface normals) — no translation."""
    data = np.asarray(data, dtype=float)
    ps = validate_landmark_triangle(ps, label="digital landmarks (ps)")
    pm = validate_landmark_triangle(pm, label="physical_landmarks_mm (pm)")
    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    normal_scan = _safe_unit(
        np.cross(ps[1] - ps[0], ps[2] - ps[0]), name="scan landmark normal"
    )
    vs = _safe_unit(ps[1] - ps[0], name="scan landmark edge p0→p1")

    normal_meas = _safe_unit(
        np.cross(pm[1] - pm[0], pm[2] - pm[0]), name="physical landmark normal"
    )
    vm = _safe_unit(pm[1] - pm[0], name="physical landmark edge p0→p1")

    _, scan_ttilde = plane_coordinates_rotation_matrices(normal_scan, vs)
    _, meas_ttilde = plane_coordinates_rotation_matrices(normal_meas, vm)

    scan_t = np.linalg.inv(scan_ttilde)
    rotated = (data @ scan_t) @ meas_ttilde
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rotated = rotated / norms
    return rotated[0] if single else rotated
