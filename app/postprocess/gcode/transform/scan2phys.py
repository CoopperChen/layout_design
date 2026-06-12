"""Scan-to-physical landmark registration."""

from __future__ import annotations

import numpy as np

from .plane_frames import plane_coordinates_rotation_matrices


def scan2phys(data: np.ndarray, ps: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """
    Transform points/normals from scan frame to physical frame.

    Matches MATLAB scan2phys.m: fun_scan2meas = @(p) ((p-scan_sys.p0)*scan_sys.T)*meas_sys.Ttilde+meas_sys.p0
    """
    data = np.asarray(data, dtype=float)
    ps = np.asarray(ps, dtype=float)
    pm = np.asarray(pm, dtype=float)
    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    normal_scan = np.cross(ps[1] - ps[0], ps[2] - ps[0])
    normal_scan = normal_scan / np.linalg.norm(normal_scan)
    vs = ps[1] - ps[0]
    vs = vs / np.linalg.norm(vs)

    normal_meas = np.cross(pm[1] - pm[0], pm[2] - pm[0])
    normal_meas = normal_meas / np.linalg.norm(normal_meas)
    vm = pm[1] - pm[0]
    vm = vm / np.linalg.norm(vm)

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
    ps = np.asarray(ps, dtype=float)
    pm = np.asarray(pm, dtype=float)
    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    normal_scan = np.cross(ps[1] - ps[0], ps[2] - ps[0])
    normal_scan = normal_scan / np.linalg.norm(normal_scan)
    vs = ps[1] - ps[0]
    vs = vs / np.linalg.norm(vs)

    normal_meas = np.cross(pm[1] - pm[0], pm[2] - pm[0])
    normal_meas = normal_meas / np.linalg.norm(normal_meas)
    vm = pm[1] - pm[0]
    vm = vm / np.linalg.norm(vm)

    _, scan_ttilde = plane_coordinates_rotation_matrices(normal_scan, vs)
    _, meas_ttilde = plane_coordinates_rotation_matrices(normal_meas, vm)

    scan_t = np.linalg.inv(scan_ttilde)
    rotated = (data @ scan_t) @ meas_ttilde
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rotated = rotated / norms
    return rotated[0] if single else rotated
