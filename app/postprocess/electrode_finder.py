"""
Planar electrode disk: interconnect row 0 → surface normal → disk at gap above STL.

Perimeter points on a circle boundary, connected in zigzag order (alternating ±u per row).
All points are coplanar; constant outward normal on export (AdjPoints repmat).
"""

from __future__ import annotations

import numpy as np

DEFAULT_ELECTRODE_AREA_CM2 = 1.5
DEFAULT_NLINES = 10
_COPLANAR_TOL_MM = 1e-6


def diameter_mm_from_area_cm2(area_cm2: float = DEFAULT_ELECTRODE_AREA_CM2) -> float:
    return float(2.0 * np.sqrt(area_cm2 / np.pi) * 10.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("Zero-length vector.")
    return v / norm


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (t1, t2, n) with n outward."""
    n = _unit(normal)
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(n[2]) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    t1 = np.cross(ref, n)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2, n


def perimeter_zigzag_uv(radius_mm: float, nlines: int) -> np.ndarray:
    """
    Disk boundary zigzag in local (u, v) coords.

    For each row k = 0..nlines: v = -R + k*(2R/nlines), u = ±sqrt(R²-v²), alternating sign.
    """
    radius_mm = float(radius_mm)
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be positive.")
    if nlines < 1:
        raise ValueError("nlines must be >= 1.")

    uv: list[list[float]] = []
    step = 2.0 * radius_mm / nlines
    for row in range(nlines + 1):
        v = -radius_mm + row * step
        u_mag = float(np.sqrt(max(0.0, radius_mm**2 - v**2)))
        u = u_mag if row % 2 == 0 else -u_mag
        uv.append([u, v])
    return np.asarray(uv, dtype=np.float64)


def uv_to_global(
    uv: np.ndarray,
    plane_origin: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    return origin + uv[:, 0:1] * t1 + uv[:, 1:2] * t2


def coplanar_residual_mm(
    points: np.ndarray,
    plane_origin: np.ndarray,
    normal: np.ndarray,
) -> float:
    n = _unit(normal)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    return float(np.max(np.abs((np.asarray(points, dtype=np.float64) - origin) @ n)))


def build_electrode_disk_zigzag(
    interconnect_trace: np.ndarray,
    mesh,
    ctx,
    diameter_mm: float,
    gap_size_mm: float,
    *,
    nlines: int = DEFAULT_NLINES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build coplanar electrode zigzag on a disk plane ``gap_size_mm`` above the STL.

    Returns (xyz, plane_origin, surface_point, outward_normal).
    """
    from app.postprocess.mesh_export import closest_points_on_surface, normals_at_points
    from app.postprocess.mesh_normals import orient_normals_outward

    trace = np.asarray(interconnect_trace, dtype=np.float64)
    hint = trace[0, :3]
    surface = closest_points_on_surface(mesh, hint.reshape(1, 3))[0]
    normal = _unit(normals_at_points(ctx, surface.reshape(1, 3))[0])
    normal = _unit(
        orient_normals_outward(
            surface.reshape(1, 3),
            normal.reshape(1, 3),
            ctx.head_center,
        )[0]
    )
    ic_normal = trace[0, 3:6]
    ic_norm = np.linalg.norm(ic_normal)
    if ic_norm > 0.0 and float(np.dot(normal, ic_normal / ic_norm)) < 0.0:
        normal = -normal
    plane_origin = surface + float(gap_size_mm) * normal

    t1, t2, _ = tangent_basis(normal)
    radius_mm = diameter_mm / 2.0
    uv = perimeter_zigzag_uv(radius_mm, nlines)
    xyz = uv_to_global(uv, plane_origin, t1, t2)

    residual = coplanar_residual_mm(xyz, plane_origin, normal)
    if residual > _COPLANAR_TOL_MM:
        raise RuntimeError(f"Electrode disk not coplanar (residual {residual:.2e} mm).")

    height = float((plane_origin - surface) @ normal)
    if abs(height - float(gap_size_mm)) > 1e-4:
        raise RuntimeError(
            f"Plane height {height:.4f} mm != gap {gap_size_mm:.4f} mm."
        )

    return xyz, plane_origin, surface, normal


def export_electrode_xyzn(
    electrode_xyz: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """AdjPoints.m export: constant outward normal on every pad point."""
    unit = _unit(plane_normal)
    xyz = np.asarray(electrode_xyz, dtype=np.float64)
    return np.column_stack([xyz, np.tile(unit, (len(xyz), 1))])
