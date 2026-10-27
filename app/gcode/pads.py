"""Coplanar electrode disk, built from a surface point at G-code time."""

from __future__ import annotations

import numpy as np

from app.config_loader import load_defaults
from app.gcode.surface import (
    MeshContext,
    closest_points_on_surface,
    normals_at_points,
)

ELECTRODE_AREA_CM2 = 0.5
DEFAULT_NLINES = 10
_COPLANAR_TOL_MM = 1e-6


def electrode_pad_settings() -> tuple[float, int]:
    """Pad area (cm²) and zigzag row count from config/defaults.yaml."""
    section = load_defaults().get("electrode") or {}
    area_cm2 = float(section.get("area_cm2", ELECTRODE_AREA_CM2))
    nlines = int(section.get("nlines", DEFAULT_NLINES))
    if area_cm2 <= 0.0:
        raise ValueError(f"electrode.area_cm2 must be positive, got {area_cm2}")
    if nlines < 1:
        raise ValueError(f"electrode.nlines must be >= 1, got {nlines}")
    return area_cm2, nlines


def diameter_mm_from_area_cm2(area_cm2: float = ELECTRODE_AREA_CM2) -> float:
    return float(2.0 * np.sqrt(area_cm2 / np.pi) * 10.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("Zero-length vector.")
    return v / norm


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def export_electrode_xyzn(electrode_xyz: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    unit = _unit(plane_normal)
    xyz = np.asarray(electrode_xyz, dtype=np.float64)
    return np.column_stack([xyz, np.tile(unit, (len(xyz), 1))])


def build_electrode_disk(
    center: np.ndarray,
    mesh,
    ctx: MeshContext,
    diameter_mm: float,
    gap_size_mm: float,
    *,
    nlines: int = DEFAULT_NLINES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Disk zigzag on a plane ``gap_size_mm`` above the surface at ``center``."""
    hint = np.asarray(center, dtype=np.float64).reshape(3)
    surface = closest_points_on_surface(mesh, hint.reshape(1, 3))[0]
    normal = _unit(normals_at_points(ctx, surface.reshape(1, 3))[0])
    plane_origin = surface + float(gap_size_mm) * normal

    t1, t2, _ = tangent_basis(normal)
    uv = perimeter_zigzag_uv(diameter_mm / 2.0, nlines)
    xyz = uv_to_global(uv, plane_origin, t1, t2)

    residual = coplanar_residual_mm(xyz, plane_origin, normal)
    if residual > _COPLANAR_TOL_MM:
        raise RuntimeError(f"Electrode disk not coplanar (residual {residual:.2e} mm).")
    height = float((plane_origin - surface) @ normal)
    if abs(height - float(gap_size_mm)) > 1e-4:
        raise RuntimeError(f"Plane height {height:.4f} mm != gap {gap_size_mm:.4f} mm.")
    return xyz, plane_origin, surface, normal
