"""Shared mesh prep and batched normal lookup for bundle / .mat export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from app.postprocess.mesh_normals import (
    head_center_from_points,
    orient_normals_outward,
)


@dataclass
class MeshExportContext:
    mesh: pv.PolyData
    kdtree: KDTree
    points: np.ndarray
    point_normals: np.ndarray
    head_center: np.ndarray


_mesh_context_cache: dict[str, MeshExportContext] = {}


def clear_mesh_context_cache() -> None:
    _mesh_context_cache.clear()


def prepare_mesh_export_context(mesh: pv.PolyData) -> MeshExportContext:
    if not hasattr(mesh, "point_normals") or mesh.point_normals is None:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    points = np.asarray(mesh.points, dtype=np.float64)
    head_center = head_center_from_points(points)
    normals = np.asarray(mesh.point_normals, dtype=np.float64)
    normals = orient_normals_outward(points, normals, head_center)
    return MeshExportContext(
        mesh=mesh,
        kdtree=KDTree(points),
        points=points,
        point_normals=normals,
        head_center=head_center,
    )


def load_mesh_context(
    mesh_path: str | Path,
    *,
    use_cache: bool = True,
) -> MeshExportContext:
    """Load mesh from disk with optional in-process cache (keyed by resolved path)."""
    path = Path(mesh_path).resolve()
    key = str(path)
    if use_cache and key in _mesh_context_cache:
        return _mesh_context_cache[key]
    mesh = pv.read(str(path))
    ctx = prepare_mesh_export_context(mesh)
    if use_cache:
        _mesh_context_cache[key] = ctx
    return ctx


def normals_at_points(ctx: MeshExportContext, points_3d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    _, idx = ctx.kdtree.query(pts)
    if np.isscalar(idx):
        idx = [idx]
    normals = ctx.point_normals[np.asarray(idx, dtype=int)]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms
    return orient_normals_outward(pts, normals, ctx.head_center)


def xyzn_from_path(ctx: MeshExportContext, path_3d: np.ndarray) -> np.ndarray:
    path = np.asarray(path_3d, dtype=np.float64)
    normals = normals_at_points(ctx, path)
    return np.column_stack([path, normals])
