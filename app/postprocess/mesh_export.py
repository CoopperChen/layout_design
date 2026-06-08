"""Shared mesh prep and batched normal lookup for bundle / .mat export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree


@dataclass
class MeshExportContext:
    mesh: pv.PolyData
    kdtree: KDTree
    points: np.ndarray
    point_normals: np.ndarray


def prepare_mesh_export_context(mesh: pv.PolyData) -> MeshExportContext:
    if not hasattr(mesh, "point_normals") or mesh.point_normals is None:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    points = np.asarray(mesh.points, dtype=np.float64)
    normals = np.asarray(mesh.point_normals, dtype=np.float64)
    return MeshExportContext(
        mesh=mesh,
        kdtree=KDTree(points),
        points=points,
        point_normals=normals,
    )


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
    return normals / norms


def xyzn_from_path(ctx: MeshExportContext, path_3d: np.ndarray) -> np.ndarray:
    path = np.asarray(path_3d, dtype=np.float64)
    normals = normals_at_points(ctx, path)
    return np.column_stack([path, normals])
