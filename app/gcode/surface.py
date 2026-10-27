"""Closest-point and outward-normal lookup on a head mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from app.gcode.mesh_normals import head_center_from_points, orient_normals_outward


@dataclass
class MeshContext:
    mesh: pv.PolyData
    kdtree: KDTree
    points: np.ndarray
    point_normals: np.ndarray
    head_center: np.ndarray


def prepare_mesh_context(mesh: pv.PolyData) -> MeshContext:
    if not hasattr(mesh, "point_normals") or mesh.point_normals is None:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    points = np.asarray(mesh.points, dtype=np.float64)
    head_center = head_center_from_points(points)
    normals = np.asarray(mesh.point_normals, dtype=np.float64)
    normals = orient_normals_outward(points, normals, head_center)
    return MeshContext(
        mesh=mesh,
        kdtree=KDTree(points),
        points=points,
        point_normals=normals,
        head_center=head_center,
    )


def load_mesh_context(mesh_path: str | Path) -> MeshContext:
    mesh = pv.read(str(mesh_path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    return prepare_mesh_context(mesh)


def normals_at_points(ctx: MeshContext, points_3d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    _, idx = ctx.kdtree.query(pts)
    if np.isscalar(idx):
        idx = [idx]
    normals = ctx.point_normals[np.asarray(idx, dtype=int)]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return orient_normals_outward(pts, normals, ctx.head_center)


def closest_points_on_surface(mesh: pv.PolyData, points_3d: np.ndarray) -> np.ndarray:
    import vtk

    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    surface = mesh.triangulate()
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    closest = np.empty_like(pts)
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    query = [0.0, 0.0, 0.0]
    for i, point in enumerate(pts):
        locator.FindClosestPoint(point, query, cell_id, sub_id, dist2)
        closest[i] = query
    return closest
