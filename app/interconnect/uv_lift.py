"""Lift a flat wire layout onto the scalp the way synthesize did on master.

Each 2D sample is interpolated from the nearest polar-grid vertices, then snapped
to the cleaned head. Endpoints stay on the electrode and the strip entry.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from app.interconnect.design import WireLayout, polar_projection

UV_RESOLUTION = 100


def create_uv_grid(
    points_3d: np.ndarray,
    cz: np.ndarray,
    *,
    resolution: int = UV_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular polar grid. Each cell stores the mesh vertex with the nearest flat coordinate."""
    points_3d = np.asarray(points_3d, dtype=float)
    points_2d = polar_projection(points_3d, cz)
    low = points_2d.min(axis=0)
    high = points_2d.max(axis=0)
    xs = np.linspace(low[0], high[0], int(resolution))
    ys = np.linspace(low[1], high[1], int(resolution))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    grid_2d = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    _, index = cKDTree(points_2d).query(grid_2d)
    return grid_2d, points_3d[np.asarray(index, dtype=int)]


def _nearest(tree: cKDTree, point: np.ndarray) -> np.ndarray:
    _, index = tree.query(np.asarray(point, dtype=float))
    return np.asarray(tree.data[int(index)], dtype=float)


def entry_on_surface(
    entry_2d: np.ndarray,
    terminal_3d: np.ndarray,
    terminal_2d: np.ndarray,
    grid_tree: cKDTree,
    grid_3d: np.ndarray,
    mesh_tree: cKDTree,
    zone_radius: float,
) -> np.ndarray:
    """Strip entry on the scalp. Near the clicked hub, blend toward that click."""
    _, index = grid_tree.query(np.asarray(entry_2d, dtype=float))
    end = np.asarray(grid_3d[int(index)], dtype=float)
    end = _nearest(mesh_tree, end)
    span = max(float(zone_radius) * 2.5, 1e-6)
    distance = float(np.linalg.norm(np.asarray(entry_2d, dtype=float) - np.asarray(terminal_2d, dtype=float)))
    blend = max(0.0, 1.0 - distance / span)
    if blend > 1e-6:
        mixed = (1.0 - blend) * end + blend * np.asarray(terminal_3d, dtype=float)
        end = _nearest(mesh_tree, mixed)
    return end


def reconstruct_path(
    electrode_3d: np.ndarray,
    entry_3d: np.ndarray,
    path_2d: np.ndarray,
    grid_tree: cKDTree,
    grid_3d: np.ndarray,
    mesh_tree: cKDTree,
) -> np.ndarray:
    """Inverse-distance blend of the four nearest grid vertices, then a mesh snap."""
    path_2d = np.asarray(path_2d, dtype=float)
    count = len(path_2d)
    out = np.empty((count, 3), dtype=float)
    if count == 0:
        return out
    out[0] = np.asarray(electrode_3d, dtype=float)
    if count == 1:
        return out
    out[-1] = np.asarray(entry_3d, dtype=float)
    if count == 2:
        return out
    distances, indices = grid_tree.query(path_2d[1:-1], k=min(4, len(grid_3d)))
    distances = np.asarray(distances, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    blended = np.einsum("ij,ijk->ik", weights, grid_3d[indices])
    _, snapped = mesh_tree.query(blended)
    out[1:-1] = mesh_tree.data[np.asarray(snapped, dtype=int)]
    return out


def lift_layout(
    layout: WireLayout,
    electrodes: dict[str, np.ndarray],
    terminals: dict[str, np.ndarray],
    cz: np.ndarray,
    mesh_points: np.ndarray,
    zone_radius: float,
) -> dict[str, np.ndarray]:
    """One scalp polyline per electrode, from the flat path after odd swap and even bends."""
    grid_2d, grid_3d = create_uv_grid(mesh_points, cz)
    grid_tree = cKDTree(grid_2d)
    mesh_tree = cKDTree(np.asarray(mesh_points, dtype=float))
    routes: dict[str, np.ndarray] = {}
    for name, path_2d in layout.paths_2d.items():
        terminal = layout.assignments[name]
        end = entry_on_surface(
            layout.entries_2d[name],
            terminals[terminal],
            layout.terminals_2d[terminal],
            grid_tree,
            grid_3d,
            mesh_tree,
            zone_radius,
        )
        routes[name] = reconstruct_path(
            electrodes[name], end, path_2d, grid_tree, grid_3d, mesh_tree
        )
    return routes
