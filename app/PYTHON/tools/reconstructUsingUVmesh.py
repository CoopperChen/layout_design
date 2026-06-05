import numpy as np
from scipy.spatial import KDTree

_MESH_KDTREE_CACHE = {}


def mesh_kdtree(mesh):
    """Reuse one KDTree per mesh object (STL load is expensive to re-index)."""
    key = id(mesh)
    cached = _MESH_KDTREE_CACHE.get(key)
    if cached is None:
        cached = KDTree(np.asarray(mesh.points))
        _MESH_KDTREE_CACHE[key] = cached
    return cached


class UVReconstructionContext:
    """Pre-built 2D/3D UV grid + mesh surface trees for batch path reconstruction."""

    def __init__(self, uv_grid, mesh):
        self.grid_2d = np.asarray(uv_grid['points_2d'], dtype=float)
        self.grid_3d = np.asarray(uv_grid['points_3d'], dtype=float)
        self.kdtree_2d = KDTree(self.grid_2d)
        self.mesh_kdtree = mesh_kdtree(mesh)

    def reconstruct(self, electrode_3d, terminal_3d, modified_path_2d):
        electrode_3d = np.asarray(electrode_3d, dtype=float)
        terminal_3d = np.asarray(terminal_3d, dtype=float)
        modified_path_2d = np.asarray(modified_path_2d, dtype=float)
        n = len(modified_path_2d)
        if n == 0:
            return np.empty((0, 3), dtype=float)

        reconstructed = np.empty((n, 3), dtype=float)
        reconstructed[0] = electrode_3d
        if n == 1:
            return reconstructed
        reconstructed[-1] = terminal_3d
        if n == 2:
            return reconstructed

        for i in range(1, n - 1):
            dists, indices = self.kdtree_2d.query(modified_path_2d[i], k=4)
            weights = 1.0 / (dists + 1e-8)
            weights /= weights.sum()
            interpolated_3d = np.zeros(3, dtype=float)
            for j, idx in enumerate(indices):
                interpolated_3d += weights[j] * self.grid_3d[idx]
            _, surface_idx = self.mesh_kdtree.query(interpolated_3d)
            reconstructed[i] = self.mesh_kdtree.data[surface_idx]

        return reconstructed


def reconstruct_with_uv_grid(
    electrode_3d,
    terminal_3d,
    modified_path_2d,
    uv_grid,
    mesh,
    context=None,
):
    """High-fidelity reconstruction using UV grid (build context once for many paths)."""
    if context is None:
        context = UVReconstructionContext(uv_grid, mesh)
    return context.reconstruct(electrode_3d, terminal_3d, modified_path_2d)
