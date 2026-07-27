"""Poisson reconstruct helpers."""
from __future__ import annotations

import numpy as np
import pytest


def _require_open3d():
    pytest.importorskip("open3d")
    import open3d as o3d

    return o3d


def test_trim_poisson_by_density_keeps_mesh():
    """Open3D remove_vertices_by_mask returns None — must not wipe the mesh."""
    o3d = _require_open3d()
    from app.preprocess.reconstruct import trim_poisson_by_density

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    dens = np.linspace(0.0, 1.0, len(mesh.vertices))
    out = trim_poisson_by_density(mesh, dens, quantile=0.05)
    assert out is not None
    assert isinstance(out, o3d.geometry.TriangleMesh)
    assert not out.is_empty()
    out.compute_triangle_normals()
