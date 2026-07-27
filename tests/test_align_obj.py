"""Unit tests for align-obj ICP (imported textured OBJ → STL frame)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.preprocess.align_obj import (
    align_obj_to_stl,
    mean_surface_distance_mm,
    rough_similarity_align,
    run_icp,
)


def _require_open3d():
    pytest.importorskip("open3d")
    import open3d as o3d

    return o3d


def _colored_sphere(*, radius: float = 50.0, translate=None, scale: float = 1.0):
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    colors = np.linspace(0.0, 1.0, len(mesh.vertices))
    rgb = np.column_stack([colors, 1.0 - colors, np.full_like(colors, 0.4)])
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    if scale != 1.0:
        mesh.scale(scale, center=mesh.get_center())
    if translate is not None:
        mesh.translate(np.asarray(translate, dtype=float))
    return mesh


def test_rough_similarity_align_centers_and_scales():
    o3d = _require_open3d()
    target = _colored_sphere(radius=50.0)
    source = _colored_sphere(radius=50.0, scale=2.0, translate=[100.0, -40.0, 25.0])
    T = rough_similarity_align(source, target)
    source.transform(T)
    src = np.asarray(source.vertices)
    tgt = np.asarray(target.vertices)
    assert np.allclose(src.mean(axis=0), tgt.mean(axis=0), atol=1e-5)
    ext_src = float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))
    ext_tgt = float(np.linalg.norm(tgt.max(axis=0) - tgt.min(axis=0)))
    assert abs(ext_src - ext_tgt) / ext_tgt < 0.02


def test_icp_recovers_small_rigid_offset(tmp_path: Path):
    o3d = _require_open3d()
    target = _colored_sphere(radius=50.0)
    source = o3d.geometry.TriangleMesh(target)
    # Rotate ~15° about Z and translate
    R = source.get_rotation_matrix_from_xyz((0.0, 0.0, np.deg2rad(15.0)))
    source.rotate(R, center=source.get_center())
    source.translate([8.0, -5.0, 3.0])

    T, fitness = run_icp(
        source,
        target,
        n_samples=8000,
        max_correspondence_mm=25.0,
        max_iteration=100,
    )
    source.transform(T)
    mean_d = mean_surface_distance_mm(source, target, n_samples=5000)
    assert fitness > 0.5
    assert mean_d < 2.0


def test_align_obj_to_stl_writes_synced_obj(tmp_path: Path):
    o3d = _require_open3d()
    target = _colored_sphere(radius=50.0)
    source = o3d.geometry.TriangleMesh(target)
    source.translate([12.0, -7.0, 4.0])

    src_obj = tmp_path / "import.obj"
    stl = tmp_path / "target.stl"
    out_obj = tmp_path / "synced.obj"
    assert o3d.io.write_triangle_mesh(str(src_obj), source)
    assert o3d.io.write_triangle_mesh(str(stl), target)

    align_obj_to_stl(
        src_obj,
        stl,
        out_obj,
        match_scale=False,
        n_samples=8000,
        max_correspondence_mm=25.0,
        preview=False,
        fitness_min=0.2,
        mean_dist_max_mm=5.0,
    )
    assert out_obj.is_file()
    assert out_obj.with_name("synced_vertex_colors.npy").is_file()
    assert out_obj.with_name("synced_obj_to_stl.npy").is_file()

    synced = o3d.io.read_triangle_mesh(str(out_obj))
    mean_d = mean_surface_distance_mm(synced, target, n_samples=5000)
    assert mean_d < 1.5
