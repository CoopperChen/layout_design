"""Smoke test: PyVista loads real OBJ UV+JPG (not vague vertex-color bake)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_load_pyvista_textured_obj(tmp_path: Path):
    pytest.importorskip("pyvista")
    pytest.importorskip("open3d")
    import open3d as o3d
    import pyvista as pv

    from app.preprocess.obj_texture import load_pyvista_textured_obj

    obj = tmp_path / "head.obj"
    mtl = tmp_path / "head.mtl"
    jpg = tmp_path / "head.jpg"

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = 40
    img[..., 2] = 40
    o3d.io.write_image(str(jpg), o3d.geometry.Image(img))
    mtl.write_text("newmtl material_0\nKd 1 1 1\nmap_Kd head.jpg\n", encoding="utf-8")
    obj.write_text(
        "mtllib head.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl material_0\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
        encoding="utf-8",
    )

    mesh, tex = load_pyvista_textured_obj(obj)
    assert mesh.n_points >= 3
    assert mesh.active_texture_coordinates is not None
    assert tex is not None
    assert getattr(mesh, "texture", None) is tex
    assert isinstance(mesh, pv.PolyData)


def test_stl_transform_does_not_wipe_without_normals(tmp_path: Path):
    """Regression: Open3D binary STL write truncates to 0 bytes without normals."""
    pytest.importorskip("open3d")
    import open3d as o3d

    from app.preprocess.obj_texture import apply_transform_to_triangle_mesh_file

    stl = tmp_path / "head.stl"
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=10)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    assert o3d.io.write_triangle_mesh(str(stl), mesh, write_ascii=False)
    before = stl.stat().st_size
    assert before > 0

    wiped_probe = tmp_path / "probe.stl"
    bad = o3d.io.read_triangle_mesh(str(stl))
    bad.triangle_normals = o3d.utility.Vector3dVector()
    bad.vertex_normals = o3d.utility.Vector3dVector()
    # Demonstrate the Open3D trap (truncates on failure).
    ok = o3d.io.write_triangle_mesh(str(wiped_probe), bad, write_ascii=False)
    assert not ok
    assert wiped_probe.stat().st_size == 0

    T = np.eye(4)
    T[2, 3] = 5.0
    apply_transform_to_triangle_mesh_file(stl, T)
    assert stl.stat().st_size > 0
    restored = o3d.io.read_triangle_mesh(str(stl))
    assert not restored.is_empty()
    assert len(restored.vertices) > 0


def test_stl_transform_rejects_empty_file(tmp_path: Path):
    pytest.importorskip("open3d")
    from app.preprocess.obj_texture import apply_transform_to_triangle_mesh_file

    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        apply_transform_to_triangle_mesh_file(empty, np.eye(4))


def test_polydata_surface_centroid_ignores_unused_verts():
    """Unused OBJ verts must not pull the pivot to the crown."""
    pytest.importorskip("pyvista")
    import pyvista as pv

    from app.preprocess.obj_texture import polydata_surface_centroid

    # Unit square in XY at z=0 (area centroid ≈ (0.5, 0.5, 0)).
    # Plus a cloud of unused verts high on +Z (would bias a plain mean).
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 100.0],
            [0.4, 0.4, 101.0],
            [0.6, 0.6, 99.0],
        ],
        dtype=float,
    )
    faces = np.hstack(
        [
            [3, 0, 1, 2],
            [3, 0, 2, 3],
        ]
    )
    mesh = pv.PolyData(verts, faces)
    c = polydata_surface_centroid(mesh)
    assert np.allclose(c, [0.5, 0.5, 0.0], atol=1e-6)
    assert not np.allclose(c, verts.mean(axis=0), atol=1.0)
