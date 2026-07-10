"""Head mesh I/O — vertex color sidecar and color_ref fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.preprocess.mesh_io import (
    _colors_look_valid,
    attach_vertex_colors,
    color_ref_path,
    load_head_mesh,
    open3d_to_pyvista,
    save_color_reference,
    transfer_vertex_colors_from_color_ref,
    transfer_vertex_colors_from_points,
    vertex_colors_sidecar,
    write_vtk_compatible_obj,
)


def _require_open3d():
    import open3d as o3d

    return o3d


def test_write_and_reload_vertex_colors(tmp_path: Path):
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=4)
    mesh.compute_vertex_normals()
    colors = np.random.default_rng(0).random((len(mesh.vertices), 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    obj_path = tmp_path / "head.obj"
    write_vtk_compatible_obj(mesh, obj_path)

    assert color_ref_path(obj_path).is_file()
    reloaded = load_head_mesh(obj_path)
    assert "RGB" in reloaded.array_names
    assert reloaded.n_points == len(mesh.vertices)


def test_stale_sidecar_falls_back_to_color_ref(tmp_path: Path):
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=4)
    mesh.compute_vertex_normals()
    rng = np.random.default_rng(1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(rng.random((len(mesh.vertices), 3)))

    obj_path = tmp_path / "5.obj"
    write_vtk_compatible_obj(mesh, obj_path)

    sidecar = vertex_colors_sidecar(obj_path)
    np.save(sidecar, np.full((len(mesh.vertices), 3), 0.5))

    o3d_mesh = o3d.io.read_triangle_mesh(str(obj_path))
    assert attach_vertex_colors(o3d_mesh, obj_path)
    pv_mesh = open3d_to_pyvista(o3d_mesh)
    assert "RGB" in pv_mesh.array_names
    restored = np.load(sidecar)
    assert _colors_look_valid(restored, len(mesh.vertices))


def test_colors_look_valid_rejects_washed_out():
    colors = np.full((10000, 3), 0.5)
    assert not _colors_look_valid(colors, 10000)
    varied = np.random.default_rng(0).random((10000, 3))
    assert _colors_look_valid(varied, 10000)


def test_transfer_vertex_colors_from_points():
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=3)
    source_pts = np.asarray(mesh.vertices, dtype=float)
    source_colors = np.tile([1.0, 0.0, 0.0], (len(source_pts), 1))

    transfer_vertex_colors_from_points(mesh, source_pts, source_colors)
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors)[0], [1.0, 0.0, 0.0])


def test_color_ref_roundtrip(tmp_path: Path):
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile([0.2, 0.6, 0.9], (len(mesh.vertices), 1))
    )
    obj_path = tmp_path / "9.obj"
    ref = save_color_reference(mesh, obj_path)

    blank = o3d.geometry.TriangleMesh()
    blank.vertices = mesh.vertices
    blank.triangles = mesh.triangles
    assert transfer_vertex_colors_from_color_ref(blank, ref)
    assert np.allclose(np.asarray(blank.vertex_colors)[0], [0.2, 0.6, 0.9], atol=0.01)
