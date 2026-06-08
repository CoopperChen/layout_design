"""Head mesh I/O: Open3D ↔ PyVista, VTK-compatible OBJ export."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required for textured OBJ I/O. Install with: pip install open3d"
        ) from e
    return o3d


def vertex_colors_sidecar(obj_path: Path) -> Path:
    return obj_path.with_name(f"{obj_path.stem}_vertex_colors.npy")


def open3d_to_pyvista(mesh):
    """Convert Open3D TriangleMesh to PyVista PolyData with optional RGB scalars."""
    import pyvista as pv

    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return pv.PolyData(verts)

    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(verts, faces)

    if mesh.has_vertex_colors():
        rgb = (np.asarray(mesh.vertex_colors, dtype=float) * 255.0).clip(0, 255)
        poly["RGB"] = rgb.astype(np.uint8)

    if mesh.has_vertex_normals():
        poly.point_data["Normals"] = np.asarray(mesh.vertex_normals, dtype=float)

    return poly


def load_open3d_mesh(mesh_path: Path):
    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")
    if not mesh.has_vertex_colors():
        sidecar = vertex_colors_sidecar(mesh_path)
        if sidecar.is_file():
            colors = np.load(sidecar)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def load_head_mesh(mesh_path: Path):
    """
    Load head mesh for fiducial picking.

    OBJ: Open3D (supports vertex colors; avoids VTK OBJ warnings).
    STL: PyVista.
    """
    import pyvista as pv

    path = Path(mesh_path)
    if path.suffix.lower() == ".obj":
        return open3d_to_pyvista(load_open3d_mesh(path))

    data = pv.read(str(path))
    if isinstance(data, pv.MultiBlock):
        combined = data.combine()
        return combined if combined is not None else data
    return data


def write_vtk_compatible_obj(mesh, obj_path: Path) -> None:
    """
    Write OBJ with ``v x y z`` lines only (VTK/PyVista-safe).

    Vertex colors are stored in ``{stem}_vertex_colors.npy`` for reload.
    """
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)

    with obj_path.open("w", encoding="utf-8") as fp:
        fp.write("# layout_design VTK-compatible OBJ\n")
        for x, y, z in verts:
            fp.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
        for a, b, c in tris:
            fp.write(f"f {a + 1} {b + 1} {c + 1}\n")

    if mesh.has_vertex_colors():
        np.save(vertex_colors_sidecar(obj_path), np.asarray(mesh.vertex_colors))
    elif vertex_colors_sidecar(obj_path).is_file():
        vertex_colors_sidecar(obj_path).unlink(missing_ok=True)
