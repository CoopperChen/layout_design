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


def color_ref_path(obj_path: Path) -> Path:
    """Aligned vertex+color point cloud written with the OBJ (mesh coordinates)."""
    return obj_path.with_name(f"{obj_path.stem}_color_ref.ply")


def _colors_look_valid(colors: np.ndarray, n_verts: int) -> bool:
    if len(colors) != n_verts:
        return False
    if n_verts < 100:
        return True
    unique = len(np.unique(np.round(colors, 3), axis=0))
    min_unique = min(5000, max(200, n_verts // 100))
    return unique >= min_unique


def transfer_vertex_colors_from_points(
    mesh,
    source_points: np.ndarray,
    source_colors: np.ndarray,
) -> None:
    """Nearest-neighbor color transfer (same logic as reconstruct)."""
    import scipy.spatial

    o3d = _require_open3d()
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    source_points = np.asarray(source_points, dtype=float)
    source_colors = np.asarray(source_colors, dtype=float)
    if len(source_points) == 0 or len(source_colors) == 0:
        raise ValueError("Source point cloud has no points/colors")
    tree = scipy.spatial.cKDTree(source_points)
    _, indices = tree.query(mesh_vertices, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(source_colors[indices])


def transfer_vertex_colors_from_color_ref(mesh, ref_path: Path) -> bool:
    """Reload colors from mesh-aligned reference written at reconstruct time."""
    o3d = _require_open3d()
    ref_path = Path(ref_path)
    if not ref_path.is_file():
        return False

    pcd = o3d.io.read_point_cloud(str(ref_path))
    if pcd.is_empty() or not pcd.has_colors():
        return False

    ref_points = np.asarray(pcd.points, dtype=float)
    ref_colors = np.asarray(pcd.colors, dtype=float)
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)

    if len(ref_points) == len(mesh_vertices) and np.allclose(ref_points, mesh_vertices, atol=1e-3):
        mesh.vertex_colors = o3d.utility.Vector3dVector(ref_colors)
    else:
        transfer_vertex_colors_from_points(mesh, ref_points, ref_colors)

    return mesh.has_vertex_colors()


def save_color_reference(mesh, obj_path: Path) -> Path:
    """Persist vertex positions + colors in mesh space for reliable reload."""
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if not mesh.has_vertex_colors():
        raise ValueError(f"Mesh has no vertex colors to save for {obj_path.name}")

    ref_path = color_ref_path(obj_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float))
    pcd.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(str(ref_path), pcd)
    return ref_path


def attach_vertex_colors(mesh, obj_path: Path) -> bool:
    """
    Ensure an Open3D mesh has vertex colors for fiducial picking.

    Order: sidecar npy → aligned color_ref.ply (same frame as OBJ).
    """
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if mesh.has_vertex_colors():
        return True

    n_verts = len(mesh.vertices)
    sidecar = vertex_colors_sidecar(obj_path)
    if sidecar.is_file():
        colors = np.asarray(np.load(sidecar), dtype=float)
        if _colors_look_valid(colors, n_verts):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            if mesh.has_vertex_colors():
                return True
        reason = (
            f"count mismatch ({len(colors)} vs {n_verts})"
            if len(colors) != n_verts
            else "colors look washed out (stale sidecar)"
        )
        print(f"Warning: ignoring {sidecar.name} — {reason}.")

    ref_path = color_ref_path(obj_path)
    if transfer_vertex_colors_from_color_ref(mesh, ref_path):
        np.save(sidecar, np.asarray(mesh.vertex_colors, dtype=float))
        print(f"Restored vertex colors from {ref_path.name} → {sidecar.name}")
        return True

    print(
        f"Could not load vertex colors for {obj_path.name}. "
        f"Re-run reconstruct to regenerate OBJ colors:\n"
        f"  python -m app preprocess --subject {obj_path.stem} --step reconstruct"
    )
    return False


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
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")
    if mesh_path.suffix.lower() == ".obj":
        attach_vertex_colors(mesh, mesh_path)
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
    _require_open3d()
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

    sidecar = vertex_colors_sidecar(obj_path)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=float)
        if len(colors) != len(verts):
            raise ValueError(
                f"Vertex/color count mismatch when writing {obj_path.name}: "
                f"{len(verts)} vertices vs {len(colors)} colors"
            )
        np.save(sidecar, colors)
        save_color_reference(mesh, obj_path)
    elif sidecar.is_file():
        sidecar.unlink(missing_ok=True)
        color_ref_path(obj_path).unlink(missing_ok=True)
