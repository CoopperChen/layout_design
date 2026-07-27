"""
PyVista textured OBJ I/O and interactive head rotation (real UV + JPG).

Fiducials and align-obj head rotation must display the Wavefront texture atlas,
not Open3D vertex-color bakes (those look vague / washed out).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pyvista_textured_obj(obj_path: Path):
    """
    Load ``.obj`` + ``.mtl`` + ``.jpg`` for sharp textured display.

    Returns ``(polydata, texture)``. Attaches ``texture`` via
    ``pyvista.set_new_attribute`` for ``getattr(mesh, "texture", None)``
    (fiducials UI).
    """
    import pyvista as pv

    from app.preprocess.mesh_io import (
        discover_obj_texture_images,
        ensure_obj_material_paths,
        obj_has_texcoords,
    )

    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    ensure_obj_material_paths(obj_path)
    if not obj_has_texcoords(obj_path):
        raise ValueError(
            f"{obj_path.name} has no UV coordinates (vt). "
            f"Use the textured package OBJ (with .mtl + .jpg)."
        )

    mesh = pv.read(str(obj_path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    images = discover_obj_texture_images(obj_path)
    if not images:
        raise FileNotFoundError(
            f"No texture image next to {obj_path.name}. "
            f"Expected {obj_path.stem}.jpg (or map_Kd from the MTL)."
        )

    tex = pv.read_texture(str(images[0]))
    # Prefer explicit tcoords if VTK stored them under the material name.
    if mesh.active_texture_coordinates is None:
        for name in ("TCoords", "tcoords", "material_0"):
            if name in mesh.array_names:
                arr = np.asarray(mesh[name])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    mesh.active_texture_coordinates = arr[:, :2]
                    break
    if mesh.active_texture_coordinates is None:
        raise ValueError(
            f"{obj_path.name} loaded without texture coordinates — "
            f"cannot display JPG atlas."
        )

    # PyVista forbids arbitrary attrs; set_new_attribute is the supported hook.
    pv.set_new_attribute(mesh, "texture", tex)
    print(
        f"Loaded textured OBJ {obj_path.name}: "
        f"{mesh.n_points} pts, {mesh.n_cells} cells, texture={images[0].name}"
    )
    return mesh, tex


def rotation_about_point_matrix(R: np.ndarray, center: np.ndarray) -> np.ndarray:
    """4×4 rigid transform: ``x' = R @ (x - c) + c``."""
    R = np.asarray(R, dtype=float)
    c = np.asarray(center, dtype=float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T


def apply_transform_to_triangle_mesh_file(path: Path, transform: np.ndarray) -> None:
    """
    Apply a 4×4 transform to an STL/OBJ geometry file (Open3D).

    Writes via a temp file then replaces, and always computes normals before
    STL export. Open3D's binary STL writer truncates the destination to 0 bytes
    when normals are missing / write fails — never overwrite in place first.
    """
    import os
    import tempfile

    import open3d as o3d

    path = Path(path)
    if not path.is_file():
        return
    if path.stat().st_size <= 0:
        raise ValueError(
            f"Cannot transform empty mesh file: {path}\n"
            f"Re-run reconstruct / clear-islands for this subject."
        )
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(
            f"Cannot transform empty mesh: {path}\n"
            f"Re-run reconstruct / clear-islands for this subject."
        )
    mesh.transform(np.asarray(transform, dtype=float))
    if path.suffix.lower() == ".stl":
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        fd, tmp_name = tempfile.mkstemp(
            suffix=".stl", prefix=f".{path.stem}_", dir=str(path.parent)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            ok = o3d.io.write_triangle_mesh(
                str(tmp_path), mesh, write_ascii=False
            )
            if not ok or tmp_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"Open3D failed to write transformed STL (would have wiped "
                    f"{path}). Normals computed; check mesh integrity."
                )
            tmp_path.replace(path)
        finally:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)
    else:
        # Prefer Wavefront vertex rewrite for OBJ with materials.
        from app.preprocess.mesh_io import obj_has_texcoords, write_transformed_wavefront_obj

        if obj_has_texcoords(path):
            write_transformed_wavefront_obj(path, transform, path)
        else:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            ok = o3d.io.write_triangle_mesh(str(path), mesh)
            if not ok:
                raise RuntimeError(f"Open3D failed to write transformed mesh: {path}")


def sync_stl_files_with_transform(
    transform: np.ndarray,
    *,
    subject_id: int | None = None,
    stl_paths: list[Path] | None = None,
) -> list[Path]:
    """Apply the same rigid transform to raw + cleaned STL (keep OBJ/STL synced)."""
    from app import paths

    if stl_paths is None:
        if subject_id is None:
            raise ValueError("subject_id or stl_paths required")
        stl_paths = [
            paths.raw_scan(subject_id, ext="stl"),
            paths.cleaned_scan(subject_id, ext="stl"),
        ]
    updated: list[Path] = []
    for p in stl_paths:
        p = Path(p)
        if not p.is_file():
            continue
        apply_transform_to_triangle_mesh_file(p, transform)
        updated.append(p)
        print(f"Synced STL transform → {p}")
    return updated


def polydata_surface_centroid(mesh) -> np.ndarray:
    """
    Centroid of the *surface* (area-weighted triangle centers).

    Wavefront OBJs often list many unused ``v`` rows (UV atlas leftovers). A
    plain ``points.mean()`` is then pulled off the visible head — often toward
    the crown — so axes look stuck on top of the skull. Prefer faces only.
    """
    pts = np.asarray(mesh.points, dtype=float)
    if len(pts) == 0:
        raise ValueError("Cannot compute centroid of empty mesh")
    if getattr(mesh, "n_cells", 0) <= 0:
        return pts.mean(axis=0)

    # VTK face array: [n, i0, i1, ..., n, ...] — use triangles only.
    if bool(getattr(mesh, "is_all_triangles", False)):
        tris = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    else:
        try:
            tri = mesh.triangulate()
            tris = np.asarray(tri.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
            pts = np.asarray(tri.points, dtype=float)
        except Exception:  # noqa: BLE001
            # Fallback: mean of verts referenced by any face.
            cells = mesh.faces
            idxs = []
            i = 0
            while i < len(cells):
                n = int(cells[i])
                idxs.extend(int(x) for x in cells[i + 1 : i + 1 + n])
                i += 1 + n
            if not idxs:
                return pts.mean(axis=0)
            return pts[np.unique(np.asarray(idxs, dtype=np.int64))].mean(axis=0)

    if len(tris) == 0:
        return pts.mean(axis=0)

    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = float(areas.sum())
    if total < 1e-18:
        used = np.unique(tris.ravel())
        return pts[used].mean(axis=0)
    centers = (v0 + v1 + v2) / 3.0
    return (centers * areas[:, None]).sum(axis=0) / total


def add_world_xyz_arrows(
    plotter,
    origin: np.ndarray,
    *,
    length: float,
    name_prefix: str = "xyz",
) -> None:
    """
    Fixed world-frame XYZ arrows at ``origin`` (do not rotate with the head).

    +X/−X red, +Y/−Y green, +Z/−Z blue — same convention as reconstruct.
    """
    import pyvista as pv

    origin = np.asarray(origin, dtype=float).reshape(3)
    length = float(max(length, 1.0))

    axes = (
        ((1.0, 0.0, 0.0), "red", "+X"),
        ((-1.0, 0.0, 0.0), "red", "-X"),
        ((0.0, 1.0, 0.0), "lime", "+Y"),
        ((0.0, -1.0, 0.0), "lime", "-Y"),
        ((0.0, 0.0, 1.0), "blue", "+Z"),
        ((0.0, 0.0, -1.0), "blue", "-Z"),
    )
    label_pts = []
    labels = []
    for direction, color, label in axes:
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            tip_length=0.2,
            tip_radius=0.025,
            shaft_radius=0.008,
            scale=length,
        )
        plotter.add_mesh(
            arrow,
            color=color,
            smooth_shading=True,
            name=f"{name_prefix}_{label}",
        )
        label_pts.append(origin + np.asarray(direction, dtype=float) * (length * 1.06))
        labels.append(label)

    plotter.add_point_labels(
        np.asarray(label_pts, dtype=float),
        labels,
        font_size=14,
        text_color="black",
        point_size=0,
        shape=None,
        always_visible=True,
        name=f"{name_prefix}_labels",
    )


def interactive_textured_head_rotation(obj_path: Path) -> np.ndarray | None:
    """
    Rotate the textured OBJ interactively (real JPG atlas).

    Display is centered so the mesh centroid sits at the world origin with
    fixed ±XYZ arrows. Returns a 4×4 rotation about the *original* centroid
    (for applying to on-disk OBJ + STL). ``None`` if discarded.
    """
    import pyvista as pv

    mesh, tex = load_pyvista_textured_obj(obj_path)
    orig = np.asarray(mesh.points, dtype=float).copy()
    if len(orig) == 0:
        raise ValueError(f"Empty mesh for head rotation: {obj_path}")
    # Surface centroid (not mean of all OBJ verts — unused vt/v rows bias that).
    center = polydata_surface_centroid(mesh)
    print(
        f"Head rotation pivot: surface centroid {center.tolist()} "
        f"(vertex-mean would be {orig.mean(axis=0).tolist()})"
    )
    # Display geometry centered at the origin (centroid → 0).
    centered = orig - center
    mesh.points = centered
    # Axis length ~ 40% of bounding-box diagonal so arrows stay readable.
    extent = float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)))
    axis_len = max(extent * 0.35, 50.0)
    state = {
        "R": np.eye(3, dtype=float),
        "accepted": False,
        "discarded": False,
    }

    def _rot_x(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def _rot_y(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def _rot_z(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    plotter = pv.Plotter(window_size=(1400, 1000))
    plotter.set_background("white")
    actor = plotter.add_mesh(
        mesh,
        texture=tex,
        smooth_shading=True,
        show_edges=False,
        name="head",
    )
    _ = actor
    plotter.add_axes()
    add_world_xyz_arrows(plotter, np.zeros(3), length=axis_len)

    step = np.deg2rad(2.5)
    view_dist = max(extent * 1.6, 120.0)

    def _apply() -> None:
        # Rotate about origin in display frame (= mesh centroid in world).
        mesh.points = centered @ state["R"].T
        plotter.render()

    def _turn(R_delta: np.ndarray, label: str) -> None:
        state["R"] = R_delta @ state["R"]
        print(label)
        _apply()

    def _axis_view(axis: str) -> None:
        """Orthographic look at origin along ±X / ±Y / ±Z."""
        # Camera sits on the named axis looking toward the origin.
        positions = {
            "+X": (view_dist, 0.0, 0.0),
            "-X": (-view_dist, 0.0, 0.0),
            "+Y": (0.0, view_dist, 0.0),
            "-Y": (0.0, -view_dist, 0.0),
            "+Z": (0.0, 0.0, view_dist),
            "-Z": (0.0, 0.0, -view_dist),
        }
        # Keep +Z up for side views; for ±Z use +Y up.
        ups = {
            "+X": (0.0, 0.0, 1.0),
            "-X": (0.0, 0.0, 1.0),
            "+Y": (0.0, 0.0, 1.0),
            "-Y": (0.0, 0.0, 1.0),
            "+Z": (0.0, 1.0, 0.0),
            "-Z": (0.0, 1.0, 0.0),
        }
        plotter.camera.position = positions[axis]
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = ups[axis]
        plotter.reset_camera_clipping_range()
        plotter.render()
        print(f"Camera: {axis} view")

    plotter.add_key_event("Left", lambda: _turn(_rot_z(step), "Roll +2.5°"))
    plotter.add_key_event("Right", lambda: _turn(_rot_z(-step), "Roll -2.5°"))
    plotter.add_key_event("Up", lambda: _turn(_rot_x(step), "Pitch +2.5°"))
    plotter.add_key_event("Down", lambda: _turn(_rot_x(-step), "Pitch -2.5°"))
    plotter.add_key_event("a", lambda: _turn(_rot_y(step), "Yaw +2.5°"))
    plotter.add_key_event("d", lambda: _turn(_rot_y(-step), "Yaw -2.5°"))
    plotter.add_key_event("A", lambda: _turn(_rot_y(step), "Yaw +2.5°"))
    plotter.add_key_event("D", lambda: _turn(_rot_y(-step), "Yaw -2.5°"))

    plotter.add_key_event("1", lambda: _axis_view("+X"))
    plotter.add_key_event("2", lambda: _axis_view("-X"))
    plotter.add_key_event("3", lambda: _axis_view("+Y"))
    plotter.add_key_event("4", lambda: _axis_view("-Y"))
    plotter.add_key_event("5", lambda: _axis_view("+Z"))
    plotter.add_key_event("6", lambda: _axis_view("-Z"))

    def _accept() -> None:
        state["accepted"] = True
        plotter.close()

    def _discard() -> None:
        state["discarded"] = True
        plotter.close()

    plotter.add_key_event("space", _accept)
    plotter.add_key_event("Return", _accept)
    plotter.add_key_event("s", _accept)
    plotter.add_key_event("S", _accept)
    plotter.add_key_event("q", _discard)
    plotter.add_key_event("Q", _discard)
    plotter.add_key_event("Escape", _discard)

    print("\n" + "=" * 72)
    print("HEAD ROTATION — textured OBJ (JPG atlas)")
    print("  Display: head centroid at origin; ±XYZ world axes fixed")
    print("  Mouse: rotate/zoom view")
    print("  Arrows: roll/pitch ±2.5° | A/D: yaw ±2.5°")
    print("  1 +X | 2 -X | 3 +Y | 4 -Y | 5 +Z | 6 -Z")
    print("  Axes: ±X red, ±Y green, ±Z blue")
    print("  Space / Enter / S = confirm (same transform applied to STL)")
    print("  Q / Esc = discard rotation")
    print("=" * 72 + "\n")

    plotter.show()

    if state["discarded"]:
        print("Head rotation discarded")
        return None
    # Window close without Q = accept current orientation
    print("Head rotation confirmed")
    return rotation_about_point_matrix(state["R"], center)
