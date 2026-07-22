"""
Stage A — PLY point cloud → textured OBJ + STL (Poisson reconstruction).

Adapted from point-cloud reconstruction workflow: statistical outlier removal,
interactive normal flip, Poisson meshing, hole fill, color transfer, optional
head alignment. Outputs paired ``data/raw/{id}.obj`` and ``data/raw/{id}.stl``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.spatial

from app import paths
from app.runtime import setup_runtime


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "Poisson reconstruction requires open3d. Install with: pip install open3d"
        ) from e
    return o3d


def flip_mesh_winding(mesh) -> None:
    o3d = _require_open3d()
    triangles = np.asarray(mesh.triangles)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[:, [0, 2, 1]])


def configure_renderer_lighting(visualizer) -> None:
    opt = visualizer.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True
    opt.point_size = 5.0
    opt.line_width = 1.0


def display_inlier_outlier(cloud, indices) -> None:
    o3d = _require_open3d()
    inlier_cloud = cloud.select_by_index(indices)
    outlier_cloud = cloud.select_by_index(indices, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    print(
        "\nOutlier preview: close the window to continue "
        "(review only — no keys).\n"
    )
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        window_name="Outlier removal (gray=inlier, red=outlier)",
    )


# GLFW key codes used by Open3D VisualizerWithKeyCallback
_KEY_SPACE = 32
_KEY_ENTER = 257
_KEY_ESCAPE = 256


def interactive_normal_flip_viewer(geometry):
    """
    F — toggle normals; Space/Enter/S — confirm; Esc/Q — skip flip adjustment.
    Closing the window confirms the current flip state.
    Returns True if normals were flipped and confirmed.
    """
    o3d = _require_open3d()
    flip_state = {"flipped": False, "should_skip": False}
    geometry_copy = geometry

    def on_key_press(vis, key, action, _mods):
        if action != 2:
            return False
        if key in (ord("f"), ord("F")):
            flip_state["flipped"] = not flip_state["flipped"]
            if isinstance(geometry_copy, o3d.geometry.PointCloud):
                np.asarray(geometry_copy.normals)[:] *= -1
            elif isinstance(geometry_copy, o3d.geometry.TriangleMesh):
                flip_mesh_winding(geometry_copy)
                geometry_copy.compute_triangle_normals()
                geometry_copy.compute_vertex_normals()
            print(f"Normals {'flipped' if flip_state['flipped'] else 'restored'}")
            vis.update_geometry(geometry_copy)
            return False
        if key in (_KEY_ENTER, _KEY_SPACE, ord("s"), ord("S")):
            vis.destroy_window()
            return False
        if key in (_KEY_ESCAPE, ord("q"), ord("Q")):
            flip_state["should_skip"] = True
            vis.destroy_window()
            return False
        return False

    print(
        "\nNormal flip viewer:\n"
        "  F = toggle normals\n"
        "  Space / Enter / S (or close) = confirm\n"
        "  Esc / Q = skip adjustment\n"
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Normal Flip Viewer")
    vis.add_geometry(geometry_copy)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
    vis.add_geometry(coord_frame)
    opt = vis.get_render_option()
    opt.point_show_normal = True
    configure_renderer_lighting(vis)
    key_codes = (
        ord("f"),
        ord("F"),
        _KEY_ENTER,
        _KEY_SPACE,
        ord("s"),
        ord("S"),
        _KEY_ESCAPE,
        ord("q"),
        ord("Q"),
    )
    for code in key_codes:
        vis.register_key_callback(code, lambda v, k=code: on_key_press(v, k, 2, 0))
    vis.run()

    if flip_state["should_skip"] and flip_state["flipped"]:
        if isinstance(geometry_copy, o3d.geometry.PointCloud):
            np.asarray(geometry_copy.normals)[:] *= -1
        elif isinstance(geometry_copy, o3d.geometry.TriangleMesh):
            flip_mesh_winding(geometry_copy)
            geometry_copy.compute_triangle_normals()
            geometry_copy.compute_vertex_normals()
        print("Normal adjustment skipped — restored original normals.")
        return False
    if flip_state["flipped"]:
        print("Normals flipped and applied.")
    return flip_state["flipped"]


def transfer_colors_to_mesh(mesh, source) -> None:
    o3d = _require_open3d()
    if isinstance(source, o3d.geometry.PointCloud):
        if not source.has_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.ones((len(mesh.vertices), 3))
            )
            return
        source_points = np.asarray(source.points)
        source_colors = np.asarray(source.colors)
    elif isinstance(source, o3d.geometry.TriangleMesh):
        if not source.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.ones((len(mesh.vertices), 3))
            )
            return
        source_points = np.asarray(source.vertices)
        source_colors = np.asarray(source.vertex_colors)
    else:
        raise TypeError("Source must be PointCloud or TriangleMesh")

    mesh_vertices = np.asarray(mesh.vertices)
    tree = scipy.spatial.cKDTree(source_points)
    _, indices = tree.query(mesh_vertices, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(source_colors[indices])
    print(f"Transferred colors to {len(mesh.vertices)} mesh vertices")


def center_geometry_at_origin(geometry) -> None:
    if isinstance(geometry, _require_open3d().geometry.PointCloud):
        centroid = geometry.get_center()
        geometry.translate(-centroid)
    elif isinstance(geometry, _require_open3d().geometry.TriangleMesh):
        centroid = np.asarray(geometry.vertices).mean(axis=0)
        geometry.translate(-centroid)
    print(f"Centered at origin (was at {centroid})")


def _add_negative_axis_cylinders(vis, o3d, *, cyl_radius: float = 3, cyl_height: float = 200) -> None:
    """Extend coordinate frame with negative X/Y/Z axis cylinders."""
    x_neg = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=cyl_height)
    x_neg.paint_uniform_color([1, 0, 0])
    x_neg.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]), center=[0, 0, 0])
    x_neg.translate([-cyl_height / 2, 0, 0])
    vis.add_geometry(x_neg)

    y_neg = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=cyl_height)
    y_neg.paint_uniform_color([0, 1, 0])
    y_neg.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]), center=[0, 0, 0])
    y_neg.translate([0, -cyl_height / 2, 0])
    vis.add_geometry(y_neg)

    z_neg = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=cyl_height)
    z_neg.paint_uniform_color([0, 0, 1])
    z_neg.translate([0, 0, -cyl_height / 2])
    vis.add_geometry(z_neg)


def interactive_head_rotation_viewer(mesh) -> bool:
    """Manual head alignment with plane-view shortcuts and full axis guides."""
    o3d = _require_open3d()
    original_vertices = np.asarray(mesh.vertices).copy()
    centroid = original_vertices.mean(axis=0)
    rotation_state = {"confirmed": False, "canceled": False, "rotation_matrix": np.eye(3)}

    def rot_x(angle):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rot_y(angle):
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rot_z(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    def set_camera(vis, front, up, label: str) -> bool:
        ctrl = vis.get_view_control()
        ctrl.set_front(front)
        ctrl.set_up(up)
        ctrl.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        print(f"Camera: {label}")
        return False

    def on_key_press(vis, key, action, _mods):
        if action != 2:
            return False
        step = 2.5 * np.pi / 180
        new_rot = None
        if key == 263:
            new_rot = rot_z(step)
            print("Rotating LEFT (roll +2.5°)")
        elif key == 262:
            new_rot = rot_z(-step)
            print("Rotating RIGHT (roll -2.5°)")
        elif key == 265:
            new_rot = rot_x(step)
            print("Rotating UP (pitch +2.5°)")
        elif key == 264:
            new_rot = rot_x(-step)
            print("Rotating DOWN (pitch -2.5°)")
        elif key in (ord("a"), ord("A")):
            new_rot = rot_y(-step)
            print("Rotating LEFT (yaw -2.5°)")
        elif key in (ord("d"), ord("D")):
            new_rot = rot_y(step)
            print("Rotating RIGHT (yaw +2.5°)")
        elif key == ord("1"):
            return set_camera(vis, [0, 0, -1], [0, -1, 0], "XY plane (top)")
        elif key == ord("2"):
            return set_camera(vis, [-1, 0, 0], [0, 0, 1], "YZ plane (side)")
        elif key == ord("3"):
            return set_camera(vis, [0, -1, 0], [0, 0, 1], "XZ plane (front)")
        elif key == ord("4"):
            return set_camera(vis, [1, 0, 0], [0, 0, 1], "-X axis (back)")
        elif key == ord("5"):
            return set_camera(vis, [0, 1, 0], [0, 0, 1], "-Y axis (back side)")
        elif key == ord("6"):
            return set_camera(vis, [0, 0, 1], [0, -1, 0], "-Z axis (bottom)")
        elif key in (_KEY_ENTER, _KEY_SPACE, ord("s"), ord("S")):
            rotation_state["confirmed"] = True
            vis.destroy_window()
            print("Confirmed — keeping current orientation")
            return False
        elif key in (_KEY_ESCAPE, ord("q"), ord("Q")):
            rotation_state["canceled"] = True
            vis.destroy_window()
            print("Canceled — restoring original orientation")
            return False
        if new_rot is not None:
            rotation_state["rotation_matrix"] = new_rot @ rotation_state["rotation_matrix"]
            rotated = (original_vertices - centroid) @ rotation_state["rotation_matrix"].T + centroid
            mesh.vertices = o3d.utility.Vector3dVector(rotated)
            if mesh.has_vertex_normals():
                vn = np.asarray(mesh.vertex_normals) @ rotation_state["rotation_matrix"].T
                mesh.vertex_normals = o3d.utility.Vector3dVector(vn)
            if mesh.has_triangle_normals():
                tn = np.asarray(mesh.triangle_normals) @ rotation_state["rotation_matrix"].T
                mesh.triangle_normals = o3d.utility.Vector3dVector(tn)
            vis.update_geometry(mesh)
        return False

    print("\n" + "=" * 72)
    print("HEAD ROTATION VIEWER")
    print("  Mouse: drag rotate | scroll zoom")
    print("  Arrows: roll/pitch ±2.5° | A/D: yaw ±2.5°")
    print("  1 top (XY) | 2 side (YZ) | 3 front (XZ)")
    print("  4 back (-X) | 5 back side (-Y) | 6 bottom (-Z)")
    print("  Space / Enter / S (or close) = confirm")
    print("  Esc / Q = cancel (restore original)")
    print("  Axes: +X red, +Y green, +Z blue; negative axes shown as cylinders")
    print("=" * 72 + "\n")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Head Rotation Viewer")
    vis.add_geometry(mesh)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=200))
    _add_negative_axis_cylinders(vis, o3d)
    configure_renderer_lighting(vis)

    key_codes = (
        262, 263, 264, 265,
        ord("a"), ord("A"), ord("d"), ord("D"),
        ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"),
        _KEY_ENTER, _KEY_SPACE, ord("s"), ord("S"),
        _KEY_ESCAPE, ord("q"), ord("Q"),
    )
    for code in key_codes:
        vis.register_key_callback(code, lambda v, k=code: on_key_press(v, k, 2, 0))
    vis.run()

    if rotation_state["canceled"]:
        mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return False
    # Closing the window without a key = accept current orientation
    if not rotation_state["confirmed"]:
        print("Window closed — keeping current orientation")
    return True


def reconstruct_from_ply(
    ply_path: Path,
    out_stl: Path,
    out_obj: Path,
    *,
    align_head: bool = True,
    poisson_depth: int = 12,
    scale_to_mm: float = 1000.0,
) -> tuple[Path, Path]:
    o3d = _require_open3d()
    ply_path = Path(ply_path)
    if not ply_path.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {ply_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {ply_path}")

    if scale_to_mm != 1.0:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * scale_to_mm)
        print(f"Scaled point cloud by {scale_to_mm}× (meters → millimeters)")

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    print(f"Outlier removal: kept {len(ind)} / {len(pcd.points)} points")
    display_inlier_outlier(pcd, ind)

    cl.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cl.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
    )
    cl.orient_normals_consistent_tangent_plane(
        k=30, lambda_penalty=10.0, cos_alpha_tol=0.5
    )

    print("Examining point cloud normals (interactive)...")
    should_flip = interactive_normal_flip_viewer(cl)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
        mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cl, depth=poisson_depth
        )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    if should_flip:
        flip_mesh_winding(mesh)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tmesh.fill_holes()
    filled_legacy = filled.to_legacy()
    filled_legacy.compute_triangle_normals()
    filled_legacy.compute_vertex_normals()
    if should_flip:
        flip_mesh_winding(filled_legacy)
        filled_legacy.compute_triangle_normals()
        filled_legacy.compute_vertex_normals()

    transfer_colors_to_mesh(filled_legacy, cl)
    center_geometry_at_origin(filled_legacy)

    if align_head:
        interactive_head_rotation_viewer(filled_legacy)

    out_stl = Path(out_stl)
    out_obj = Path(out_obj)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_triangle_mesh(str(out_stl), filled_legacy):
        raise RuntimeError(f"Failed to write STL: {out_stl}")

    from app.preprocess.mesh_io import write_vtk_compatible_obj

    write_vtk_compatible_obj(filled_legacy, out_obj)

    print(f"Wrote STL → {out_stl}")
    print(f"Wrote OBJ → {out_obj}")
    return out_stl, out_obj


def run_reconstruct(
    subject_id: int,
    *,
    ply_path: Path | None = None,
    align_head: bool = True,
    poisson_depth: int = 12,
) -> int:
    setup_runtime()
    ply = Path(ply_path) if ply_path else paths.raw_point_cloud(subject_id)
    if not ply.is_file():
        raw_dir = paths.DATA_DIR / "raw"
        available = sorted(p.name for p in raw_dir.glob("*.ply")) if raw_dir.is_dir() else []
        hint = (
            f"  Place {paths.raw_point_cloud(subject_id).name} under data/raw/\n"
            f"  or pass --ply PATH"
        )
        if available:
            hint += f"\n  PLY files found in data/raw/: {', '.join(available)}"
        raise FileNotFoundError(f"Missing input PLY: {ply}\n{hint}")
    out_stl = paths.raw_scan(subject_id, ext="stl")
    out_obj = paths.raw_scan(subject_id, ext="obj")
    reconstruct_from_ply(
        ply,
        out_stl,
        out_obj,
        align_head=align_head,
        poisson_depth=poisson_depth,
    )
    print(
        "Next: clear-islands (STL) → fiducials (OBJ). "
        "Both meshes share the same geometry."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PLY → Poisson mesh (OBJ + STL)")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--ply", type=Path, default=None, help="Override input .ply path")
    p.add_argument("--no-align-head", action="store_true", help="Skip head rotation UI")
    p.add_argument("--depth", type=int, default=12, help="Poisson octree depth")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_reconstruct(
            args.subject,
            ply_path=args.ply,
            align_head=not args.no_align_head,
            poisson_depth=args.depth,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())