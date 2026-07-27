"""
Stage A — align an imported textured OBJ to the reconstructed/cleaned STL.

Fiducials are picked on the OBJ; layout uses the STL. After ICP, both share
the same coordinate frame so picks remain valid downstream.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app import paths
from app.runtime import setup_runtime


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "align-obj requires open3d. Install with: pip install open3d"
        ) from e
    return o3d


def _mesh_to_pcd(mesh, *, n_samples: int, estimate_normals: bool = True):
    o3d = _require_open3d()
    n = max(1000, int(n_samples))
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    if estimate_normals:
        if mesh.has_triangle_normals():
            # Prefer mesh normals via another sample with normals when available.
            pass
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
        )
    return pcd


def rough_similarity_align(source, target) -> np.ndarray:
    """
    Center-to-center translation + isotropic scale so extents match.

    Returns 4x4 transform applied to *source* (scale about source centroid,
    then translate to target centroid).
    """
    src = np.asarray(source.vertices, dtype=float)
    tgt = np.asarray(target.vertices, dtype=float)
    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("Cannot rough-align empty meshes")

    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)
    ext_src = float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))
    ext_tgt = float(np.linalg.norm(tgt.max(axis=0) - tgt.min(axis=0)))
    if ext_src < 1e-9:
        raise ValueError("Source mesh has zero extent")
    scale = ext_tgt / ext_src

    T = np.eye(4, dtype=float)
    # x' = scale * (x - c_src) + c_tgt
    T[:3, :3] = np.eye(3) * scale
    T[:3, 3] = c_tgt - scale * c_src
    return T


def run_icp(
    source_mesh,
    target_mesh,
    *,
    init: np.ndarray | None = None,
    n_samples: int = 50000,
    max_correspondence_mm: float = 15.0,
    max_iteration: int = 80,
) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP. Returns (4x4 transform, fitness)."""
    o3d = _require_open3d()
    src_pcd = _mesh_to_pcd(source_mesh, n_samples=n_samples)
    tgt_pcd = _mesh_to_pcd(target_mesh, n_samples=n_samples)
    if init is None:
        init = np.eye(4, dtype=float)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(max_iteration)
    )
    # Coarse → fine correspondence distances help large initial offsets.
    transform = np.asarray(init, dtype=float)
    last = None
    for dist in (max_correspondence_mm * 2.0, max_correspondence_mm, max_correspondence_mm * 0.5):
        last = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            float(dist),
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        transform = np.asarray(last.transformation, dtype=float)
    assert last is not None
    return transform, float(last.fitness)


def bake_texture_to_vertex_colors(mesh) -> bool:
    """
    Sample OBJ material textures onto vertices when vertex colors are missing.

    Returns True if the mesh ends with vertex colors.
    """
    o3d = _require_open3d()
    if mesh.has_vertex_colors():
        return True

    textures = list(getattr(mesh, "textures", []) or [])
    if not textures or not mesh.has_triangle_uvs():
        return False

    uvs = np.asarray(mesh.triangle_uvs, dtype=float).reshape(-1, 3, 2)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts = len(mesh.vertices)
    if len(uvs) != len(tris):
        return False

    # Use first texture image (typical single-material head scan).
    img = np.asarray(textures[0])
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w = img.shape[:2]

    accum = np.zeros((n_verts, 3), dtype=float)
    counts = np.zeros(n_verts, dtype=float)
    for ti, tri in enumerate(tris):
        for k in range(3):
            u, v = uvs[ti, k]
            # OBJ / Open3D: v often bottom-up; image rows top-down.
            x = int(np.clip(u, 0.0, 1.0) * (w - 1))
            y = int(np.clip(1.0 - v, 0.0, 1.0) * (h - 1))
            accum[tri[k]] += img[y, x] / 255.0
            counts[tri[k]] += 1.0

    missing = counts < 1.0
    counts[missing] = 1.0
    colors = accum / counts[:, None]
    colors[missing] = 0.7
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return True


def load_textured_source(obj_path: Path):
    """Load imported OBJ; bake UV texture → vertex colors when needed."""
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"Imported OBJ not found: {obj_path}")

    mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {obj_path}")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    had_vertex_colors = mesh.has_vertex_colors()
    baked = bake_texture_to_vertex_colors(mesh)
    if had_vertex_colors:
        print(f"Source colors: vertex colors ({len(mesh.vertices)} vertices)")
    elif baked and mesh.has_vertex_colors():
        print(f"Source colors: texture baked → {len(mesh.vertices)} vertices")
    else:
        print(
            "Warning: imported OBJ has no vertex colors or bakeable texture — "
            "fiducial picking will show a gray surface."
        )
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    return mesh


def load_target_stl(stl_path: Path):
    o3d = _require_open3d()
    stl_path = Path(stl_path)
    if not stl_path.is_file():
        raise FileNotFoundError(f"Target STL not found: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError(f"Empty STL: {stl_path}")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def mean_surface_distance_mm(source, target, *, n_samples: int = 20000) -> float:
    """Mean nearest-neighbor distance from sampled source points to target surface samples."""
    import scipy.spatial

    n = max(1000, int(n_samples))
    src = np.asarray(source.sample_points_uniformly(number_of_points=n).points, dtype=float)
    tgt = np.asarray(target.sample_points_uniformly(number_of_points=n).points, dtype=float)
    tree = scipy.spatial.cKDTree(tgt)
    dists, _ = tree.query(src, k=1)
    return float(np.mean(dists))


def interactive_alignment_preview(source, target) -> bool:
    """
    Overlay aligned OBJ (colored) on STL (gray). Space/Enter/S = accept, Q/Esc = reject.
    Closing the window accepts.
    """
    o3d = _require_open3d()
    state = {"accepted": True, "done": False}

    src = o3d.geometry.TriangleMesh(source)
    tgt = o3d.geometry.TriangleMesh(target)
    tgt.paint_uniform_color([0.55, 0.55, 0.55])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="align-obj preview (Space=accept, Q=reject)", width=1280, height=900)
    vis.add_geometry(tgt)
    vis.add_geometry(src)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True

    def _accept(_vis):
        state["accepted"] = True
        state["done"] = True
        _vis.close()
        return False

    def _reject(_vis):
        state["accepted"] = False
        state["done"] = True
        _vis.close()
        return False

    # GLFW: Space=32, Enter=257, S=83/115, Q=81/113, Esc=256
    for key in (32, 257, 83, 115):
        vis.register_key_callback(key, _accept)
    for key in (81, 113, 256):
        vis.register_key_callback(key, _reject)

    print("Preview: colored OBJ over gray STL. Space/Enter/S = accept · Q/Esc = reject")
    vis.run()
    vis.destroy_window()
    return bool(state["accepted"])


def align_obj_to_stl(
    source_obj: Path,
    target_stl: Path,
    out_obj: Path,
    *,
    match_scale: bool = True,
    n_samples: int = 50000,
    max_correspondence_mm: float = 15.0,
    max_iteration: int = 80,
    preview: bool = True,
    fitness_min: float = 0.3,
    mean_dist_max_mm: float = 3.0,
) -> Path:
    """
    Rigid(+optional scale) align imported OBJ → STL frame; write ``data/raw/{id}.obj``.
    """
    from app.preprocess.mesh_io import write_vtk_compatible_obj

    source = load_textured_source(source_obj)
    target = load_target_stl(target_stl)

    T = np.eye(4, dtype=float)
    if match_scale:
        T = rough_similarity_align(source, target)
        source.transform(T)
        print(
            f"Rough scale/center align applied "
            f"(scale≈{float(np.linalg.norm(T[:3, 0])):.4g})"
        )

    icp_T, fitness = run_icp(
        source,
        target,
        init=np.eye(4),
        n_samples=n_samples,
        max_correspondence_mm=max_correspondence_mm,
        max_iteration=max_iteration,
    )
    source.transform(icp_T)
    T = icp_T @ T

    mean_d = mean_surface_distance_mm(source, target)
    print(f"ICP fitness={fitness:.3f}, mean surface distance={mean_d:.3f} mm")
    if fitness < fitness_min or mean_d > mean_dist_max_mm:
        print(
            "Warning: alignment quality looks poor. "
            "Check that the OBJ and STL are the same head / units, "
            "or retry with --no-scale / larger --max-correspondence."
        )

    if preview:
        if not interactive_alignment_preview(source, target):
            raise RuntimeError("align-obj rejected in preview — nothing written")

    out_obj = Path(out_obj)
    write_vtk_compatible_obj(source, out_obj)
    # Persist the 4x4 for audit / re-apply
    xform_path = out_obj.with_name(f"{out_obj.stem}_obj_to_stl.npy")
    np.save(xform_path, T)
    print(f"Wrote synced OBJ → {out_obj}")
    print(f"Wrote transform  → {xform_path}")
    print("Next: fiducials (pick on OBJ; coordinates valid on cleaned STL).")
    return out_obj


def resolve_target_stl(subject_id: int) -> Path:
    cleaned = paths.cleaned_scan(subject_id)
    if cleaned.is_file():
        return cleaned
    raw = paths.raw_scan(subject_id, ext="stl")
    if raw.is_file():
        print(f"Note: using raw STL (run clear-islands first for canonical mesh): {raw}")
        return raw
    raise FileNotFoundError(
        f"No STL for subject {subject_id}. Run reconstruct (+ clear-islands).\n"
        f"  looked for {cleaned} and {raw}"
    )


def resolve_source_obj(subject_id: int, obj_path: Path | None) -> Path:
    if obj_path is not None:
        p = Path(obj_path)
        return p if p.is_absolute() else paths.REPO_ROOT / p
    default = paths.imported_textured_obj(subject_id)
    if default.is_file():
        return default
    raise FileNotFoundError(
        f"No imported textured OBJ for subject {subject_id}.\n"
        f"  Place file at {default}\n"
        f"  or pass --obj PATH"
    )


def run_align_obj(
    subject_id: int,
    *,
    obj_path: Path | None = None,
    match_scale: bool = True,
    preview: bool = True,
    n_samples: int | None = None,
    max_correspondence_mm: float | None = None,
) -> int:
    setup_runtime()
    from app.config_loader import preprocess_defaults

    prep = preprocess_defaults()
    align_cfg = prep.get("align_obj") or {}
    n_samp = int(
        n_samples
        if n_samples is not None
        else align_cfg.get("n_samples", 50000)
    )
    max_corr = float(
        max_correspondence_mm
        if max_correspondence_mm is not None
        else align_cfg.get("max_correspondence_mm", 15.0)
    )
    fitness_min = float(align_cfg.get("fitness_min", 0.3))
    mean_dist_max = float(align_cfg.get("mean_dist_max_mm", 3.0))
    do_preview = preview if preview is not None else bool(align_cfg.get("preview", True))
    do_scale = match_scale if match_scale is not None else bool(align_cfg.get("match_scale", True))

    source = resolve_source_obj(subject_id, obj_path)
    target = resolve_target_stl(subject_id)
    out = paths.raw_scan(subject_id, ext="obj")
    print(f"align-obj subject {subject_id}")
    print(f"  source OBJ: {source}")
    print(f"  target STL: {target}")
    print(f"  output OBJ: {out}")

    align_obj_to_stl(
        source,
        target,
        out,
        match_scale=do_scale,
        n_samples=n_samp,
        max_correspondence_mm=max_corr,
        preview=do_preview,
        fitness_min=fitness_min,
        mean_dist_max_mm=mean_dist_max,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Align imported textured OBJ to reconstructed STL"
    )
    p.add_argument("--subject", type=int, required=True)
    p.add_argument(
        "--obj",
        type=Path,
        default=None,
        help="Imported textured OBJ (default: data/raw/{id}.obj)",
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="Skip isotropic scale matching before ICP",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip interactive overlay confirmation",
    )
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--max-correspondence", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_align_obj(
            args.subject,
            obj_path=args.obj,
            match_scale=not args.no_scale,
            preview=not args.no_preview,
            n_samples=args.n_samples,
            max_correspondence_mm=args.max_correspondence,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
