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
    from app.preprocess.mesh_io import bake_texture_to_vertex_colors as _bake

    return _bake(mesh)


def load_textured_source(obj_path: Path):
    """Load imported OBJ (+ MTL/JPG); bake UV texture → vertex colors when needed."""
    from app.preprocess.mesh_io import load_textured_obj_mesh

    return load_textured_obj_mesh(Path(obj_path))


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


def interactive_alignment_preview(source_obj: Path, target_stl: Path) -> bool:
    """
    Overlay textured OBJ (real JPG) on gray STL. Space/Enter/S = accept, Q/Esc = reject.
    """
    import pyvista as pv

    from app.preprocess.mesh_io import load_head_mesh, obj_has_texcoords
    from app.preprocess.obj_texture import load_pyvista_textured_obj

    state = {"accepted": True}

    stl = pv.read(str(target_stl))
    if isinstance(stl, pv.MultiBlock):
        stl = stl.combine()

    plotter = pv.Plotter(window_size=(1400, 1000))
    plotter.set_background("white")
    plotter.add_mesh(
        stl,
        color="lightgray",
        opacity=0.35,
        smooth_shading=True,
        name="stl",
    )
    if obj_has_texcoords(source_obj):
        obj_mesh, tex = load_pyvista_textured_obj(source_obj)
        plotter.add_mesh(
            obj_mesh,
            texture=tex,
            smooth_shading=True,
            show_edges=False,
            name="obj",
        )
    else:
        obj_mesh = load_head_mesh(source_obj)
        if "RGB" in obj_mesh.array_names:
            plotter.add_mesh(
                obj_mesh,
                scalars="RGB",
                rgb=True,
                smooth_shading=True,
                lighting=False,
                show_edges=False,
                name="obj",
            )
        else:
            plotter.add_mesh(
                obj_mesh,
                color="tomato",
                opacity=0.7,
                smooth_shading=True,
                name="obj",
            )
    plotter.add_axes()

    def _accept() -> None:
        state["accepted"] = True
        plotter.close()

    def _reject() -> None:
        state["accepted"] = False
        plotter.close()

    plotter.add_key_event("space", _accept)
    plotter.add_key_event("Return", _accept)
    plotter.add_key_event("s", _accept)
    plotter.add_key_event("S", _accept)
    plotter.add_key_event("q", _reject)
    plotter.add_key_event("Q", _reject)
    plotter.add_key_event("Escape", _reject)

    print("Preview: textured OBJ over gray STL. Space/Enter/S = accept · Q/Esc = reject")
    plotter.show()
    return bool(state["accepted"])


def align_obj_to_stl(
    source_obj: Path,
    target_stl: Path,
    out_obj: Path,
    *,
    subject_id: int | None = None,
    match_scale: bool = True,
    n_samples: int = 50000,
    max_correspondence_mm: float = 15.0,
    max_iteration: int = 80,
    preview: bool = True,
    rotate_head: bool = True,
    fitness_min: float = 0.3,
    mean_dist_max_mm: float = 3.0,
) -> Path:
    """
    ICP-align textured OBJ → STL frame, optional head rotation on the OBJ,
    keep STL synced with the same transforms.
    """
    from app.preprocess.mesh_io import (
        obj_has_texcoords,
        write_transformed_wavefront_obj,
        write_vtk_compatible_obj,
    )
    from app.preprocess.obj_texture import (
        interactive_textured_head_rotation,
        sync_stl_files_with_transform,
    )

    source_obj = Path(source_obj)
    from app.preprocess.mesh_io import backup_textured_obj_if_needed

    # Only snapshot the source when align would overwrite it in place.
    if Path(out_obj).resolve() == Path(source_obj).resolve():
        backup_textured_obj_if_needed(source_obj)
    source_text_path = source_obj
    preserve_materials = obj_has_texcoords(source_obj)
    if not preserve_materials:
        companions = []
        try:
            from app.preprocess.mesh_io import discover_obj_texture_images

            companions = discover_obj_texture_images(source_obj)
        except Exception:  # noqa: BLE001
            pass
        if companions:
            raise ValueError(
                f"Cannot align {source_obj.name}: it has no UVs, but "
                f"{', '.join(p.name for p in companions)} is present.\n"
                f"Replace the OBJ with the textured one from your "
                f".obj/.mtl/.jpg package (must contain 'vt' lines)."
            )

    # Open3D mesh is geometry-only for ICP (not for display).
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

    out_obj = Path(out_obj)
    # Write ICP-aligned textured OBJ first (needed for textured preview / rotation).
    if preserve_materials:
        write_transformed_wavefront_obj(source_text_path, T, out_obj)
    else:
        write_vtk_compatible_obj(source, out_obj)

    if preview:
        if not interactive_alignment_preview(out_obj, target_stl):
            raise RuntimeError("align-obj rejected in preview — nothing written")

    # Head rotation on the *textured* OBJ; same transform applied to STL.
    if rotate_head and preserve_materials:
        T_rot = interactive_textured_head_rotation(out_obj)
        if T_rot is not None and not np.allclose(T_rot, np.eye(4)):
            write_transformed_wavefront_obj(out_obj, T_rot, out_obj)
            T = T_rot @ T
            if subject_id is not None:
                sync_stl_files_with_transform(T_rot, subject_id=subject_id)
            else:
                sync_stl_files_with_transform(T_rot, stl_paths=[Path(target_stl)])

    xform_path = out_obj.with_name(f"{out_obj.stem}_obj_to_stl.npy")
    np.save(xform_path, T)
    print(f"Wrote synced OBJ → {out_obj}")
    print(f"Wrote transform  → {xform_path}")
    print(
        "Next: fiducials (pick on aligned OBJ in cleaned_scans/; "
        "import raw OBJ was left unchanged)."
    )
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
    rotate_head: bool | None = None,
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
    do_rotate = (
        bool(rotate_head)
        if rotate_head is not None
        else bool(align_cfg.get("rotate_head", True))
    )

    source = resolve_source_obj(subject_id, obj_path)
    target = resolve_target_stl(subject_id)
    out = paths.aligned_textured_obj(subject_id)
    print(f"align-obj subject {subject_id}")
    print(f"  source OBJ (unchanged): {source}")
    print(f"  target STL: {target}")
    print(f"  output OBJ (synced):    {out}")

    align_obj_to_stl(
        source,
        target,
        out,
        subject_id=subject_id,
        match_scale=do_scale,
        n_samples=n_samp,
        max_correspondence_mm=max_corr,
        preview=do_preview,
        rotate_head=do_rotate,
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
    p.add_argument(
        "--no-rotate-head",
        action="store_true",
        help="Skip textured OBJ head-rotation UI (STL stays as-is after ICP)",
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
            rotate_head=not args.no_rotate_head,
            n_samples=args.n_samples,
            max_correspondence_mm=args.max_correspondence,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
