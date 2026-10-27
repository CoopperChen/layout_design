"""PLY + textured OBJ → one upright head mesh."""

from __future__ import annotations

from pathlib import Path

from app import paths
from app.config_loader import mesh_config
from app.mesh.align import align_obj_to_stl
from app.mesh.import_sources import ensure_subject_sources
from app.mesh.islands import clear_islands
from app.mesh.reconstruct import reconstruct_from_ply


def _newer(output: Path, inputs: list[Path]) -> bool:
    if not output.is_file():
        return False
    stamp = output.stat().st_mtime
    return all(src.is_file() and stamp >= src.stat().st_mtime for src in inputs)


def run_mesh(
    subject_id: int,
    *,
    ply_path: Path | None = None,
    obj_path: Path | None = None,
    zip_path: Path | None = None,
    poisson_depth: int | None = None,
    preview: bool = True,
    rotate_head: bool = True,
    match_scale: bool = True,
) -> int:
    paths.ensure_subject_dir(subject_id)
    ply, obj = ensure_subject_sources(
        subject_id,
        ply_path=ply_path,
        obj_path=obj_path,
        zip_path=zip_path,
    )

    mesh_cfg = mesh_config()
    depth = int(mesh_cfg["poisson_depth"] if poisson_depth is None else poisson_depth)
    print(
        f"Mesh settings: depth {depth}, density quantile {mesh_cfg['poisson_density_quantile']}, "
        f"scale ×{mesh_cfg['scale_to_mm']:g}, "
        f"ICP ≤{mesh_cfg['icp_mean_dist_max_mm']:g} mm"
    )
    stl = paths.head_stl(subject_id)
    aligned = paths.head_obj(subject_id)
    sources = [ply, obj]
    if _newer(stl, sources) and _newer(aligned, sources):
        print(f"Mesh already current for subject {subject_id}")
        print(f"  {stl}")
        print(f"  {aligned}")
        return 0

    if not _newer(stl, [ply]):
        print(f"mesh subject {subject_id}: Poisson from {ply}")
        reconstruct_from_ply(
            ply,
            stl,
            align_head=False,
            poisson_depth=depth,
            scale_to_mm=float(mesh_cfg["scale_to_mm"]),
            poisson_density_quantile=float(mesh_cfg["poisson_density_quantile"]),
            outlier_neighbors=int(mesh_cfg["outlier_neighbors"]),
            outlier_std_ratio=float(mesh_cfg["outlier_std_ratio"]),
        )
        clear_islands(
            stl,
            min_size_ratio=float(mesh_cfg["island_min_size_ratio"]),
            min_absolute_size=int(mesh_cfg["island_min_cells"]),
        )
    else:
        print(f"Poisson mesh is current: {stl}")

    if not _newer(aligned, sources):
        print(f"mesh subject {subject_id}: align {obj} → {stl}")
        align_obj_to_stl(
            obj,
            stl,
            aligned,
            subject_id=subject_id,
            match_scale=match_scale,
            preview=preview,
            rotate_head=rotate_head,
            n_samples=int(mesh_cfg["icp_samples"]),
            max_correspondence_mm=float(mesh_cfg["icp_max_correspondence_mm"]),
            max_iteration=int(mesh_cfg["icp_max_iteration"]),
            fitness_min=float(mesh_cfg["icp_fitness_min"]),
            mean_dist_max_mm=float(mesh_cfg["icp_mean_dist_max_mm"]),
        )
    else:
        print(f"Aligned texture is current: {aligned}")
    return 0
