"""Overlap a printed PLY/OBJ on the planned head. The planned electrodes stay put."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config_loader import mesh_config
from app.mesh.align import (
    _mesh_to_pcd,
    _require_open3d,
    load_target_stl,
    mean_surface_distance_mm,
    orientation_init_candidates,
)

# Extent ratios inside this band stay unscaled, so a small print shrink remains in the error.
_SCALE_BAND = 0.05


@dataclass(frozen=True)
class Overlap:
    transform: np.ndarray
    scale: float
    fitness: float
    mean_dist_mm: float
    init: str
    source: str


def initial_similarity(source_points, target_points, *, match_scale: bool) -> tuple[np.ndarray, float]:
    """Center the source on the target. Scale only when the extents disagree.

    Returns the 4×4 that maps source points, and the scale written into that matrix.
    """
    src = np.asarray(source_points, dtype=float).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=float).reshape(-1, 3)
    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("Cannot overlap empty geometry")
    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)
    ext_src = float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))
    ext_tgt = float(np.linalg.norm(tgt.max(axis=0) - tgt.min(axis=0)))
    if ext_src < 1e-9:
        raise ValueError("Printed geometry has zero extent")
    ratio = ext_tgt / ext_src
    scale = ratio if match_scale and abs(ratio - 1.0) > _SCALE_BAND else 1.0
    transform = np.eye(4, dtype=float)
    transform[:3, :3] *= scale
    transform[:3, 3] = c_tgt - scale * c_src
    return transform, float(scale)


def align_geometry(
    source,
    target,
    *,
    match_scale: bool,
    n_samples: int,
    max_correspondence_mm: float,
    max_iteration: int,
    label: str = "ICP",
) -> tuple[np.ndarray, float, str, float]:
    """Rigid ICP after centering. Returns ``(transform, fitness, init, mean_dist_mm)``.

    The transform maps the original source onto the target. Reflections are not tried,
    so left and right stay on their own sides.
    """
    rough, scale = initial_similarity(_points(source), _points(target), match_scale=match_scale)
    print(f"  {label}: initial scale {scale:.4g}")
    src = _to_cloud(source, rough, n_samples)
    tgt = _to_cloud(target, np.eye(4), n_samples)
    delta, fitness, init, mean_d = _icp_multistart(
        src,
        tgt,
        max_correspondence_mm=max_correspondence_mm,
        max_iteration=max_iteration,
        label=label,
    )
    return delta @ rough, fitness, init, mean_d


def overlap_printed(
    obj_path: Path,
    ply_path: Path,
    stl_path: Path,
    *,
    match_scale: bool = True,
    n_samples: int | None = None,
    max_correspondence_mm: float | None = None,
    max_iteration: int | None = None,
) -> Overlap:
    """Map the printed OBJ into the planned ``head.stl`` frame.

    The printed scan is the geometric target: the texture is fit to the scan, then
    the scan is fit to the planned head. When that pair lands loosely, the textured
    OBJ is fit to the planned head on its own and the tighter result is kept.
    """
    cfg = mesh_config()
    samples = int(cfg["icp_samples"] if n_samples is None else n_samples)
    max_corr = float(
        cfg["icp_max_correspondence_mm"] if max_correspondence_mm is None else max_correspondence_mm
    )
    iterations = int(cfg["icp_max_iteration"] if max_iteration is None else max_iteration)
    mean_max = float(cfg["icp_mean_dist_max_mm"])
    fitness_min = float(cfg["icp_fitness_min"])

    planned = load_target_stl(stl_path)
    obj = _load_triangle_mesh(obj_path)
    if obj is None:
        raise ValueError(f"Printed OBJ has no triangles: {obj_path}")
    printed = _load_printed(ply_path)
    kw = dict(
        match_scale=match_scale,
        n_samples=samples,
        max_correspondence_mm=max_corr,
        max_iteration=iterations,
    )

    print(f"Overlap printed head onto {Path(stl_path).name}")
    chosen = _chain_overlap(obj, printed, planned, **kw)
    if chosen.mean_dist_mm > mean_max:
        print(
            f"Printed scan overlap is {chosen.mean_dist_mm:.2f} mm. "
            "Overlapping the textured OBJ on the planned head."
        )
        direct = _direct_overlap(obj, planned, **kw)
        if direct.mean_dist_mm < chosen.mean_dist_mm:
            chosen = direct

    if chosen.fitness < fitness_min or chosen.mean_dist_mm > mean_max:
        print(
            "Warning: overlap quality looks poor "
            f"(fitness {chosen.fitness:.3f}, mean surface {chosen.mean_dist_mm:.3f} mm). "
            "Check that the printed PLY and OBJ are this same head."
        )
    print(
        f"Overlap source={chosen.source}  scale={chosen.scale:.4g}  "
        f"fitness={chosen.fitness:.3f}  mean_surface={chosen.mean_dist_mm:.3f} mm  "
        f"init={chosen.init}"
    )
    return chosen


def _chain_overlap(obj, printed, planned, **kw) -> Overlap:
    print("Textured OBJ → printed scan")
    tex_T, tex_fit, tex_init, tex_dist = align_geometry(obj, printed, label="texture", **kw)
    print(
        f"  texture on scan: fitness={tex_fit:.3f}  mean={tex_dist:.3f} mm  init={tex_init}"
    )
    print("Printed scan → planned head")
    scan_T, scan_fit, scan_init, _scan_dist = align_geometry(printed, planned, label="scan", **kw)
    transform = scan_T @ tex_T
    score = _composed_distance(obj, planned, transform, n_samples=kw["n_samples"])
    return _pack(transform, scan_fit, scan_init, score, "printed scan")


def _direct_overlap(obj, planned, **kw) -> Overlap:
    print("Textured OBJ → planned head")
    transform, fitness, init, _dist = align_geometry(obj, planned, label="obj", **kw)
    score = _composed_distance(obj, planned, transform, n_samples=kw["n_samples"])
    return _pack(transform, fitness, init, score, "printed texture")


def _pack(transform, fitness, init, mean_dist_mm, source: str) -> Overlap:
    matrix = np.asarray(transform, dtype=float)
    linear = matrix[:3, :3]
    scale = float(np.mean(np.linalg.norm(linear, axis=0)))
    return Overlap(
        transform=matrix,
        scale=scale,
        fitness=float(fitness),
        mean_dist_mm=float(mean_dist_mm),
        init=str(init),
        source=source,
    )


def _composed_distance(obj_mesh, planned_mesh, transform, *, n_samples: int) -> float:
    o3d = _require_open3d()
    moved = o3d.geometry.TriangleMesh(obj_mesh)
    moved.transform(np.asarray(transform, dtype=float))
    return mean_surface_distance_mm(moved, planned_mesh, n_samples=min(8000, int(n_samples)))


def _points(geom) -> np.ndarray:
    if hasattr(geom, "vertices"):
        return np.asarray(geom.vertices, dtype=float)
    return np.asarray(geom.points, dtype=float)


def _to_cloud(geom, transform, n_samples: int):
    o3d = _require_open3d()
    matrix = np.asarray(transform, dtype=float)
    if hasattr(geom, "vertices"):
        mesh = o3d.geometry.TriangleMesh(geom)
        mesh.transform(matrix)
        return _mesh_to_pcd(mesh, n_samples=n_samples)
    cloud = o3d.geometry.PointCloud(geom)
    cloud.transform(matrix)
    return _prepare_cloud(cloud, n_samples)


def _prepare_cloud(cloud, n_samples: int):
    count = len(np.asarray(cloud.points))
    keep = max(1000, int(n_samples))
    if count > keep:
        choice = np.random.default_rng(0).choice(count, size=keep, replace=False)
        cloud = cloud.select_by_index(choice.astype(np.int64).tolist())
    _orient_outward(cloud)
    if len(np.asarray(cloud.points)) < 30:
        raise ValueError("Printed scan does not have enough points to overlap")
    return cloud


def _orient_outward(cloud) -> None:
    o3d = _require_open3d()
    pts = np.asarray(cloud.points, dtype=float)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
    )
    cloud.orient_normals_towards_camera_location(pts.mean(axis=0))
    np.asarray(cloud.normals)[:] *= -1.0


def _rigid_inits(center: np.ndarray):
    for label, matrix in orientation_init_candidates(center):
        if float(np.linalg.det(np.asarray(matrix, dtype=float)[:3, :3])) < 0.0:
            continue
        yield label, matrix


def _icp_pair(source_pcd, target_pcd, init, *, max_correspondence_mm: float, max_iteration: int):
    o3d = _require_open3d()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iteration))
    transform = np.asarray(init, dtype=float)
    last = None
    for dist in (max_correspondence_mm * 2.0, max_correspondence_mm, max_correspondence_mm * 0.5):
        last = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            float(dist),
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        transform = np.asarray(last.transformation, dtype=float)
    assert last is not None
    return transform, float(last.fitness)


def _icp_multistart(
    source_pcd,
    target_pcd,
    *,
    max_correspondence_mm: float,
    max_iteration: int,
    label: str,
):
    o3d = _require_open3d()
    center = np.asarray(source_pcd.points, dtype=float).mean(axis=0)
    target_pts = np.asarray(target_pcd.points, dtype=float)
    best: tuple[np.ndarray, float, str, float] | None = None
    for init_label, init in _rigid_inits(center):
        transform, fitness = _icp_pair(
            source_pcd,
            target_pcd,
            init,
            max_correspondence_mm=max_correspondence_mm,
            max_iteration=max_iteration,
        )
        if not np.isfinite(transform).all():
            print(f"  {label} init={init_label:10s}  non-finite transform, skipped")
            continue
        moved = o3d.geometry.PointCloud(source_pcd)
        moved.transform(transform)
        mean_d = _nn_mean(np.asarray(moved.points, dtype=float), target_pts)
        print(
            f"  {label} init={init_label:10s}  fitness={fitness:.3f}  mean_dist={mean_d:.3f} mm"
        )
        if best is None or mean_d < best[3] - 1e-6 or (
            abs(mean_d - best[3]) <= 1e-6 and fitness > best[1]
        ):
            best = (transform, fitness, init_label, mean_d)
    if best is None:
        raise RuntimeError("Could not overlap the printed head on the planned head")
    print(f"Selected {label} init={best[2]} (mean_dist={best[3]:.3f} mm)")
    return best


def _nn_mean(source_points, target_points) -> float:
    import scipy.spatial

    tree = scipy.spatial.cKDTree(np.asarray(target_points, dtype=float))
    dists, _ = tree.query(np.asarray(source_points, dtype=float), k=1)
    return float(np.mean(dists))


def _load_triangle_mesh(path: Path):
    o3d = _require_open3d()
    path = Path(path)
    for post in (False, True):
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=post)
        if len(np.asarray(mesh.triangles)) > 0:
            mesh.compute_vertex_normals()
            return mesh
    return None


def _load_printed(path: Path):
    mesh = _load_triangle_mesh(path)
    if mesh is not None:
        return mesh
    o3d = _require_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Printed PLY has no geometry: {path}")
    return cloud
