"""Inverse machine-zero and tool-offset transforms for nozzle tip recovery."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from app.postprocess.gcode.kinematics.axis_angles import compute_axis_angles
from app.postprocess.gcode.models import MachineConfig
from app.postprocess.mesh_normals import head_center_from_points, orient_normals_outward

# Squared-degree tolerance for B/C recovery (~0.22 deg RMS).
_BC_TOL = 0.05
_INWARD_NORMAL_PENALTY = 500.0


def undo_machine_zero_offset(
    positions: np.ndarray, machine: MachineConfig
) -> np.ndarray:
    """Reverse apply_machine_zero_offset."""
    g = positions.copy()
    c0, b0 = machine.c0_deg, machine.b0_deg
    a, d, calgap_z = machine.a_mm, machine.d_mm, machine.calgap_z_mm

    if c0 == 0 and b0 == 0:
        g[:, 0] += a
        g[:, 2] -= d + calgap_z
    elif c0 == 90 and b0 == 0:
        g[:, 1] += a
        g[:, 2] += d + calgap_z
    elif c0 == 0 and b0 == 90:
        g[:, 1] += a + d + calgap_z
    else:
        raise ValueError(
            "Unsupported machine zero pose "
            f"(c0={c0}, b0={b0}); only (0,0), (90,0), (0,90) are supported"
        )
    return g


def undo_tool_offset(
    positions: np.ndarray,
    normals: np.ndarray,
    c_angles: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """Reverse apply_tool_offset."""
    g = positions.copy()
    t = machine.d_mm + machine.gap_size_mm
    a = machine.a_mm
    npts = g.shape[0]

    for i in range(npts):
        x_offset = normals[i, 0] * t - abs(np.cos(np.deg2rad(c_angles[i]))) * a
        g[i, 0] -= x_offset

        y_offset = normals[i, 1] * t + np.sin(np.deg2rad(c_angles[i])) * a
        g[i, 1] -= y_offset

        cross_p = np.cross(normals[i], [0, 0, 1])
        if np.linalg.norm(cross_p) < 0.01:
            total_offset = normals[i] * t
        else:
            total_offset = normals[i] * t - cross_p / np.linalg.norm(cross_p) * a
        g[i, 2] -= total_offset[2]

    return g


def _spherical_to_unit(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def _seed_to_spherical(seed: np.ndarray) -> tuple[float, float]:
    seed = seed / np.linalg.norm(seed)
    theta = float(np.arccos(np.clip(seed[2], -1.0, 1.0)))
    phi = float(np.arctan2(seed[1], seed[0]))
    return theta, phi


def _bc_loss(normal: np.ndarray, b_deg: float, c_deg: float) -> float:
    b_hat, c_hat = compute_axis_angles(normal.reshape(1, 3))
    return float((b_hat[0] - b_deg) ** 2 + (c_hat[0] - c_deg) ** 2)


def _optimize_normal(b_deg: float, c_deg: float, seed: np.ndarray) -> np.ndarray | None:
    theta0, phi0 = _seed_to_spherical(seed)

    def loss(x: np.ndarray) -> float:
        return _bc_loss(_spherical_to_unit(x[0], x[1]), b_deg, c_deg)

    result = minimize(loss, [theta0, phi0], method="Nelder-Mead", tol=1e-8)
    if not result.success and result.fun > _BC_TOL:
        return None
    n = _spherical_to_unit(result.x[0], result.x[1])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    if _bc_loss(n, b_deg, c_deg) > _BC_TOL:
        return None
    return n


def _dedupe_normals(candidates: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for n in candidates:
        n = n / np.linalg.norm(n)
        if any(np.dot(n, existing) > 0.995 for existing in unique):
            continue
        unique.append(n)
    return unique


def normal_candidates_from_bc(
    b_deg: float,
    c_deg: float,
    *,
    extra_seeds: list[np.ndarray] | None = None,
    bc_cache: dict[tuple[float, float], list[np.ndarray]] | None = None,
) -> list[np.ndarray]:
    """Collect unit normals that reproduce commanded B/C (multiple branches)."""
    key = (round(b_deg, 2), round(c_deg, 2))
    if bc_cache is not None and key in bc_cache:
        return list(bc_cache[key])

    seeds: list[np.ndarray] = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    if extra_seeds:
        seeds.extend(extra_seeds)

    candidates: list[np.ndarray] = []
    for seed in seeds:
        if np.linalg.norm(seed) < 1e-12:
            continue
        for trial in (seed, -seed):
            n = _optimize_normal(b_deg, c_deg, trial)
            if n is not None:
                candidates.append(n)
    candidates = _dedupe_normals(candidates)
    if bc_cache is not None:
        bc_cache[key] = list(candidates)
    return candidates


def _nozzle_position(
    gcode_xyz: np.ndarray,
    normal: np.ndarray,
    c_deg: float,
    machine: MachineConfig,
) -> np.ndarray:
    pos = gcode_xyz.reshape(1, 3)
    pos = undo_tool_offset(pos, normal.reshape(1, 3), np.array([c_deg]), machine)
    return undo_machine_zero_offset(pos, machine)[0]


def _is_outward_normal(
    normal: np.ndarray, surface_point: np.ndarray, head_center: np.ndarray | None
) -> bool:
    if head_center is None:
        return True
    radial = np.asarray(surface_point, dtype=float).reshape(3) - np.asarray(
        head_center, dtype=float
    ).reshape(3)
    norm = np.linalg.norm(radial)
    if norm < 1e-12:
        return True
    return float(np.dot(normal, radial / norm)) > 0.0


def _inward_normal_penalty(
    normal: np.ndarray, surface_point: np.ndarray, head_center: np.ndarray | None
) -> float:
    if _is_outward_normal(normal, surface_point, head_center):
        return 0.0
    return _INWARD_NORMAL_PENALTY


def _filter_outward_candidates(
    candidates: list[np.ndarray],
    gcode_xyz: np.ndarray,
    b_deg: float,
    c_deg: float,
    machine: MachineConfig,
    *,
    head_center: np.ndarray | None,
) -> list[np.ndarray]:
    if head_center is None:
        return candidates
    outward: list[np.ndarray] = []
    for normal in candidates:
        surface = _nozzle_position(gcode_xyz, normal, c_deg, machine)
        if _is_outward_normal(normal, surface, head_center):
            outward.append(normal)
    return outward


def _mesh_outward_normals_at(
    surface_xyz: np.ndarray,
    mesh_tree: cKDTree,
    mesh_points: np.ndarray,
    mesh_normals: np.ndarray,
    head_center: np.ndarray,
) -> np.ndarray:
    """Outward mesh normal at the vertex nearest each surface point."""
    _dists, idxs = mesh_tree.query(np.asarray(surface_xyz, dtype=float))
    if np.isscalar(idxs):
        idxs = np.array([int(idxs)])
    out = np.zeros((len(idxs), 3), dtype=float)
    for i, idx in enumerate(np.atleast_1d(idxs)):
        idx = int(idx)
        pt = mesh_points[idx]
        out[i] = orient_normals_outward(
            pt.reshape(1, 3),
            mesh_normals[idx].reshape(1, 3),
            head_center,
        )[0]
    return out


def _mesh_bc_candidates(
    b_deg: float,
    c_deg: float,
    rough_nozzle: np.ndarray,
    mesh_tree: cKDTree,
    mesh_normals: np.ndarray,
    mesh_points: np.ndarray,
    head_center: np.ndarray,
    *,
    k: int = 40,
) -> list[np.ndarray]:
    _dists, idxs = mesh_tree.query(rough_nozzle, k=min(k, mesh_normals.shape[0]))
    if np.isscalar(idxs):
        idxs = [int(idxs)]

    candidates: list[np.ndarray] = []
    for idx in idxs:
        idx = int(idx)
        pt = mesh_points[idx]
        n_mesh = orient_normals_outward(
            pt.reshape(1, 3),
            mesh_normals[idx].reshape(1, 3),
            head_center,
        )[0]
        n = n_mesh / max(np.linalg.norm(n_mesh), 1e-12)
        if _bc_loss(n, b_deg, c_deg) <= _BC_TOL:
            candidates.append(n)
    return _dedupe_normals(candidates)


def _continuity_penalty(normal: np.ndarray, prev_normal: np.ndarray | None) -> float:
    if prev_normal is None or np.linalg.norm(prev_normal) < 1e-12:
        return 0.0
    prev_unit = prev_normal / np.linalg.norm(prev_normal)
    dot = float(np.dot(normal, prev_unit))
    penalty = 2.0 * (1.0 - dot)
    if dot < 0.0:
        penalty += 5.0 * (-dot)
    return penalty


def _pick_normal(
    candidates: list[np.ndarray],
    gcode_xyz: np.ndarray,
    b_deg: float,
    c_deg: float,
    machine: MachineConfig,
    *,
    mesh_tree: cKDTree | None = None,
    prev_normal: np.ndarray | None = None,
    head_center: np.ndarray | None = None,
) -> np.ndarray:
    if not candidates:
        fallback = _optimize_normal(
            b_deg, c_deg, prev_normal or np.array([0.0, 0.0, 1.0])
        )
        if fallback is not None:
            return fallback
        raise ValueError(f"Cannot recover normal for B={b_deg}, C={c_deg}")

    filtered = _filter_outward_candidates(
        candidates, gcode_xyz, b_deg, c_deg, machine, head_center=head_center
    )
    if filtered:
        candidates = filtered

    best = candidates[0]
    best_score = float("inf")

    for normal in candidates:
        nozzle = _nozzle_position(gcode_xyz, normal, c_deg, machine)
        score = 0.0

        if mesh_tree is not None:
            dist, _idx = mesh_tree.query(nozzle)
            score += float(dist)

        score += _continuity_penalty(normal, prev_normal)
        score += _inward_normal_penalty(normal, nozzle, head_center)

        if score < best_score:
            best_score = score
            best = normal

    return best


def normal_from_bc(
    b_deg: float,
    c_deg: float,
    *,
    seed: np.ndarray | None = None,
    mesh_tree: cKDTree | None = None,
    mesh_normals: np.ndarray | None = None,
    mesh_points: np.ndarray | None = None,
    gcode_xyz: np.ndarray | None = None,
    machine: MachineConfig | None = None,
    head_center: np.ndarray | None = None,
    bc_cache: dict[tuple[float, float], list[np.ndarray]] | None = None,
) -> np.ndarray:
    """
    Recover surface normal for commanded B/C, disambiguated with mesh proximity.

    Note: G-code B/C may include correct_flip prefix negation from the postprocessor;
    mesh-assisted selection is required for reliable inversion on real files.
    """
    extra = [seed] if seed is not None and np.linalg.norm(seed) > 0 else None

    if (
        mesh_tree is not None
        and mesh_normals is not None
        and gcode_xyz is not None
        and machine is not None
    ):
        rough_seed = seed if seed is not None else np.array([0.0, 0.0, 1.0])
        rough = _optimize_normal(b_deg, c_deg, rough_seed)
        if rough is None:
            rough = rough_seed / max(np.linalg.norm(rough_seed), 1e-12)
        rough_nozzle = _nozzle_position(gcode_xyz, rough, c_deg, machine)
        candidates = _mesh_bc_candidates(
            b_deg,
            c_deg,
            rough_nozzle,
            mesh_tree,
            mesh_normals,
            mesh_points,
            head_center,
        )
        if not candidates:
            candidates = normal_candidates_from_bc(
                b_deg, c_deg, extra_seeds=extra, bc_cache=bc_cache
            )
        filtered = _filter_outward_candidates(
            candidates,
            gcode_xyz,
            b_deg,
            c_deg,
            machine,
            head_center=head_center,
        )
        if filtered:
            candidates = filtered
        normal = _pick_normal(
            candidates,
            gcode_xyz,
            b_deg,
            c_deg,
            machine,
            mesh_tree=mesh_tree,
            prev_normal=seed,
            head_center=head_center,
        )
        return normal

    candidates = normal_candidates_from_bc(
        b_deg, c_deg, extra_seeds=extra, bc_cache=bc_cache
    )
    if not candidates:
        fallback = _optimize_normal(b_deg, c_deg, seed or np.array([0.0, 0.0, 1.0]))
        if fallback is not None:
            return fallback
        raise ValueError(f"Cannot recover normal for B={b_deg}, C={c_deg}")
    if seed is not None and np.linalg.norm(seed) > 0:
        seed_u = seed / np.linalg.norm(seed)
        return max(candidates, key=lambda n: float(np.dot(n, seed_u)))
    return candidates[0]


def compute_mesh_vertex_normals(
    mesh_points: np.ndarray, mesh_faces: np.ndarray
) -> np.ndarray:
    """Outward-oriented vertex normals via Open3D."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_points, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_faces, dtype=int))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    head_center = head_center_from_points(mesh_points)
    return orient_normals_outward(mesh_points, normals, head_center)


def gcode_to_poses(
    gcode_matrix: np.ndarray,
    machine: MachineConfig,
    *,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert parsed G-code rows to programmed machine coords and nozzle tip poses.

    Returns (programmed_xyz, nozzle_tip_xyz, surface_normals), each Nx3 in the
    machine/physical frame. programmed_xyz is the raw X/Y/Z from the file.
    """
    if gcode_matrix.ndim == 1:
        gcode_matrix = gcode_matrix.reshape(1, -1)

    npts = gcode_matrix.shape[0]
    programmed_xyz = gcode_matrix[:, :3].copy()
    nozzle_xyz = np.zeros((npts, 3), dtype=float)
    normals = np.zeros((npts, 3), dtype=float)

    mesh_tree: cKDTree | None = None
    mesh_normals: np.ndarray | None = None
    head_center: np.ndarray | None = None
    if mesh_points is not None and mesh_faces is not None:
        mesh_tree = cKDTree(mesh_points)
        mesh_normals = compute_mesh_vertex_normals(mesh_points, mesh_faces)
        head_center = head_center_from_points(mesh_points)

    bc_cache: dict[tuple[float, float], list[np.ndarray]] = {}
    prev_normal: np.ndarray | None = None
    for i in range(npts):
        b_deg = float(gcode_matrix[i, 3])
        c_deg = float(gcode_matrix[i, 4])
        gcode_xyz = gcode_matrix[i, :3]

        normal = normal_from_bc(
            b_deg,
            c_deg,
            seed=prev_normal,
            mesh_tree=mesh_tree,
            mesh_normals=mesh_normals,
            mesh_points=mesh_points,
            gcode_xyz=gcode_xyz,
            machine=machine,
            head_center=head_center,
            bc_cache=bc_cache,
        )
        normals[i] = normal
        prev_normal = normal
        nozzle_xyz[i] = _nozzle_position(gcode_xyz, normal, c_deg, machine)

    return programmed_xyz, nozzle_xyz, normals


def nozzle_tip_print_positions(
    gcode_matrix: np.ndarray,
    machine: MachineConfig,
    *,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover printhead nozzle tip poses for simulation.

    Inverts the postprocessor (surface target on the scalp) then offsets outward
    along the surface normal by ``gap_size_mm`` so the tip path is not on the mesh.

    ``mesh_points`` must be in the same frame as ``convert-gcode`` registration
    (landmark frame: central landmark at ``pm[0]``). Use
    ``nozzle_tip_print_positions_machine_frame`` when the mesh is already in
    controller machine frame.

    Returns (programmed_xyz, nozzle_tip_xyz, surface_normals).
    """
    programmed_xyz, surface_xyz, normals = gcode_to_poses(
        gcode_matrix,
        machine,
        mesh_points=mesh_points,
        mesh_faces=mesh_faces,
    )
    gap = float(machine.gap_size_mm)
    if mesh_points is not None and mesh_faces is not None:
        head_center = head_center_from_points(mesh_points)
        mesh_tree = cKDTree(mesh_points)
        mesh_normals = compute_mesh_vertex_normals(mesh_points, mesh_faces)
        standoff_normals = _mesh_outward_normals_at(
            surface_xyz,
            mesh_tree,
            mesh_points,
            mesh_normals,
            head_center,
        )
        tip_xyz = surface_xyz + standoff_normals * gap
        return programmed_xyz, tip_xyz, standoff_normals
    tip_xyz = surface_xyz + normals * gap
    return programmed_xyz, tip_xyz, normals


def _machine_frame_kw(machine: MachineConfig) -> dict[str, float]:
    return {
        "a_mm": machine.a_mm,
        "d_mm": machine.d_mm,
        "calgap_z_mm": machine.calgap_z_mm,
    }


def decode_postprocessor_paths(
    gcode_matrix: np.ndarray,
    machine: MachineConfig,
    *,
    mesh_points_machine: np.ndarray,
    mesh_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Invert G-code in landmark frame, return scalp + print tips in machine frame.

    ``convert-gcode`` writes C-pivot commands from scalp traces registered with
    ``pm[0]`` at the origin. ``simulate-gcode`` displays mesh in controller machine
    frame; decode must use landmark mesh for inversion then shift results to match
    forward FK and the viewer mesh.

    Returns (scalp_machine, print_tips_machine, scalp_landmark, print_tips_landmark).
    """
    from app.postprocess.gcode.kinematics.machine_fk import (
        machine_to_registration_frame,
        registration_to_machine_frame,
    )

    fk_kw = _machine_frame_kw(machine)
    mesh_landmark = machine_to_registration_frame(mesh_points_machine, **fk_kw)

    _, scalp_lm, _ = gcode_to_poses(
        gcode_matrix,
        machine,
        mesh_points=mesh_landmark,
        mesh_faces=mesh_faces,
    )
    _, print_tips_lm, _ = nozzle_tip_print_positions(
        gcode_matrix,
        machine,
        mesh_points=mesh_landmark,
        mesh_faces=mesh_faces,
    )

    scalp_machine = registration_to_machine_frame(scalp_lm, **fk_kw)
    print_tips_machine = registration_to_machine_frame(print_tips_lm, **fk_kw)
    return scalp_machine, print_tips_machine, scalp_lm, print_tips_lm
