"""Register pad toolpaths and the head mesh into the print frame."""

from __future__ import annotations

import numpy as np

from ..models import PadTrace
from ..transform.rotations import apply_rotation, roty, rotz
from ..transform.scan2phys import scan2phys, scan2phys_direction


def _normalize_normals(trace: np.ndarray) -> np.ndarray:
    out = trace.copy()
    norms = out[:, 3:6]
    for i in range(norms.shape[0]):
        n = norms[i]
        length = np.linalg.norm(n)
        if length > 0:
            out[i, 3:6] = n / length
    return out


def register_mesh_points(
    mesh_points: np.ndarray,
    landmarks_xyz: np.ndarray,
    physical_landmarks_mm: np.ndarray,
    *,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
) -> np.ndarray:
    """Register mesh vertices from the scan frame into the landmark frame."""
    ps = np.asarray(landmarks_xyz, dtype=float).copy()
    pm = np.asarray(physical_landmarks_mm, dtype=float)
    mesh = np.asarray(mesh_points, dtype=float) - ps[0]
    ps_aligned = ps - ps[0]

    rot = roty(rot0y_deg) @ rotz(rot0z_deg)
    mesh = apply_rotation(mesh, rot)
    ps_aligned = apply_rotation(ps_aligned, rot)

    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset
    return scan2phys(mesh + offset, ps_shifted, pm)


def align_pads(
    pads: list[PadTrace],
    mesh_points: np.ndarray,
    landmarks_xyz: np.ndarray,
    physical_landmarks_mm: np.ndarray,
    *,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
) -> tuple[list[PadTrace], np.ndarray]:
    """Origin-shift, optional bed rotation, and scan2phys for pads and mesh."""
    ps = np.asarray(landmarks_xyz, dtype=float).copy()
    pm = np.asarray(physical_landmarks_mm, dtype=float)

    aligned: list[PadTrace] = []
    for pad in pads:
        xyzn = pad.xyzn.copy()
        xyzn[:, :3] -= ps[0]
        aligned.append(PadTrace(name=pad.name, xyzn=_normalize_normals(xyzn)))

    rot = roty(rot0y_deg) @ rotz(rot0z_deg)
    ps_aligned = apply_rotation(ps - ps[0], rot)
    for pad in aligned:
        pad.xyzn[:, :3] = apply_rotation(pad.xyzn[:, :3], rot)
        pad.xyzn[:, 3:6] = apply_rotation(pad.xyzn[:, 3:6], rot)

    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset
    mesh_registered = register_mesh_points(
        mesh_points,
        ps,
        pm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    for pad in aligned:
        pad.xyzn[:, :3] = scan2phys(pad.xyzn[:, :3] + offset, ps_shifted, pm)
        pad.xyzn[:, 3:6] = scan2phys_direction(pad.xyzn[:, 3:6], ps_shifted, pm)
    return aligned, mesh_registered
