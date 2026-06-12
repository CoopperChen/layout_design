"""Coordinate alignment: origin shift, rough rotation, scan2phys."""

from __future__ import annotations

import numpy as np

from ..models import JobConfig, SubjectBundle, TraceChannel
from ..transform.rotations import apply_rotation, roty, rotz
from ..transform.scan2phys import scan2phys
from app.postprocess.mesh_normals import head_center_from_points, orient_trace_xyzn


def _normalize_normals(trace: np.ndarray) -> np.ndarray:
    out = trace.copy()
    norms = out[:, 3:6]
    for i in range(norms.shape[0]):
        n = norms[i]
        norm = np.linalg.norm(n)
        if norm > 0:
            out[i, 3:6] = n / norm
    return out


def register_mesh_points(
    bundle: SubjectBundle,
    physical_landmarks_mm: np.ndarray,
    *,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
) -> np.ndarray:
    """Register bundle mesh vertices from scan frame into machine frame."""
    ps = bundle.landmarks_xyz.copy()
    pm = np.asarray(physical_landmarks_mm, dtype=float)

    mesh = bundle.mesh_points - ps[0]
    ps_aligned = ps - ps[0]

    rot = roty(rot0y_deg) @ rotz(rot0z_deg)
    mesh = apply_rotation(mesh, rot)
    ps_aligned = apply_rotation(ps_aligned, rot)

    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset
    return scan2phys(mesh + offset, ps_shifted, pm)


def align_subject(
    bundle: SubjectBundle,
    job: JobConfig,
) -> tuple[list[TraceChannel], np.ndarray]:
    """
    Apply origin shift, rough rotation, and scan2phys registration.

    Returns (aligned_channels, registered_mesh_points).
    """
    ps = bundle.landmarks_xyz.copy()
    pm = np.asarray(job.physical_landmarks_mm, dtype=float)

    channels: list[TraceChannel] = []
    for ch in bundle.channels:
        ic = ch.interconnect.copy()
        el = ch.electrode.copy()
        ic[:, :3] -= ps[0]
        el[:, :3] -= ps[0]
        ic = _normalize_normals(ic)
        el = _normalize_normals(el)
        channels.append(
            TraceChannel(
                name=ch.name,
                interconnect=ic,
                electrode=el,
                terminal=ch.terminal,
            )
        )

    rot = roty(job.rot0y_deg) @ rotz(job.rot0z_deg)
    ps_aligned = apply_rotation(ps - ps[0], rot)

    for ch in channels:
        ch.interconnect[:, :3] = apply_rotation(ch.interconnect[:, :3], rot)
        ch.interconnect[:, 3:6] = apply_rotation(ch.interconnect[:, 3:6], rot)
        ch.electrode[:, :3] = apply_rotation(ch.electrode[:, :3], rot)
        ch.electrode[:, 3:6] = apply_rotation(ch.electrode[:, 3:6], rot)

    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset

    mesh_registered = register_mesh_points(
        bundle,
        pm,
        rot0y_deg=job.rot0y_deg,
        rot0z_deg=job.rot0z_deg,
    )

    for ch in channels:
        ch.interconnect[:, :3] = scan2phys(ch.interconnect[:, :3] + offset, ps_shifted, pm)
        ch.interconnect[:, 3:6] = scan2phys(ch.interconnect[:, 3:6], ps_shifted, pm)
        ch.electrode[:, :3] = scan2phys(ch.electrode[:, :3] + offset, ps_shifted, pm)
        ch.electrode[:, 3:6] = scan2phys(ch.electrode[:, 3:6], ps_shifted, pm)

    head_center = head_center_from_points(mesh_registered)
    for ch in channels:
        ch.interconnect = orient_trace_xyzn(ch.interconnect, head_center)
        ch.electrode = orient_trace_xyzn(ch.electrode, head_center)

    return channels, mesh_registered
